# 13. Agentic RL 场景下的跨实例分布式 KV Cache

> 整理时间：2026-08-17  
> 基于工作区提交：`6f005e4da17dc9efdd929d89e21425a4eec990f0`  
> 主题：SGLang HiCache、Mooncake Store、PD Disaggregation、Cache-Aware Routing，以及它们在 agentic RL 多轮 rollout 中的关系。
>
> [https://vllm.ai/blog/2026-05-06-mooncake-store](https://vllm.ai/blog/2026-05-06-mooncake-store)

## 0. 一句话结论

SGLang 已经支持真正的跨实例全局 KV cache：

```text
HiCache L1 = 当前实例 GPU HBM
HiCache L2 = 当前实例 Host DRAM
HiCache L3 = 可由 Mooncake Store / HF3FS / NIXL / AIBrix KVCache 提供的集群共享存储
```

其中 L1/L2 是实例私有缓存，L3 可以在整个集群的 SGLang 实例之间共享。

对 agentic RL，推荐的完整组合是：

```text
Cache-aware routing
+ Prefill 侧 HiCache + Mooncake Store 全局 L3
+ PD 侧 Mooncake TransferEngine
+ Decode 侧异步写回新增 KV
+ 每次权重更新后的全局 L3 失效机制
```

最重要的概念边界是：

```text
Mooncake TransferEngine != Mooncake Store
P -> D KV transfer       != 跨请求、跨实例的持久化 prefix cache
```



### 0.1 使用 HiCache + Mooncake 实现跨实例分布式 KV

实现跨实例分布式 KV，需要让各个 SGLang worker 连接同一个 Mooncake Store，并把可复用的 prefix KV 主动写入共享 L3：

```bash
--enable-hierarchical-cache
--hicache-storage-backend mooncake
--hicache-write-policy write_through
```

三个缓存层的职责如下：

```text
HiCache L1：每个 worker 私有的 GPU KV pool，供 Attention 直接读取
HiCache L2：每个 worker 私有的 Host KV pool，负责本机卸载和恢复
HiCache L3：所有 worker 共享的 Mooncake 分布式 DRAM/SSD KV Store
```

Mooncake 的多个节点贡献 DRAM/SSD，组成由 Mooncake master 管理的集群级存储池。接入同一 master、tenant 和 cache namespace 的 SGLang worker，通过相同的 prefix/page hash 访问同一份共享 KV。

写入和复用路径如下：

```text
1. worker A 在 L1 中计算 prefix KV
2. page-aligned、可复用的 prefix KV 被插入 HiCache
3. write_through 异步执行 L1 -> L2 -> Mooncake L3
4. worker B 收到相同 prefix，并计算出相同的 page hash
5. worker B 查询并读取 Mooncake L3
6. KV 被加载到 worker B 自己的 L2/L1
7. worker B 跳过已经命中的 prefix，只 prefill 剩余 suffix
```

对应的数据流是：

```text
worker A L1
    -> worker A L2
    -> Mooncake distributed L3
    -> worker B L2
    -> worker B L1
    -> Attention
```

因此，请求第一轮由 worker A 执行、下一轮被调度到 worker B 时，只要共享 L3 写入已经完成，worker B 就可以复用第一轮形成的 prefix KV。这个过程不要求 worker B 持有 worker A 的本地 RadixTree，也不要求请求始终粘在同一个 worker 上。

`L3` 表示它位于本地 GPU/Host cache 之后，不表示它是单机缓存。Mooncake L3 提供的是集群共享、内容寻址的 KV 副本，以及跨实例恢复能力。

该方案实现的不是“全局统一 HBM KV 地址空间”。两种语义必须严格区分：


| 目标语义                                  | HiCache + Mooncake 是否提供 |
| ------------------------------------- | ----------------------- |
| 跨请求、跨实例按 prefix hash 共享 KV            | 是                       |
| 聚合多台机器 DRAM/SSD，形成全局 KV 容量池           | 是                       |
| 本地 KV 淘汰后从其他节点恢复而不重新 prefill          | 是                       |
| 请求换到另一个实例后加载共享 KV 并继续计算               | 是                       |
| Attention kernel 直接读取另一台机器 GPU 上的 KV  | 否                       |
| 所有 GPU 共享一个强一致、无需搬运的 HBM KV allocator | 否                       |
| L3 命中后无需在目标实例分配 GPU KV slot           | 否                       |


原因是 Attention 在后续每个 decode step、每一层都会反复读取历史 KV。如果每次 attention 都远程访问其他 GPU/节点，网络延迟和带宽成本通常不可接受。合理的数据路径是：

```text
远程共享 Store 命中
  -> 一次性把可复用 prefix KV 搬到目标实例
  -> 后续所有 decode step 从本地 HBM 读取
```

Mooncake 负责 **分布式共享副本和跨实例恢复**，SGLang 负责 **当前请求的本地执行副本**。L3 命中后仍需把 KV 加载到目标 worker 的本地 GPU KV slot，后续 Attention 才能持续从本地 HBM 高效读取。

使用 `write_through` 时，符合条件、page-aligned、已经进入 prefix cache 的 KV 会异步生成 L3 副本。它不会把正在计算的每个临时 KV slot 同步写入 Mooncake，因此跨实例可见性存在一个异步写入窗口。三种写策略及其可见时机在下一节展开。

### 0.2 L1/L2/L3 是独立物理层；是否主动进入 L3 由 write policy 决定

L1、L2、L3 首先是三个独立的物理存储池：

```text
L1：worker A 的 GPU KV pool
L2：worker A 的 Host KV pool
L3：Mooncake 的集群级分布式 DRAM/SSD pool
```

它们不是地址空间上的包含关系：

```text
不是：L1 内部包含 L2，L2 内部又包含 L3

而是：同一份逻辑 KV page 可以在一个或多个 tier 中各有一份物理副本
```

例如同一个 KV page 在不同时间可能处于以下状态：


| 时刻               | L1 GPU | L2 Host | L3 Mooncake | 含义                                  |
| ---------------- | ------ | ------- | ----------- | ----------------------------------- |
| 正在计算             | 有      | 无       | 无           | 只有当前 worker 能使用                     |
| write-through 完成 | 有      | 有       | 有           | 当前 worker 本地最快命中，其他 worker 可从 L3 恢复 |
| L1 eviction 后    | 无      | 有       | 有           | 当前 worker 从 L2 恢复，其他 worker从 L3 恢复  |
| L2 也 eviction 后  | 无      | 无       | 有           | 所有 worker 都需要从 L3 恢复                |
| L3 也 eviction 后  | 无      | 无       | 无           | 下次请求必须重新 prefill                    |


这里的核心控制项是：

```bash
--hicache-write-policy write_through|write_through_selective|write_back
```



#### `write_through`：本地空间够也主动复制到 L3

这是“主动全局分布式共享”需要的策略。

逻辑时序是：

```text
1. 模型在 worker A 的 L1 GPU 中生成 KV
2. page-aligned KV 被插入可复用 prefix cache
3. 不等待 L1/L2 内存不足，主动发起 L1 -> L2 copy
4. L1 -> L2 完成后，异步发起 L2 -> Mooncake L3 write
5. 原 KV 仍可留在 L1；L2/L3 保存副本
6. worker B 收到相同 prefix 后，查询 L3 并加载到自己的 L2/L1
```

因此即使：

```text
worker A 的 L1 + L2 容量完全足够
从来没有发生过 eviction
```

符合写入条件的 KV 仍然会主动写入 Mooncake。当前工作区的 `hicache_write_policy` 默认值就是 `write_through`。

这个结论可以概括为：

```text
write_through 的 L3 写入触发条件是“产生了可缓存 prefix”
而不是“本地 L1/L2 已经不够”
```



#### `write_through_selective`：第二次变热后主动复制到 L3

当前实现中，`write_through_selective` 的命中阈值是 2：

```text
第一次产生 prefix：主要保留在 L1，不主动生成 L2/L3 副本
prefix 再次被命中：达到热度阈值，再主动复制到 L2/L3
```

它适合存在大量一次性 rollout/trajectory 的场景，避免 Mooncake 被只使用一次的 KV 淹没。但它不能保证第一轮完成后，下一轮第一次换 worker 就一定能从 L3 命中。

#### `write_back`：本地空间够时可能一直不进入 L3

`write_back` 的逻辑是：

```text
KV 最初只留在 worker A 的 L1
L1 发生 eviction、需要下沉时，才写入 L2，并在启用 storage 时继续写 L3
```

所以如果 worker A 的 L1 一直没有 eviction：

```text
worker A：本地有 KV
Mooncake L3：可能没有
worker B：无法从 Mooncake 命中，只能重新 prefill
```

如果 L1 已经发生 eviction，即使 L2 空间仍然足够，当前 storage-enabled write-back 路径也会在完成 L1 -> L2 backup 后继续提交 L3。真正决定“是否主动产生全局副本”的差异是：`write_back` 要等本地 eviction 触发，而 `write_through` 不需要等待 eviction。

这正是为什么 `write_back` 不能满足“无论本地容量够不够，都尽快形成全局副本”的强需求。它主要用于降低远程写带宽，而不是最大化即时跨实例可见性。

#### 三种策略对分布式共享语义的影响


| 策略                        | 没有本地 eviction 时是否写 L3 | worker B 第一次跨实例复用     | 代价       |
| ------------------------- | --------------------- | --------------------- | -------- |
| `write_through`           | 是，符合条件后异步主动写          | 最有机会直接命中              | L3 写流量最大 |
| `write_through_selective` | 只有变热后写                | 首次迁移可能 miss           | 写流量较低    |
| `write_back`              | 通常不写，等待 eviction      | 本地未 eviction 前通常 miss | 写流量最低    |


对于本文的 agentic RL 目标：

```text
不同 turn 可能被发给不同 worker
希望第一轮结束后尽快让其他 worker 可复用
不希望依赖 worker A 先发生 L1/L2 eviction
```

推荐明确设置：

```bash
--enable-hierarchical-cache
--hicache-storage-backend mooncake
--hicache-write-policy write_through
```

但 `write_through` 仍是异步副本传播，不是强一致同步提交。以下窗口仍可能导致 worker B 暂时 miss：

```text
worker B 到达时，worker A 的 L2/L3 write 尚未完成
KV 尚未形成 page-aligned、可复用的 radix prefix
请求 abort，或者对应 KV 没有进入 cache-finished/insert 路径
Host/Mooncake 空间或写入操作失败
PD Decode 没有开启新增 KV 的 async offload
```

因此工程上的准确承诺是：

```text
write_through 提供主动、最终可见的跨实例 L3 副本
不是每产生一个 token 就同步、强一致地提交到 Mooncake
```



### 0.3 worker B 为什么能发现 worker A 写入的 KV

worker B 不需要复制 worker A 的本地 RadixTree。收到请求后，它会：

```text
1. 先在自己的本地 radix/HiCache metadata 中匹配 L1/L2
2. 对本地没有命中的后续 prefix page 计算相同的内容 hash
3. 用这些 hash 查询共享 Mooncake L3 是否存在
4. 如果存在，prefetch 到 worker B 的 Host pool
5. 再 load 到 worker B 的 GPU KV slot
6. 只计算 L3 也没有命中的 suffix token
```

只要两个 worker 的 token、模型版本、page/hash 语义和 namespace 一致，相同 prefix 就会产生相同的 L3 key。这就是不依赖 session sticky routing 的跨实例发现机制。

## 1. 先区分三种容易混淆的能力



### 1.1 实例内 Prefix Cache

SGLang 默认通过 RadixAttention / RadixCache 复用当前实例中已经计算过的 prefix KV。

```text
request A 在 worker 0 计算 prefix
request B 也发到 worker 0，并且 token prefix 相同
request B 命中 worker 0 的 GPU KV cache
```

这是最快的命中路径，但它只存在于 worker 0。请求 B 如果被发给 worker 1，worker 1 默认看不到 worker 0 的 HBM KV。

### 1.2 PD 请求级 KV Transfer

PD 分离后，同一个请求先在 Prefill server 上计算 prompt KV，然后必须把 KV 发送给 Decode server：

```text
client
  -> router
  -> prefill server 计算 prompt KV
  -> Mooncake/NIXL TransferEngine
  -> decode server 接收 KV 并继续生成
```

这是一条请求生命周期内的点对点传输通道。它解决的是：

```text
同一个请求的 KV 如何从 P 到 D
```

它本身不表示这些 KV 已经进入一个可供未来任意请求查询的全局缓存池。

### 1.3 跨请求、跨实例的全局 KV Store

全局 KV Store 以 token block/prefix hash 为 key，把 KV 保存到共享后端。这里的“L3”就是跨实例分布式 KV 的实现层，而不是单个实例自己的第三层缓存：

```text
worker 0 计算 prefix KV
  -> 写入 Mooncake Store

worker 1 收到相同 token prefix
  -> 查询 Mooncake Store
  -> 命中并把 KV 拉到自己的 Host/GPU
  -> 跳过对应 prefix 的重复 prefill
```

这才是本文所说的“全局分布式 KV cache”。

换言之：

```text
Mooncake 不接管正在执行的全部 GPU KV
不影响它接管全局共享、持久化、跨实例恢复这一层的数据
```

SGLang 对应配置是：

```bash
--enable-hierarchical-cache
--hicache-storage-backend mooncake
```

PD 传输对应的是另一组参数：

```bash
--disaggregation-mode prefill|decode
--disaggregation-transfer-backend mooncake
```

两者可以独立使用，也可以组合使用。

## 2. 为什么跨实例 KV 的案例经常和 PD 绑在一起

这不是功能上的硬依赖，主要是架构收益和实现历史造成的。

### 2.1 PD 天然必须跨 GPU/实例搬运 KV

Unified serving 中，prefill 和 decode 使用同一个 engine 的 KV pool；PD 中两者位于不同 engine，因此 P -> D 传输不可避免。

PD 已经需要解决：

```text
GPU memory registration
KV block metadata
目标 KV slot 预分配
TP/PP/CP rank 与布局映射
RDMA/IB/UCX 连接
异步传输与完成事件
请求异常和节点故障时的资源回收
```

在这套基础设施上再接一个共享 Store，比从普通单体 serving 中单独建设一套远程 KV I/O 路径更自然。因此公开案例通常先出现在 PD 文档里。

### 2.2 全局命中仍然需要把 KV 拉回目标实例

“全局 L3 命中”不等于“目标 GPU 已经持有这段 KV”。大致性能层级是：

```text
本实例 GPU/HBM 命中
  > 本实例 Host DRAM 命中
  > 远程 DRAM 命中
  > 远程 SSD 命中
  > 重新计算 Prefill
```

即使 Mooncake Store 命中，也仍有远程查询、RDMA 传输、Host/GPU slot 分配和同步成本。

因此普通多副本 serving 中，首先把相同会话路由回原实例，通常比随机选择一个实例再从 L3 拉 KV 更快。全局 L3 更像是：

```text
容量扩展 + 跨实例兜底 + 扩缩容/故障恢复 + 本地 cache eviction 后的恢复层
```



### 2.3 PD 的 producer/consumer 角色天然清晰

PD 中：

```text
Prefill = prompt KV producer
Decode  = prompt KV consumer
Decode  = 新生成 token KV 的 producer
下一轮 Prefill = 完整 trajectory KV 的 consumer
```

Store 很容易嵌入这个数据流，尤其适合多轮 agent。

### 2.4 很多文档把两种“跨实例 KV”统称为 KV Connector

Connector 是接口抽象，不代表所有 connector 都具有相同语义。一个 connector 可能只做 P2P transfer，另一个 connector 才连接共享 Store。

因此看到“Mooncake + KVConnector + PD”时，必须继续确认它用的是：

```text
TransferEngine/P2P connector
还是
Distributed Store connector
```



## 3. vLLM 与 SGLang 的对应关系



### 3.1 vLLM

当前 vLLM 把 Mooncake 的两类能力明确拆开：


| vLLM 组件                  | 作用                                            |
| ------------------------ | --------------------------------------------- |
| `MooncakeConnector`      | Prefill 与 Decode 之间的点对点 KV transfer           |
| `MooncakeStoreConnector` | 连接 MooncakeDistributedStore，提供跨请求、跨实例 KV pool |
| `MultiConnector`         | 在 PD 模式中组合 P2P connector 和 Store connector    |


普通非 PD 实例也可以独立使用 Store：

```bash
vllm serve "$MODEL_PATH" \
  --kv-transfer-config '{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both"
  }'
```

PD 模式下则使用 `MultiConnector` 同时完成：

```text
MooncakeConnector       -> 当前请求 P -> D
MooncakeStoreConnector  -> 跨请求、跨实例共享
```

因此只有 `MooncakeConnector + PD` 时，不能自动推导出“任意实例都能复用过去请求的 prefix KV”。

### 3.2 SGLang


| 目标                       | SGLang 配置                                        |
| ------------------------ | ------------------------------------------------ |
| 本地 Radix prefix cache    | 默认开启；`--disable-radix-cache` 可关闭                 |
| 层级缓存                     | `--enable-hierarchical-cache`                    |
| Mooncake 全局 L3           | `--hicache-storage-backend mooncake`             |
| PD P2P 传输                | `--disaggregation-transfer-backend mooncake`     |
| Decode 新增 KV 写回 L3       | `--disaggregation-decode-enable-offload-kvcache` |
| Prefill locality routing | Gateway `--prefill-policy cache_aware`           |


SGLang HiCache 的设计语义是：

```text
L1/L2 private per inference instance
L3 shared by inference instances in the cluster
```

HiRadixTree 只精确维护本地 L1/L2 元数据。访问 L3 时，实例根据 prefix hash 实时查询后端，不要求所有实例持续同步完整的 L3 radix tree。

## 4. Agentic RL 为什么特别适合全局 KV



### 4.1 多轮 trajectory 的 prefix 持续增长

典型 agent loop：

```text
Turn 1 input:
  system + tools + user

Turn 1 decode output:
  reasoning + tool_call

Turn 2 input:
  system + tools + user + reasoning + tool_call + tool_result
```

Turn 2 的大部分输入都是 Turn 1 已经计算过的 prefix。如果下一轮落到另一个实例，没有全局 KV 时需要重复计算整段 trajectory。

### 4.2 Agent 等待工具时，本地 cache 容易变冷

Agent 发出 tool call 后可能等待：

```text
代码执行
浏览器操作
环境交互
长耗时 RPC
人工反馈
```

等待期间其他 rollout 会持续占用 KV pool，普通 LRU 可能淘汰这段 trajectory。全局 L3 可以在本地 HBM/DRAM 淘汰后继续提供恢复能力。

### 4.3 动态路由有利于消除 RL rollout 长尾

Agent trajectory 的输出长度和工具耗时差异很大。为了避免某个 worker 被长 trajectory 卡住，router 可能把下一轮发给另一个空闲 worker。

这里存在两个目标：

```text
locality: 尽量命中原实例的 HBM KV
load balance: 避免少数长任务造成 GPU 空转和尾延迟
```

Cache-aware routing 先利用本地命中，全局 L3 则让负载均衡决定跨实例迁移时不必完全丢失历史 KV。

## 5. 推荐架构



### 5.1 普通多副本，不做 PD

全局 KV Store 不依赖 PD，可以直接连接多个普通 SGLang server：

```text
                         +-> SGLang worker 0 -- L1/L2 --+
client -> model gateway -+                              +-> Mooncake Store L3
                         +-> SGLang worker 1 -- L1/L2 --+
```

适合：

```text
暂时不想引入 PD 复杂度
统一 engine 的吞吐已经足够
主要目标是跨 replica prefix 共享和容量扩展
```

建议仍启用 `cache_aware`，因为本地 HBM hit 比远程 L3 hit 快。

### 5.2 PD + 全局 KV Store

```text
                              +-> Prefill 0 -- L1/L2 --+
client -> PD router/gateway --+                         +-> Shared Mooncake L3
                              +-> Prefill 1 -- L1/L2 --+
                                         |
                                  P2P KV transfer
                                         |
                              +-> Decode 0 -----------+
                              +-> Decode 1 -----------+
                                         |
                                  async KV write-back
                                         +-----------> Shared Mooncake L3
```

这里存在两条不同数据通路：

```text
请求内低延迟通路：Prefill -> Decode，走 Mooncake TransferEngine
跨请求共享通路：各实例 <-> Mooncake Store，走 HiCache L3
```

对于多轮 agent，Decode 写回非常关键：Prefill 只持有请求开始时的 prompt KV，assistant/tool-call 等新增 token 的 KV 是 Decode 产生的。只有 Decode 把新增 KV 写回 L3，下一轮换到任意 Prefill 时才可能恢复完整 trajectory。

## 6. SGLang 最小配置骨架

下面是配置关系示例，不是可以直接复制到所有集群的生产配置。实际需要根据模型、TP/DP/PP、NIC、Host 内存和 Mooncake 部署方式调整。

### 6.1 Mooncake Store

先启动 Mooncake master。可以让 master 内嵌 metadata server：

```bash
mooncake_master \
  --port 50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95
```

生产中推荐把 Store 内存从 rollout server 解耦，启动独立 Store service。示例配置：

```json
{
  "local_hostname": "STORE_NODE_IP",
  "metadata_server": "http://MOONCAKE_MASTER_IP:8080/metadata",
  "master_server_address": "MOONCAKE_MASTER_IP:50051",
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1",
  "global_segment_size": "256gb",
  "local_buffer_size": 0
}
```

```bash
python -m mooncake.mooncake_store_service \
  --config=/path/to/mooncake_store.json \
  --port=8081
```

如果已经有独立 Store service，SGLang requester 的 `global_segment_size` 可以设为 `0`。如果没有独立 Store service，则可以让 SGLang 进程自己贡献 Host DRAM，但会让缓存生命周期和 rollout 进程生命周期耦合。

### 6.2 Prefill server

所有需要共享 L3 的实例必须连接相同的 Mooncake master/tenant，并使用兼容的模型与 KV 布局。

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30001 \
  --page-size 64 \
  --enable-metrics \
  --enable-cache-report \
  --enable-hierarchical-cache \
  --hicache-size 100 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{
    "master_server_address": "MOONCAKE_MASTER_IP:50051",
    "metadata_server": "http://MOONCAKE_MASTER_IP:8080/metadata",
    "local_hostname": "PREFILL_NODE_IP",
    "global_segment_size": 0,
    "protocol": "rdma",
    "device_name": "mlx5_0,mlx5_1",
    "tenant_id": "rollout-model-v42",
    "extra_backend_tag": "policy-step-42"
  }' \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1
```

策略取舍：

```text
write_through:
  新 KV 尽快进入 L3，跨实例复用最积极，但写带宽开销最大

wait_complete:
  等待远程 KV 完整拉回，命中利用率最高，但可能增加请求等待

timeout:
  在 L3 hit 和 TTFT/SLO 之间折中，通常更适合线上生产

best_effort:
  尽量不因远程 I/O 阻塞，可能放弃部分 L3 hit 并重新计算
```



### 6.3 Decode server

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --host 0.0.0.0 \
  --port 30011 \
  --page-size 64 \
  --enable-metrics \
  --enable-cache-report \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1 \
  --disaggregation-decode-enable-offload-kvcache \
  --hicache-ratio 2 \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-backend-extra-config '{
    "master_server_address": "MOONCAKE_MASTER_IP:50051",
    "metadata_server": "http://MOONCAKE_MASTER_IP:8080/metadata",
    "local_hostname": "DECODE_NODE_IP",
    "global_segment_size": 0,
    "protocol": "rdma",
    "device_name": "mlx5_0,mlx5_1",
    "tenant_id": "rollout-model-v42",
    "extra_backend_tag": "policy-step-42"
  }'
```

`--disaggregation-decode-enable-offload-kvcache` 是 Decode 新增 trajectory KV 写回共享存储的关键开关。

### 6.4 Router / Model Gateway

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill-0:30001 http://prefill-1:30001 \
  --decode http://decode-0:30011 http://decode-1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two \
  --cache-threshold 0.3
```

推荐思路：

```text
Prefill:
  cache_aware，优先利用本地 prefix KV，过载时允许迁移并由 L3 兜底

Decode:
  power_of_two/负载优先，重点避免 decode 长尾和并发不均衡
```

当前 Gateway 的 `cache_aware` 主要基于请求历史维护近似文本 radix tree，并结合 worker load 做选择；它不是对所有 worker HBM/L2/L3 状态进行强一致实时查询。因此它提供的是 cache affinity 估计，而不是全局缓存目录本身。

## 7. 全局 KV 可以命中的前提



### 7.1 Token prefix 必须真正一致

Prefix cache 比较的是 tokenizer 后的 token 序列，不是“语义相似”。以下变化都会让后续部分 miss：

```text
system prompt 变化
tool schema 顺序变化
chat template 变化
消息中插入动态时间戳/request id
推理框架在开头注入不同 metadata
同一工具列表序列化顺序不稳定
截断策略不同
```

在 agentic RL 中，应让稳定内容尽可能靠前：

```text
稳定 system prompt
稳定且排序固定的 tool definitions
稳定 few-shot/example context
动态 request 信息放到后面
```



### 7.2 模型与 KV 计算语义必须一致

至少需要保持：

```text
相同模型权重版本
相同 tokenizer/chat template
相同 RoPE/位置编码语义
兼容的 KV dtype
兼容的 page/block 配置
兼容的 attention/KV layout
```

不同 TP size 的共享不能直接想当然。SGLang 部分 MHA/GQA + Mooncake 布局支持通过 `tp_lcm_size` 做 heterogeneous TP 共享，但需要满足对应模型和 layout 约束：

```bash
--hicache-storage-backend-extra-config '{"tp_lcm_size": 8, ...}'
```

如果所有实例 TP 一致，优先先使用一致配置把基本链路验证正确。

### 7.3 所有实例必须使用相同 L3 namespace

需要共享的实例应使用相同：

```text
Mooncake master
tenant_id
model/cache key prefix
extra_backend_tag
```

不同模型或不同 policy 权重版本则应该隔离 namespace。

## 8. RL 权重更新是最危险的正确性边界



### 8.1 为什么旧 KV 不能复用

KV cache 是模型权重的函数：

```text
KV = f(tokens, model_weights, model_config, positions, kv_dtype, ...)
```

RL 每个训练 step 更新 policy 权重后，即使 token prefix 完全相同，旧权重计算的 KV 也不能交给新权重继续 decode。否则 rollout 会混用不同 policy 版本，破坏 on-policy 语义，甚至产生不可预测输出。

### 8.2 当前实现中的关键事实

SGLang weight update API 的 `flush_cache=True` 会清理实例本地 KV cache，但普通本地 `/flush_cache` 不等价于清空共享 L3。

当前工作区里，HiCache L3 的基础 hash key 主要来自 token prefix/page hash；`weight_version` 不会自动加入 Mooncake L3 key。

同时，请求字段 `cache_salt` 当前用于实例内 radix tree 和 external KV event namespace，源码明确说明它不负责给远程 L3 storage key 加 namespace。因此不能只给请求设置 `cache_salt=weight_version`，然后假设 Mooncake L3 已经按权重版本隔离。

Mooncake backend 当前会用 `model_name` 和可选的 `extra_backend_tag` 形成 config prefix，还支持 `tenant_id`。这两者更适合做部署级/权重版本级隔离。

### 8.3 推荐的权重更新流程

方案 A：清理当前版本的共享 L3。

```text
1. 停止接收新 rollout 请求
2. pause/retract 正在运行的请求
3. 等待所有旧版本 L3 write-back 完成或停止
4. 更新所有 Prefill/Decode/普通 rollout worker 的权重
5. 使用 flush_cache=True 清理所有实例本地 KV
6. 清理对应 tenant 的 HiCache storage backend
7. 确认所有 worker 的 weight_version 一致
8. 恢复 generation
```

清理 L3 的管理接口：

```bash
curl -X POST http://SGLANG_SERVER/hicache/storage-backend/clear
```

注意：Mooncake backend 的 `clear()` 当前调用 `remove_all()`。应给 RL job/模型使用专用 tenant，避免清理时误删其他业务的缓存；也要避免旧 worker 在 clear 之后继续把旧权重 KV 写回。

方案 B：每个权重版本使用独立 namespace。

```text
tenant_id        = rollout-model-v42
extra_backend_tag = policy-step-42
```

更新后切到：

```text
tenant_id        = rollout-model-v43
extra_backend_tag = policy-step-43
```

优点是不会发生新旧版本 key 碰撞，也不需要立即物理删除旧缓存；缺点是需要协调 backend 配置切换和旧 namespace 的垃圾回收。

对于高频 RL step，更理想的长期方案是让 weight version 成为 L3 namespace/key 的系统级组成部分，并提供安全的版本切换与异步 GC；在确认框架已经实现这一点之前，不应默认它自动成立。

## 9. 如何验证确实发生了跨实例 L3 命中

不要只通过第二次请求 TTFT 下降来判断，因为它可能命中了同一实例的 GPU cache。

建议执行受控验证：

```text
1. worker A 和 worker B 都连接同一个 Mooncake Store
2. 使用相同模型、page size、KV dtype、tenant/tag
3. 绕过 router，直接向 worker A 发送一个长且确定的 prompt
4. 使用 write_through，并等待 L3 写入完成
5. 确认 worker B 的本地 cache 是冷的，或直接新启动 worker B
6. 向 worker B 发送相同 prefix
7. 检查返回细分和 Prometheus storage_hit 指标
```

请求中可以开启：

```json
{
  "return_cached_tokens_details": true
}
```

关注返回中的：

```json
{
  "cached_tokens_details": {
    "device": 0,
    "host": 0,
    "storage": 4096,
    "storage_backend": "MooncakeStore"
  }
}
```

Prometheus 指标可以关注：

```text
sglang:prefill_effective_tokens_total{mode="device_hit"}
sglang:prefill_effective_tokens_total{mode="host_hit"}
sglang:prefill_effective_tokens_total{mode="storage_hit"}
sglang:prefill_effective_tokens_total{mode="input"}
```

只有 worker B 出现 `storage_hit > 0`，才直接证明跨实例 L3 复用链路生效。

还应分别测量：

```text
本地 HBM hit TTFT
远程 DRAM L3 hit TTFT
远程 SSD L3 hit TTFT
cold prefill TTFT
L3 read/write bandwidth
Prefill GPU compute token reduction
请求端到端吞吐和长尾延迟
```

命中率提高不一定等于吞吐提高；如果 prefix 太短，远程查询和传输可能比重新计算更贵。

## 10. Agentic RL 的实践建议



### 10.1 先后实施顺序

建议按复杂度逐级验证：

```text
阶段 1：普通多副本 + cache-aware routing
阶段 2：普通多副本 + HiCache/Mooncake 全局 L3
阶段 3：PD + P2P KV transfer
阶段 4：PD + Prefill L3 共享
阶段 5：Decode async write-back，验证跨 turn 完整 trajectory 命中
阶段 6：接入 RL weight update 的 L3 版本隔离/清理
```

每一阶段都应保留 cold/local/L3 三组基线，否则很难判断收益来自哪里。

### 10.2 路由不是越随机越好

即使已经有全局 L3，也建议：

```text
负载接近平衡时优先 cache affinity
负载明显失衡时允许迁移
迁移后由 L3 避免完全重算
```

不要因为“所有实例都能访问 Store”就完全放弃 locality。全局可访问解决可用性，本地命中解决最低延迟。

### 10.3 Prefix 结构比缓存参数更先决定命中率

如果 prompt 在开头就含有随机字段，后面的长 system prompt、tools 和历史都无法形成连续 prefix hit。应先审查请求序列化，再调 page size、prefetch policy 和 Store 容量。

### 10.4 全局 L3 需要独立容量与带宽规划

至少估算：

```text
每 token KV bytes
x 平均可复用 prefix tokens
x 并发 trajectories
x 权重版本保留数量
x 副本/TP 布局系数
```

同时测量 Store 的读写放大、RDMA NIC 带宽和 metadata QPS。`write_through` 能提高可见性，但在大量一次性 trajectory 下可能写入很多永远不会再次命中的 KV。

### 10.5 一次性 rollout 与多轮 agent 应采用不同策略

```text
一次性、低重复 prompt：
  全局 Store 收益可能小，优先算力吞吐和负载均衡

长 system prompt / few-shot：
  Prefill L3 共享价值高

多轮 agent trajectory：
  Decode write-back + 跨 turn L3 价值最高

高频权重更新：
  namespace/失效成本可能成为主要问题
```



## 11. 当前推荐组合

针对大规模 agentic RL rollout，当前推荐：

```text
SGLang Model Gateway
  Prefill policy = cache_aware
  Decode policy = power_of_two / load-aware

Prefill cluster
  RadixCache L1
  HiCache Host L2
  Mooncake Store L3
  write_through 或 selective/timeout 的生产折中

Decode cluster
  Mooncake P2P 接收 Prefill KV
  decode KV async offload 到同一个 L3

RL controller
  pause -> weight update -> local flush
  -> L3 clear 或切换 version namespace
  -> resume
```

这个组合的目标不是让每个请求随机选择任意实例，而是：

```text
能本地命中时尽量本地命中
必须迁移时仍能从全局 L3 恢复
多轮 Decode 新增 KV 能被下一轮 Prefill 复用
权重更新时绝不复用旧 policy KV
```



## 12. 工作区源码与文档入口

SGLang：

- `docs/docs/advanced_features/hicache_design.mdx`
- `docs/docs/advanced_features/hicache_best_practices.mdx`
- `docs/docs/advanced_features/pd_disaggregation.mdx`
- `docs/docs/advanced_features/sgl_model_gateway.mdx`
- `docs/docs/advanced_features/sglang_for_rl.mdx`
- `python/sglang/srt/mem_cache/storage/mooncake_store/README.md`
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/managers/scheduler_components/output_streamer.py`
- `sgl-model-gateway/src/policies/cache_aware.rs`
- `test/registered/disaggregation/test_disaggregation_decode_offload.py`

外部官方资料：

- [SGLang HiCache System Design](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx)
- [SGLang Mooncake L3 README](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md)
- [Mooncake Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html)
- [vLLM MooncakeStoreConnector Usage Guide](https://github.com/vllm-project/vllm/blob/main/docs/features/mooncake_store_connector_usage.md)
- [vLLM Distributed KV Cache Pool with Mooncake Store](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-05-06-mooncake-store.md)



## 13. 后续值得深入的问题

后续讨论可以沿以下方向展开：

1. SGLang HiCache L3 key/hash 的精确生成过程，以及 TP/PP/page-size 如何进入 key。
2. Decode async offload 的生命周期、写回时机和 abort/retract 正确性。
3. Gateway `cache_aware` 的近似 radix tree 与真实 engine KV 状态可能产生多大偏差。
4. Agent session affinity、session-aware eviction 与全局 L3 如何组合。
5. RL 高频 weight update 下，如何设计低成本的 versioned KV namespace 和 GC。
6. `write_through`、`write_through_selective`、`write_back` 在真实 agent trace 上的收益差异。
7. 什么时候远程 L3 load 比 recompute 更划算，如何建立 token 数/带宽/算力成本模型。
8. PD 中 Prefill/Decode heterogeneous TP 下的 KV layout 与 Store 复用限制。


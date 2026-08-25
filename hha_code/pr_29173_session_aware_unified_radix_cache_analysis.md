# SGLang PR #29173 深度分析：Session-reference-aware Unified Radix Cache

> 分析对象：[PR #29173](https://github.com/sgl-project/sglang/pull/29173)  
> 最终合入提交：[056474cdb0689ee5751c10f9c78e39e33c451e87](https://github.com/sgl-project/sglang/commit/056474cdb0689ee5751c10f9c78e39e33c451e87)  
> PR 创建：2026-06-24；合入：2026-08-02；本文分析时间：2026-08-15  
> 规模：23 个文件，`+1056 / -373`
>
> `合入版本： 0.5.17`

## 1. 一句话结论

这个 PR 没有把 active session 的 KV cache “锁死”，而是给 `UnifiedRadixCache` 增加了一层 **session 生命周期感知的软淘汰优先级**：优先淘汰没有活跃 session 引用的 KV；只有空间仍不足时，才回退到淘汰有引用的 KV。

它解决的是 agent/RL 多轮请求之间的缓存保活问题，不负责拼接对话 prompt，也不提供硬 pin、容量预留、TTL、准入控制或跨实例 session affinity。

我的总体判断：

- 方向正确，且比 hard pin 更不容易造成 KV 池死锁。
- 最终实现已从早期的普通 `RadixCache`/`HiRadixCache` ref-aware 设计，收敛为只改 `UnifiedRadixCache`，架构边界更清楚。
- Full/SWA/Mamba 三种 cache component 的可复用语义不同，PR 针对它们分别实现了引用范围和淘汰顺序，这是本 PR 最有价值也最复杂的部分。
- 生命周期防陈旧机制（generation）是必要且合理的；`close_session` 解除引用但不立刻释放 KV，也符合 radix cache 的复用语义。
- 合入时最大的不足是验证深度：新增行为测试实际上主要覆盖 Full-only CPU 路径；SWA、Mamba、host eviction、HTTP lifecycle、DP/PD 和异常关闭缺少 feature-specific E2E 覆盖。
- 性能图支持“高压力下命中结构改善”，但不足以独立证明作者所说的通用吞吐提升，尤其没有给出吞吐数值、误差、硬件和完整实验配置。



## 2. 它想解决什么问题

普通 radix cache 只看到 token prefix 和缓存策略（例如 LRU），不知道某个 prefix 是否仍属于一个活跃的多轮 agent session。

典型流程是：

1. Agent 第 1 轮生成完成，形成一段很长、下轮可复用的 KV prefix。
2. Agent 离开模型执行工具调用、代码编辑或环境交互；这段时间 prefix 看起来是“冷”的。
3. 其他无关请求不断写入 KV，触发内存压力。
4. 普通 LRU 可能先淘汰 agent 下一轮马上会用到的 prefix，却保留一些已经没有后续请求的 KV。
5. Agent 第 2 轮回来时必须重新 prefill，降低 cache hit、TTFT 和吞吐。

PR 引入的核心额外信息是：“这段 KV 仍被哪些活跃 session 的未来 turn 引用”。因此，淘汰决策从：

```text
base_eviction_policy(node)
```

升级成逻辑上的：

```text
先看是否被活跃 session 引用
再看引用强度/分区
最后保留原有 eviction policy 的相对顺序
```

关键点是 **soft protection**：session reference 不增加 `lock_ref`，所以 referenced KV 仍属于 evictable cache。若 unreferenced KV 不够，系统仍能释放 referenced KV，避免 hard pin 把整个 KV pool 占满后无法前进。

## 3. 三种容易混淆的“session”概念


| 概念                                      | 输入字段/对象                           | 作用                                 | 是否重建上下文          | 是否硬持有 KV                      |
| --------------------------------------- | --------------------------------- | ---------------------------------- | ---------------- | ----------------------------- |
| 本 PR 的 radix-native session             | 请求顶层 `session_id`                 | 给普通 radix KV 加 session reference   | 否                | 否，soft reference              |
| 原有 SessionController / StreamingSession | `session_params.id`、`req.session` | 管理 SGLang session，可能跨 turn 保存/拼接状态 | 是，取决于 session 模式 | StreamingSession 可能通过 lock 持有 |
| 单请求身份                                   | `rid`                             | 标识一个 request                       | 否                | 仅 request 生命周期内               |


最重要的 API 语义是：顶层 `session_id` **只是 cache lifecycle metadata**。调用方每一轮仍必须发送该轮完整、正确的 prompt；SGLang 不会因为这个字段自动补齐历史消息。

实现也兼容旧 session：`session_id_for_req()` 优先取 `req.session_id`，没有时再回退到 `req.session.session_id`。但从使用角度看，不应把顶层 `session_id` 与 `session_params` 当作可随意互换的接口。

## 4. 最终架构

```mermaid
flowchart TD
    A[Generate request<br/>top-level session_id] --> B[Scheduler]
    B --> C[ensure_session_generation]
    B --> D[正常 match / prefill / decode]
    D --> E[cache_finished_req]
    E --> F{成功完成且发生 insert?}
    F -- yes --> G[UnifiedSessionRefTracker.register_session_ref]
    G --> H[FullComponent<br/>prefix path]
    G --> I[SWAComponent<br/>window tail]
    G --> J[MambaComponent<br/>frontier state]

    K[/close_session] --> L[release_radix_session]
    L --> H
    L --> I
    L --> J

    H --> M[device / host eviction]
    I --> M
    J --> M
    M --> N[unreferenced first<br/>referenced fallback]
```



职责分层：

- `Scheduler`：识别顶层 `session_id`，为 request 绑定 generation；处理 open/close 控制消息。
- `UnifiedRadixCache`：组合 `UnifiedSessionRefTracker`，在成功缓存 finished request 后注册引用。
- `UnifiedSessionRefTracker`：管理 session generation、closed tombstone，并把 register/release 分发给每个 component。
- `TreeComponent`：维护每个 component 自己的 session frontier 索引和引用状态。
- `FullComponent`、`SWAComponent`、`MambaComponent`：定义“这个 component 对一个 session 而言，哪些节点是可复用的”。
- `UnifiedTreeCore` / `UnifiedLRUList`：实现分区 LRU、级联淘汰、节点删除和一致性检查。

这也是为什么最终版本只支持 `UnifiedRadixCache`：普通 `RadixCache` 没有 Full/SWA/Mamba 的统一 component 模型，硬把同一套语义塞进去会重新制造平行实现。

## 5. 请求生命周期逐步追踪



### 5.1 第一条带 `session_id` 的请求

当 `--enable-session-radix-cache` 开启，且请求带顶层 `session_id="agent-42"`：

1. `Scheduler.handle_generate_request()` 把它当普通非 StreamingSession 请求创建 `Req`。
2. `Req.session_id` 保存顶层 ID。
3. `tree_cache.ensure_session_generation("agent-42")`：
  - 若 session 首次出现，隐式 open；
  - 全局单调计数器加一；
  - 保存 `_session_generations[session_id] = generation`。
4. generation 写入 `req.session_generation`，成为这个 request 所属 session incarnation 的快照。
5. request 正常执行 prefix match、prefill、decode。

顶层 session 不要求调用 `/open_session`。`/open_session` 仍主要是旧 SessionController API，但成功 open 时也会显式建立 radix generation。

### 5.2 请求完成时注册引用

注册发生在 `UnifiedRadixCache.cache_finished_req()` 的尾部，而不是请求刚进入时：

1. 将 committed token/KV 插入 radix tree。
2. `InsertResult.last_device_node` 写回 `req.last_node`。
3. component 完成自己的 cache cleanup。
4. 仅当以下条件同时成立才注册：
  - session radix cache 已开启；
  - insert 确实发生并返回 result；
  - `finished_reason` 非空；
  - 不是 `FINISH_ABORT`。
5. tracker 校验 `req.session_generation == 当前 generation`。
6. 每个 component 从 `last_node` 向上寻找自己最近的可复用节点，并注册自己的 session frontier。

因此，中途的 chunked-prefill `cache_unfinished_req()` 不会直接建立 session reference；引用在最终成功完成后才生效。

Disaggregated prefill 中有一行看似无关但很关键的调整：先设置成功的 `finished_reason`，再调用 `release_kv_cache()`。否则进入 `cache_finished_req()` 时完成状态尚未就绪，session registration 会被跳过。

### 5.3 同一个 session 的后续 turn

若新 frontier 是旧 frontier 的后代：

- Full 只给新增 suffix 路径增加 coverage，不重复增加旧 prefix。
- SWA/Mamba 先为新 frontier 建立 coverage，再撤销旧 ancestor frontier 的 coverage。
- `_session_leaves[session_id]` 从旧 frontier 前移到新 frontier。

若同一个 session 产生多个分支，则可以保留多个 frontier。这里的 `session_ref` 更准确地说是“session coverage contribution 计数”，在线性单分支 session 中等价于 session 数；在同一 session 多分支共享 ancestor 时，ancestor 的计数可能大于 distinct session 数。这不会破坏 register/release 的加减平衡，但会让 Full 的 `session_ref_count` 带有“分支权重”。

### 5.4 `/close_session`

`release_radix_session(session_id)` 做四件事：

1. 把 session ID 放入最多 8192 项的 closed tombstone LRU。
2. 删除当前 generation。
3. 遍历 Full/SWA/Mamba 的 session frontier，递减各自 coverage。
4. 删除 session-to-frontier 与 frontier-to-session 的索引。

它明确 **不立即 free KV**。被解除引用的 KV：

- 仍然可被后续内容相同的请求命中；
- 从 referenced 优先级回到普通 unreferenced 淘汰区；
- 真正的内存释放由后续正常 eviction 完成。

`/close_session` HTTP 200 只表示控制消息已发给 scheduler，并不是 scheduler 回传的“所有引用已释放”确认。当前调用方通常依赖消息队列顺序，而不是 close completion ack。

### 5.5 为什么需要调用方显式 close

`close_session` 的准确含义不是“现在立刻释放这段 KV”，而是显式告诉引擎：

> 这个 session 已经结束，其 KV 不再需要享受 active-session 的优先保留待遇。

调用 close 后：

- 对应 Full/SWA/Mamba coverage 上的 `session_ref` 被解除；
- KV 仍留在 radix cache，内容相同的后续请求依然可以命中；
- SWA/Mamba 节点从 referenced 分区移入 unreferenced 分区，Full 节点也恢复为未引用淘汰优先级；
- 下次出现内存压力时，这些 KV 会排在仍被活跃 session 引用的 KV 前面被回收。

因此，close 是 **lifecycle/eviction-priority hint**，不是同步 free API。

如果调用方一直不 close：

- KV 不是 hard pin，仍属于 evictable cache；
- 内存紧张时，引擎先淘汰 unreferenced KV；
- 如果 unreferenced KV 不足，引擎仍会回退到淘汰 referenced KV，通常不会等到真正 OOM 才开始处理；
- referenced leaf 被淘汰时，tracker 可能把 session frontier 后退到更短的可复用 ancestor，因此 session 对较短 prefix 的保护还可能继续存在；
- 当相关 component 已没有任何可复用 fallback 时，该条 frontier reference 才会自然消失。

可以把三种状态概括为：

```text
普通/unreferenced KV：有压力时优先淘汰
active-session KV：   尽量晚淘汰，但仍然可淘汰
close 后的 KV：        取消“尽量晚淘汰”的待遇，不要求立即释放
```

漏 close 的主要后果不是“这段 KV 永久占住显存”，而是它可能长期获得不应有的保留优先级，挤掉其他更有价值的缓存。如果大量 session 都漏 close，越来越多 KV 都被标成 referenced，unreferenced-first 策略会逐渐失去区分能力，退化得更接近普通 LRU。

正确调用时机通常是整个 agent trajectory、RL rollout、任务或用户会话结束时，而不是每一个 turn 结束时。正常完成、取消、异常和超时清理路径都应调用 close。

## 6. 防止 close / reopen 竞态：generation

需要防的竞态是：

```text
request A(gen=1) 正在运行
    -> close session S
    -> 重新使用同一个 ID 打开 S(gen=2)
    -> A 很晚才完成
```

如果只用 `session_id`，A 会错误地把旧 incarnation 的 KV 挂到新 session 上。PR 通过单调 generation 解决：

```text
req.session_generation = 1
current_generation[S] = 2
1 != 2  => 跳过 A 的注册
```

closed tombstone 还能快速拒绝 close 后才完成的 request，并让重复 close 成为 no-op。即使 tombstone 因 8192 上限被淘汰，generation 已被删除或更新，正常 scheduler 路径下仍能阻止旧 request 注册。因此，最终实现里真正决定 incarnation 正确性的是 generation，tombstone 更偏向 closed 状态与幂等保护。

顶层 `session_id` 的新请求可以隐式 reopen：`ensure_session_generation()` 发现 generation 不存在时，会调用 `open_radix_session()`，同时清除同 ID tombstone。

## 7. 三类 component 为什么不能共用一种引用范围



### 7.1 Full attention：引用整条可复用 prefix path

Full attention 的下一 token 依赖此前所有 token 的 KV，所以一个 session 注册 leaf 后，coverage 是：

```text
root -> ... -> registered frontier
```

实现给路径上每个非 root 节点的 `ComponentData.session_ref` 加一。

设备和 host eviction 都保留原 heap 结构，但 heap key 变为：

```python
(session_ref > 0, session_ref, base_eviction_priority(node))
```

Python tuple 从小到大弹出，所以顺序是：

1. 未引用节点（`False`）先淘汰；
2. referenced 节点中，引用计数较少的先淘汰；
3. 再由原配置策略（LRU、priority 等）打破平局。

这意味着不再要求 `--radix-eviction-policy priority`，而且不会覆盖用户原有策略，只是在其外层加 session 维度。

### 7.2 SWA：只引用滑动窗口尾部

Sliding Window Attention 不需要全部历史 KV，只需要 frontier 向前一个 window 的连续覆盖。因此目标 span 是：

```text
sliding_window_size + page_size
```

多出的 page allowance 用于分页/节点边界安全。实际计数按完整 tree node 更新，所以最后一个边界节点可能让 coverage 超过目标 span；这是一种以节点粒度换实现简单和安全的软过保护。

SWA 不使用 Full 的 heap，而是把 device LRU 和 host LRU 分成两段：

```text
head
  referenced MRU ... referenced LRU
  [mid sentinel]
  unreferenced MRU ... unreferenced LRU
tail
```

淘汰从 tail 向 head：先清空 unreferenced 分区，穿过 `mid` 后才进入 referenced 分区。一个可移动的 locked cursor sentinel 让删除、tombstone 和级联淘汰过程中遍历位置仍然稳定。

### 7.3 Mamba：只引用 frontier state

Mamba 的可复用对象不是完整 token path，而是 match frontier 上的状态 checkpoint。因此 coverage 只对一个 node 的 Mamba component 加一。

它和 SWA 使用同样的两分区 LRU：

- unreferenced Mamba state 先淘汰；
- 不够时再淘汰 referenced state。

节点 split 时也体现了差异：Full/SWA 的新 parent 继承 path coverage；Mamba state 仍属于 child frontier，新 parent 的 Mamba `session_ref` 为 0。

## 8. component cascade 与 session reference 的交互

Unified cache 的内部淘汰优先级是：

```text
Full(2) > SWA(1) > Mamba(0)
```

内部节点上：

- 淘汰 Full 会级联淘汰 SWA 和 Mamba；
- 淘汰 SWA 会级联淘汰 Mamba；
- 淘汰 Mamba 不影响更高 component。

叶节点最终会整体删除，所以三个 component 的 leaf priority 都视为 0。

本 PR 增加了两个保护：

1. `_can_evict_leaf_atomically()`：如果当前 trigger component 未引用，但同一 leaf 上同级/更高价值 component 有 session reference，不允许当前 component 触发整叶删除。
2. `_cascade_evict()`：未引用的低/同级 trigger 不应顺带删掉被 session 引用的更高价值 component。

注意这仍是 soft protection：如果触发淘汰的 component 本身也 referenced，说明 eviction 已经回退到 referenced 分区，此时可以继续整体淘汰来释放空间。

## 9. tree 变形、删除时如何保持引用正确



### 9.1 Node split

Radix node 被拆分为新 parent + child 时：

- Full/SWA 的 path coverage 复制到新 parent；
- `session_ids` marker 仍只挂在 frontier child，不把 parent 错标为 frontier；
- Mamba parent 不继承 frontier state reference。



### 9.2 Referenced leaf 被正常 eviction

因为 reference 不是 pin，referenced leaf 仍可能被删。删除前 `discard_deleted_session_leaf()` 会：

1. 为每个 component 向 parent 方向寻找最近仍可复用的 fallback；
2. 撤销被删除 leaf 对应的 coverage；
3. 若存在 fallback，把该 session 的 frontier 后退到 fallback；
4. 若没有 fallback，删除这条 component frontier。

这允许一个活跃 session 在内存压力下“损失 suffix、保留较短 prefix”，而不是引用索引直接悬空。

## 10. 数据结构与复杂度

每个 node 的每个 `ComponentData` 新增：

```python
session_ref: int = 0
session_ids: Optional[set[str]] = None
```

每个 `TreeComponent` 新增：

```python
_session_leaves: dict[str, set[UnifiedTreeNode]]
```

SWA 额外记录每个 frontier 实际覆盖长度：

```python
_session_leaf_covered_len: dict[
    str, dict[UnifiedTreeNode, int]
]
```

大致复杂度：


| 操作              | Full                  | SWA                | Mamba       |
| --------------- | --------------------- | ------------------ | ----------- |
| 注册/前移 frontier  | `O(tree depth delta)` | `O(window 内节点数)`   | `O(1)`      |
| close/release   | 所有 frontier 路径        | 所有 frontier window | frontier 数  |
| device eviction | 建 heap，约 `O(L log L)` | 分区 LRU 顺序扫描        | 分区 LRU 顺序扫描 |
| host eviction   | heap                  | 分区 host LRU        | 分区 host LRU |


内存开销有两部分：

- 即使功能关闭，每个 component/node 的 `ComponentData` 也多两个 Python 字段；
- 功能开启后，frontier 节点才分配 `session_ids` set，并在每个 component 的反向索引中保存 session/node 引用。

PR 没有给这部分 CPU、内存和 close latency 的 profiling 数据。

## 11. 与 PR 之前实现的关键语义变化

这个 PR 不是从零增加同名 flag。它删除了此前的 `SessionRadixCacheMixin` 和普通 `RadixCache` 上的实现。


| 旧实现                             | 最终 PR                                    |
| ------------------------------- | ---------------------------------------- |
| 混入普通 `RadixCache`               | 组合进 `UnifiedRadixCache`                  |
| leaf 上打 session tag             | 每个 component 独立跟踪 frontier/coverage      |
| close 时尝试直接释放 session 独占 leaf   | close 只 dereference，不立即 free             |
| tag 本身不改变普通 LRU 顺序              | unreferenced-first eviction              |
| 不支持 Unified 的 Full/SWA/Mamba 差异 | 明确支持三个 component                         |
| 主要靠 closed tombstone 防迟到 finish | generation + tombstone 防 close/reopen 竞态 |


早期 PR 讨论过基于 request priority 的 high/low reference，并暴露过混合优先级泄漏、Python 3.12 import、chunked prefill、HTTP 控制路由和 HiCache host-pressure 等问题。最终合入代码已经不是那套设计：没有 high/low ref，也不再把 session 行为混入普通 Radix/HiRadix。阅读 review 时必须区分 outdated finding 和最终代码。

## 12. 配置和 API

启动条件：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 python3 -m sglang.launch_server \
  --model-path MODEL_PATH \
  --enable-session-radix-cache
```

若实际构建的 tree cache 不是 `UnifiedRadixCache`，registry 会抛出：

```text
--enable-session-radix-cache requires UnifiedRadixCache
```

请求：

```bash
curl http://localhost:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "FULL_PROMPT_FOR_THIS_TURN",
    "sampling_params": {"max_new_tokens": 128},
    "session_id": "agent-42"
  }'
```

关闭：

```bash
curl -X POST http://localhost:30000/close_session \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "agent-42"}'
```

调用方必须在正常完成、错误、取消和超时清理路径上 close。没有 close 不会直接造成 hard-pin OOM，但会让越来越多 KV 被标成 referenced，最终让“unreferenced first”的区分能力退化。

## 13. 性能结果怎么解读



### 13.1 启动命令与复现信息

先说明一个关键限制：**PR 页面没有公开作者实际使用的完整 server launch command 和 workload replay command**。已知条件只有 GLM-5.2、SWE-agent workload、UnifiedRadixCache/HiCache、batch size 32/64，以及 baseline 与 SessionAware 对比。因此，不能把下面的命令声称为作者原始实验命令。

根据合入提交的参数语义，一个最小、语义等价的 L1 GPU + L2 host HiCache 对照模板如下。真实复现时还必须补上作者未公开的模型并行、量化、显存比例和 workload 参数。

Baseline：UnifiedRadixCache + HiCache，不开启 session-aware eviction：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server \
  --model-path /path/to/GLM-5.2 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --enable-cache-report \
  <其余 GLM-5.2/TP/DP/量化/显存参数>
```

SessionAware：只在完全相同的 baseline 上增加 `--enable-session-radix-cache`：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server \
  --model-path /path/to/GLM-5.2 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --enable-session-radix-cache \
  --enable-cache-report \
  <其余参数必须与 baseline 完全一致>
```

这里显式写出的 `hicache-ratio=2.0`、`write_through` 和 `kernel` 是合入版本的默认值，用于让模板语义更清楚，并不代表作者实验一定使用这些值。图中同时报告 `device_hit_ratio` 和 `host_hit_ratio`，且 PR 正文明确把 baseline 描述为原 HiCache，因此可以确认实验至少存在 GPU device tier 和 CPU host tier；没有证据表明它开启了 storage-backed L3，本 PR 也明确不覆盖 L3。

公平对比时，请求流应该完全一致：

- 每个 agent trajectory 的所有 turns 使用稳定且唯一的顶层 `session_id`；
- 每个 turn 仍发送完整 prompt；
- 整个 trajectory 结束后调用 `/close_session`；
- baseline 也发送相同请求字段和相同 close 时序，只是服务端不开启 session-aware flag；
- 分别用作者所称的 batch size 32 和 64 重放同一 SWE-agent trace。

概念性的请求生命周期是：

```bash
# 同一 trajectory 的多个 turns 重复调用；FULL_PROMPT 每轮不同但 session_id 稳定
curl http://localhost:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "FULL_PROMPT_FOR_THIS_TURN",
    "sampling_params": {"max_new_tokens": 128},
    "session_id": "swe-agent-trajectory-42"
  }'

# 整条 trajectory 完成后调用一次
curl -X POST http://localhost:30000/close_session \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "swe-agent-trajectory-42"}'
```

PR 没有公开以下信息，所以目前无法严格复现实验：

- GLM-5.2 的具体 checkpoint、精度和量化方式；
- GPU 型号/数量、TP/DP/EP/PP 配置；
- `mem-fraction-static`、context length、page size；
- 实际 `hicache-ratio`、write policy、IO backend 和 host 内存规模；
- “batch size 32/64”在 harness 中究竟指并发数、调度 batch 上限还是其他含义；
- SWE-agent trace、trajectory 数、turn/token 长度分布和工具等待时间；
- session ID 分配和异常路径是否都正确 close；
- cache hit ratio 的采样窗口、预热方式、重复次数和误差；
- 作者所声称吞吐提升的原始数值和测量命令。

因此，以上命令只能说明“如何搭出与图相同的两层 cache 和 feature 对照”，不能作为图中数字的严格 reproduction recipe。

### 13.2 图中结果

PR 只展示了一张 [cache hit ratio 图](https://github.com/user-attachments/assets/ec63a341-f282-456a-980d-38cf8e4822e2)，场景是 GLM-5.2、SWE-agent workload、batch size 32/64，对比原 UnifiedRadixCache/HiCache 与 SessionAware。

从图上手工读取的近似值如下；图中没有数据标签，所以只能按刻度估算：


| batch size | 方案                | device hit | host hit | 总 hit |
| ---------- | ----------------- | ---------- | -------- | ----- |
| 32         | UnifiedRadixCache | ~16%       | ~69%     | ~85%  |
| 32         | SessionAware      | ~32%       | ~54%     | ~86%  |
| 64         | UnifiedRadixCache | ~4%        | ~35%     | ~39%  |
| 64         | SessionAware      | ~14%       | ~38%     | ~52%  |


可以比较可靠地得出：

- bs=32：总命中率几乎不变，但约 16 个百分点从 host hit 转成 device hit。这可能显著降低恢复成本，即使 total hit 看起来没变。
- bs=64：device hit 和 total hit 都明显改善，总命中约提升 13 个百分点。
- batch 越大仍然越容易 thrash；session reference 只改变 eviction preference，不能替代 admission control、batch throttling 或 concurrency control。

不能仅凭图得出：

- 具体端到端吞吐提升多少；PR 文本声称 throughput 提升，但未展示吞吐数字或图。
- 收益是否能泛化到其他模型、prompt 长度分布、turn 间隔、session 数和关闭策略。
- CPU bookkeeping、LRU 分区和 close traversal 的成本。
- 是否经过多次重复、是否有误差条，以及 baseline/实验组除 flag 外是否完全一致。

因此更准确的表述是：**图证明了该策略能改善缓存驻留层级，并在更高压力下提高总命中；吞吐收益是合理推论和作者实验结论，但 PR 页面没有给出足够数据让读者独立量化。**

## 14. 测试覆盖评估

新增注册测试文件有 222 行，但行为测试的实际核心只有三个：

1. Full component register/release 后 `session_ref` 从 0→1→0。
2. close/reopen 后，旧 generation request 不能重新注册。
3. Full eviction 优先删 unreferenced prefix。

其余大量断言是 architecture/source ownership 检查：

- 普通 `RadixCache` 不再拥有 session tracker；
- 删除旧 mixin 和相关 symbol；
- `UnifiedRadixCache` 使用 composition，不使用 mixin；
- Full/SWA/Mamba 源码包含 `session_ref`；
- registry 有 Unified-only guard。

合入时定向 CI 显示：

- CPU：`test_session_unified_radix_cache.py`、`test_tree_core_registry.py` 通过；
- 1-GPU：`test_unified_radix_cache_unittest.py` 通过；
- Unified radix tree 的 4-GPU H100 test group 通过；
- PR 页面最终标记 Base/Extra CI 通过。

我在当前工作区 main（2026-08-15）用已有 Python 环境重新运行 `test_session_unified_radix_cache.py`，退出码为 0。第一次运行遇到的是环境缓存目录只读，改用 `/tmp` cache 后通过。

明显缺口：

- 没有 SWA session coverage 的行为断言和 eviction 顺序测试；
- 没有 Mamba session frontier 的行为测试；
- 没有 host LRU/HiCache pressure 下的 session-aware eviction 专项测试；
- 没有多 session 共用 prefix、同 session 分支、frontier 回退、node split 的专项测试；
- 没有 `/generate -> /close_session` HTTP E2E；
- 没有 close 与 in-flight finish、close/reopen 并发 E2E；
- 没有 DP/PP/CP/PD 下控制消息和 generation 一致性的专项测试；
- 没有故障、取消、调用方漏 close 的 soak test。

评审者在合入前也明确提出：后续 PR 需要补 session radix tree E2E test。

## 15. 风险与已知限制



### 高关注：漏 close 没有 TTL

未显式 close 的 session 会长期保留 reference。因为 referenced KV 仍可淘汰，所以不会像 hard pin 一样直接把内存变成不可回收；但当大量遗留 session 都变成 referenced 后，两层优先级会逐渐失去区分度，效果退化回接近普通策略。

评审中明确提出 TTL，作者同意留作后续。这是生产接入时最重要的 lifecycle 风险。

建议后续至少增加：

- session ref TTL / idle timeout；
- per-session protected-token budget；
- 全局 referenced-token budget；
- stale session、open/close、referenced eviction 指标。



### 中关注：切换分区会改变 recency

SWA/Mamba 在 `session_ref` 发生 0↔1 变化时调用 `reset_node_mru()`，即节点不仅换分区，还会被放到新分区的 MRU 端。

所以一个原本很冷的 session 刚 close 后，会变成 unreferenced MRU，可能比真正近期访问过的 unreferenced KV 活得更久。这在 review 中被指出，作者认为对 agent workload 而言，按 close 顺序淘汰仍是可接受近似，并把保留原 recency 留作后续优化。

### 中关注：验证集中在 Full-only

代码最复杂、最容易出错的是 SWA/Mamba 分区 LRU、级联淘汰和 host 层，但 feature-specific 行为测试主要是 Full-only CPU mock。现有 Unified/HiCache 回归测试能兜住一部分结构错误，不能替代 session-aware 专项 E2E。

### 中关注：跨 rank / 跨 replica 的 locality 不由本 PR 保证

reference 是每棵本地 `UnifiedRadixCache` tree 的状态。若同一 session 的后续 turns 被路由到不同 DP rank、不同实例或不同 PD worker，本地被保护的 KV 也无法在另一棵 tree 上直接命中。

因此生产收益依赖：

- session affinity 或 cache-aware routing；
- close 控制消息能到达所有曾承载该 session KV 的 cache owner；
- 编排层稳定使用同一顶层 `session_id`。

本 PR 是 cache primitive，不是完整的分布式 session placement 方案。

### 中低关注：缺少可观测性

代码中留有 TODO，要给 session reference 增加更细日志。目前缺少：

- referenced tokens by component/layer；
- active session ref 数；
- close/release 数和耗时；
- stale-generation registration skip 数；
- unreferenced vs referenced eviction 数；
- frontier fallback/recede 数。

没有这些指标，线上很难判断“功能没有收益”究竟是漏传 ID、漏 close、路由不稳定，还是 KV 压力已经逼迫系统淘汰 referenced entries。

### 低关注：启动 guard 偏晚

最终 guard 位于 `create_tree_cache()` 中，在 cache factory 返回后才检查实际类型。它能可靠 fail fast 到启动阶段，但不一定早于 KV pool/cache 构造。review 曾建议放到 ServerArgs 或更早的 build 阶段，以免配置错误后做多余初始化。

### 明确不在范围内

- storage-backed L3 session-aware eviction；
- hard pin / exact KV range pinning；
- cache-aware admission control；
- priority decay；
- branch-aware policy；
- session affinity / router integration；
- 自动 prompt/context reconstruction。



## 16. 我认为合理的后续优先级



### P0：补正确性 E2E

- Full + SWA + Mamba 各自 device/host 两层的 unreferenced-first 测试；
- close、in-flight finish、close/reopen、abort 的 HTTP E2E；
- node split、frontier advance/recede、同 session 分支、共享 prefix；
- DP/PD 下 session_id 传播与 close fan-out。



### P1：生命周期兜底和指标

- TTL/idle timeout；
- global/per-session reference budget；
- stale generation、referenced tokens、eviction tier metrics；
- close ack 或至少可观测的异步 completion。



### P1：保留 recency 的 O(1) 分区迁移

把“引用状态”和“访问新旧”分离，避免 close 等于一次访问。可考虑每 component 独立 timestamp/sequence，或支持不改变相对顺序的跨分区 splice。

### P2：完整性能实验

至少报告：

- device hit / host hit / miss；
- input throughput、TTFT、E2E、host load-back 流量；
- CPU overhead 和 scheduler latency；
- batch/concurrency/session 数/turn gap 多维 sweep；
- 有/无 session affinity；
- 漏 close 与 TTL 场景。



## 17. 关键文件导航


| 文件                                                                                                                                                   | 作用                                                        |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `[session_ref_tracker.py](../python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py)`                                                      | generation、tombstone、register/release 总控                  |
| `[tree_component.py](../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py)`                                                     | per-component frontier 索引、coverage 抽象、删除回退                |
| `[full_component.py](../python/sglang/srt/mem_cache/unified_cache/components/full_component.py)`                                                     | Full path ref 和 heap 排序                                   |
| `[swa_component.py](../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py)`                                                       | window coverage、device/host 分区 LRU                        |
| `[mamba_component.py](../python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py)`                                                   | frontier state ref、device/host 分区 LRU                     |
| `[unified_tree_core.py](../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py)`                                                          | `session_ref` 字段、LRU sentinel/cursor、cascade、sanity check |
| `[unified_radix_cache.py](../python/sglang/srt/mem_cache/unified_radix_cache.py)`                                                                    | tracker composition、finished request 注册、公开委托 API          |
| `[scheduler.py](../python/sglang/srt/managers/scheduler.py)`                                                                                         | request generation 绑定、open/close 路由                       |
| `[registry.py](../python/sglang/srt/mem_cache/registry.py)`                                                                                          | Unified-only 配置检查                                         |
| `[test_session_unified_radix_cache.py](../test/registered/unit/mem_cache/test_session_unified_radix_cache.py)`                                       | 新增定向单测                                                    |
| [合入时文档](https://github.com/sgl-project/sglang/blob/056474cdb0689ee5751c10f9c78e39e33c451e87/docs_new/docs/advanced_features/session_radix_cache.mdx) | 官方用户语义和启动示例                                               |


注：本仓库在 PR 合入后把 `docs_new/` 重命名为 `docs/`，因此最后一项使用固定 commit 的 GitHub 链接。

## 18. 适合继续追问的问题

后续可以直接基于本文问，例如：

1. 用一个具体 radix tree 例子手推 Full/SWA/Mamba 的 `session_ref` 如何变化。
2. 分析同一个 session 分支时 shared ancestor 的 ref count 是否合理。
3. 逐行讲解 `UnifiedLRUList` 的 `mid`/`cursor` 为什么能保证 unreferenced-first。
4. 分析 `_can_evict_leaf_atomically()` 和 `_cascade_evict()` 是否存在遗漏。
5. 设计 TTL、budget 和 metrics 的后续 PR。
6. 补一套 SWA/Mamba/HiCache 的测试矩阵。
7. 分析 DP/PD 部署下 `/close_session` 应如何 fan-out。
8. 评估是否应该让 close 立即 free，还是继续保持 dereference-only。
9. 设计不改变原 recency 的 O(1) 分区迁移。
10. 根据你的实际 agent workload 判断这个特性是否值得开启，以及需要什么路由配套。



## 19. 最终评价

PR #29173 的真正贡献不是“给 cache node 加一个 session_id”，而是把 session lifecycle 映射成 UnifiedRadixCache 三种异构 component 的 **可回退软优先级**。它避免了 StreamingSession hard pin 的容量死锁风险，也保留了原 eviction policy 和正常 radix 复用。

从设计上看，generation、per-component coverage、leaf 删除回退和 cascade protection 都抓住了正确性要点；从工程成熟度看，TTL、recency preservation、observability 和 E2E coverage 还没有完成。因此它适合作为 agent-aware cache 管理的底层 primitive，而不应被误解为已经完整解决 agentic workload 的缓存调度问题。

## 20. 潜在问题：全 Agent 工作负载下，只有 `session_id` 仍然不够



### 20.1 当前策略在“全部都是活跃 Agent”时会怎样

`session_id` 只能告诉引擎一段 KV 是否仍被活跃 session 引用，不能告诉引擎多个活跃 session 中谁更值得保留。

当显存中的 agent session 都没有 close、所有相关 KV 都是 referenced 时：

- Full attention 按 `(is_referenced, session_ref_count, base_eviction_priority)` 排序；
- 如果每个 prefix 都只被一个 session 使用，那么所有节点的前两个字段基本相同；
- 最终就退回原有 base policy，默认通常是 LRU；
- SWA/Mamba 在 unreferenced 分区耗尽后，也是在 referenced 分区内部继续按 LRU 淘汰。

因此，在“全部请求都是 agent、全部 session 都活跃、各自 prefix 基本独占”的情况下，当前特性只能决定 active KV 比 inactive KV 更晚淘汰，不能解决 active agent 之间应该牺牲谁，实际会近似退化成：

```text
最久没有访问的活跃 agent KV 先淘汰
```

一个例外是共享 prefix：Full component 会让 `session_ref_count` 更高的节点更晚淘汰，因此被多个 session 共用的 system prompt、工具定义等仍会得到额外保护。

### 20.2 “轮数越多，优先级越高”方向合理，但不够准确

直觉上：

```text
agent turn 越多
→ 上下文通常越长
→ KV 被淘汰后的重新 prefill 成本越高
→ 应该更晚淘汰
```

这个方向是合理的，但 turn count 只是 recompute cost 的代理指标，并不稳定：

- 20 轮对话可能经过 history compression，实际只有 4K tokens；
- 3 轮 agent 请求可能包含大段代码、检索结果和工具输出，已经达到 80K tokens；
- 一个 100K-token session 可能正执行半小时的外部工具，短期内不会再次访问；
- 一个 20K-token session 可能 100ms 后就会进入下一轮；
- device KV 被淘汰后若 host 上仍有备份，代价是 load-back，不是完整重新 prefill；
- SWA 只复用最近 window，完整 turn 数和历史长度不能代表实际 coverage；
- Mamba 复用的是 frontier state checkpoint，更不能直接用 turn 数估算价值。

引擎已经能看到 prompt tokens、matched prefix、tree depth、device/host residency 等信息，所以调用方不一定需要显式传“第几轮”。若目标是估计计算损失，实际 cached prefix 长度通常比 turn count 更直接。

### 20.3 真正需要衡量的是“预计淘汰损失”

更合理的目标不是保护最长或轮数最多的 session，而是估计保留每段 KV 能避免多少未来成本：

```text
expected_saved_cost
≈ P(reuse soon)
 × recompute_or_restore_cost
 × business/QoS weight
```

还要除以其显存占用，得到单位资源价值：

```text
cache_value
≈ expected_saved_cost / resident_bytes
```

可以进一步写成概念模型：

```text
value(node) =
    expected_reuse_probability(node)
    × marginal_recompute_or_restore_cost(node)
    × session_business_priority
    × sharing_factor(node)
    / resident_bytes(node)
```

优先淘汰 `value` 最低的节点。

这里强调 `marginal`：如果一个长 session 的 tail 被淘汰，但较短 ancestor 仍然保留，下一轮只需要从 surviving prefix 之后重算。因此淘汰损失是“相对于最近仍驻留 ancestor 的增量重算成本”，不一定等于整个 session 的总长度。

### 20.4 长 prefix 并不应该无条件获胜

长 session 的 prefill 成本更高，但也占用更多 KV。若简单规定“越长、轮数越深就越晚淘汰”，可能出现：

- 少数超长 session 垄断 cache；
- 新 session 持续 thrash，几乎得不到复用机会；
- 已经长时间没有下一轮的深 session 挤掉即将使用的短 session；
- 形成正反馈：被保护的 session 越来越深，因此永久压制新 session。

例如，保留一个 100K-token session 可能导致十个 10K-token session 全部 miss。哪种选择吞吐更高，取决于复用概率、恢复成本和服务目标，而不只取决于单条 session 的 prefill 绝对成本。

因此 value score 还需要 aging、公平性或最大预算，防止高价值 session 永久不被淘汰。

### 20.5 调用方与引擎的职责应该分开

比较合理的接口分工是：

调用方或 agent orchestrator 提供它独有的信息：

- `session_id`：生命周期身份；
- `close_session`：生命周期结束；
- 可选 `session_priority`：业务/QoS 重要性；
- 可选 TTL、lease 或 `expected_next_use_ms`；
- 可选 deadline、trajectory 类型或任务权重。

引擎自己推导运行时信息：

- cached/matched prefix token 数；
- tree node 深度和增量 recompute tokens；
- device/host/storage residency 与恢复成本；
- 多 session 共享程度；
- KV 实际占用；
- 最近访问时间；
- eviction 后仍能保留的最近 ancestor。

turn count 可以作为一个可选 hint，但不应该成为唯一或最主要的 cache priority。

### 20.6 一个可逐步落地的策略

不必第一步就实现完整概率模型。一个渐进式 eviction hierarchy 可以是：

```text
第一层：unreferenced / closed session 优先淘汰
第二层：低业务 session_priority 优先淘汰
第三层：低预计增量重算/恢复成本优先淘汰
第四层：共享较少的 KV 优先淘汰
第五层：原有 LRU/priority policy
```

Full component 可以概念性扩展为：

```python
(
    is_referenced,
    effective_session_priority,
    estimated_marginal_saved_cost,
    session_ref_count,
    base_eviction_priority,
)
```

实际实现需要统一好排序方向，使低业务优先级、低 saved cost、少共享的节点先被淘汰。

SWA/Mamba 当前只有 referenced/unreferenced 两段 LRU。如果加入业务优先级或 value score，需要扩展为多级 partition，或者改成能动态更新分数的 heap/queue；同时要继续保持 device/host eviction 和 component cascade 的一致性。

### 20.7 Eviction policy 不能替代 admission control

如果所有活跃 session 的 referenced working set 已经超过显存容量，任何 eviction policy 都只能决定牺牲谁，不能消除重算和 thrashing。

更完整的系统需要在调度入口判断 cache working set 是否超预算：

```text
预测 referenced working set 即将超预算
        ↓
限制同时活跃的 session 数
延迟/暂停下一轮 admission
主动把低价值 session demote 到 host
或者降低其保护等级
```

所以完整的 agent-aware KV 管理至少包含三层：

```text
session_id + close
    解决 active 与 inactive 的区别

priority/cost-aware eviction
    解决多个 active session 之间保留谁

admission/concurrency control
    防止 active working set 从根本上长期超过容量
```

结论：对本 PR 的第一阶段目标而言，`session_id` 足够建立生命周期感知；对全量 agent workload 下的最优 KV 调度而言，它明显不够。后续更合理的方向是由引擎计算真实 prefix/恢复成本，由调用方提供业务优先级和预计复用时间，再配合预算、公平性、TTL 和 admission control，而不是简单只按 session turn 数排序。
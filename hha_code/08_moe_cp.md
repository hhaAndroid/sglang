# 08. MoE 里的 CP：长 prefill 的 sequence 维度并行

本文学习 CP：

```text
CP = Context Parallel
```

在 SGLang 这里，更准确地说是：

```text
attention context parallel
attn_cp_size
```

它和 TP / EP / DPA 的切分维度都不一样：

```text
TP:
  切 hidden / head / intermediate 等模型维度

EP:
  切 expert 数量

DPA:
  切请求流 / batch 流

CP:
  切同一个请求的 prefill token sequence
```

所以 CP 主要解决的是：

```text
长 prompt prefill 太重
把一个长序列按 token 维度拆给多个 CP rank
每个 CP rank 只处理其中一部分 token
```

它不是 MoE expert 并行，但它会影响 MoE。原因是：

```text
attention 阶段:
  token 可以被 CP 切开

MoE 阶段:
  router/topk/expert 计算通常需要一个 MoE group 内看到完整 token 集合
  所以 attention 后、进入 MoE 前，往往需要 CP all-gather / token 重组
```

这是本文最重要的主线。

## 0. 先给结论

CP 是一个比较“专门”的优化，不是默认所有 MoE 模型都应该打开。

它适合：

```text
长上下文 prefill
prefill 计算/显存成为瓶颈
模型和 attention backend 明确支持 CP
```

它不适合当成普通吞吐优化乱开。对 RL rollout 来说：

```text
短 prompt / decode 为主:
  先看 DPA + DeepEP

长 prompt prefill 很重:
  再考虑 CP
```

SGLang 当前参数主线是：

```bash
--enable-prefill-cp
--cp-strategy zigzag 或 interleave
--attn-cp-size N
```

旧参数仍然能看到：

```bash
--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode in-seq-split
--dsa-prefill-cp-mode round-robin-split
```

它们和新参数大致对应：

```text
in-seq-split      -> zigzag
round-robin-split -> interleave
```

注意：SGLang 的 CP 参数命名正在从旧的 DSA/NSA 专用名字迁移到更通用的 `prefill-cp` 名字。你看官方文档、旧脚本、当前源码时可能会同时看到两套写法，学习时先按“同一类 prefill CP 功能的不同入口”理解，真正运行时以当前分支 `--help` 和 server_args 为准。

## 1. CP 切的到底是什么

假设一个请求 prefill 有 16 个 tokens：

```text
token0 token1 token2 token3 token4 token5 token6 token7
token8 token9 token10 token11 token12 token13 token14 token15
```

如果：

```text
attn_cp_size = 4
```

那么 CP 会把这 16 个 tokens 拆给 4 个 CP rank。

### 1.1 interleave / round-robin

`interleave` 也就是旧文档里的 `round-robin-split`：

```text
cp rank0:
  token0, token4, token8, token12

cp rank1:
  token1, token5, token9, token13

cp rank2:
  token2, token6, token10, token14

cp rank3:
  token3, token7, token11, token15
```

源码里 `layers/cp/interleave.py` 的注释也是这个例子。

它的优点是 token 分布比较均匀，DeepSeek V3.2 文档里也说它支持：

```text
multi-batch prefill
fused MoE
FP8 KV cache
```

但它不能和 DP attention 一起开。

### 1.2 zigzag / in-seq-split

`zigzag` 也就是旧文档里的 `in-seq-split`。

如果：

```text
cp_size = 4
```

它会把序列切成：

```text
2 * cp_size = 8 个 block
```

然后每个 CP rank 拿一个前段 block 和一个后段 block：

```text
cp rank0:
  block0, block7

cp rank1:
  block1, block6

cp rank2:
  block2, block5

cp rank3:
  block3, block4
```

这个策略更像是为了长序列 attention 的负载均衡设计。DeepSeek V3.2 文档里对 `in-seq-split` 的限制更强：

```text
prefill batch size = 1
moe_dense_tp_size = 1
moe_a2a_backend = deepep
ep_size = tp_size
tp_size > dp_size
```

## 2. CP 和 TP / DPA 的关系

SGLang 初始化 attention 相关并行组时有这个公式：

```text
attn_dp_size = attention_data_parallel_size
attn_cp_size = attention_context_model_parallel_size
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size
```

换成启动参数理解：

```text
如果不开 DPA:
  attn_dp_size = 1
  attn_tp_size = tp_size / attn_cp_size

如果开 DPA:
  attn_dp_size = dp_size
  attn_tp_size = tp_size / dp_size / attn_cp_size
```

这说明 CP 会“吃掉”一部分原本属于 attention TP 的 rank。

例如：

```text
tp_size = 8
dp_size = 2
enable_dp_attention = true
attn_cp_size = 4
```

那么：

```text
attn_tp_size = 8 / 2 / 4 = 1
```

也就是：

```text
2 路 DP 请求流
每一路里面有 4 个 CP rank
attention TP 维度变成 1
```

rank 形状可以画成：

```text
global ranks:
  0 1 2 3 4 5 6 7

dp rank0:
  cp rank0: global rank0
  cp rank1: global rank1
  cp rank2: global rank2
  cp rank3: global rank3

dp rank1:
  cp rank0: global rank4
  cp rank1: global rank5
  cp rank2: global rank6
  cp rank3: global rank7
```

这时 HTTP 对外仍然是一个 URL：

```text
http://$SERVER_IP:30000
```

如果开了 DPA，请求可以路由到不同 DP rank：

```text
routed_dp_rank: 0 或 1
```

但不能路由到某个 CP rank。CP rank 是一个 DP 请求流内部的 attention sequence 切分，对外不可见。

## 3. 启动例子一：DPA + DeepEP + zigzag CP

这个是最接近“长 prefill + MoE + DeepEP”的学习例子。

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 2 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --attn-cp-size 4 \
  --max-running-requests 32 \
  --host 0.0.0.0 \
  --port 30000
```

旧写法大致是：

```bash
--enable-dsa-prefill-context-parallel \
--dsa-prefill-cp-mode in-seq-split
```

内部并行形态：

```text
tp_size = 8
dp_size = 2
enable_dp_attention = true
attn_cp_size = 4
attn_tp_size = 1

ep_size = 8
moe_a2a_backend = deepep
moe_tp_size = 1
```

对请求流的理解：

```text
DP rank0:
  可以处理一批请求
  每个长 prefill 被 CP4 切开

DP rank1:
  可以处理另一批请求
  每个长 prefill 也被 CP4 切开

MoE:
  DeepEP 在 8 个 ranks 上做 expert dispatch
```

注意：在当前 SGLang 代码里，DSA / MLA prefill CP 这类路径会自动强化一些限制：

```text
enable_dp_attention = true
moe_dense_tp_size = 1
moe_a2a_backend = deepep
ep_size = tp_size
attn_cp_size = tp_size / dp_size
prefill cuda graph disabled
```

所以如果你是 MoE + DeepEP + CP，心智模型应该是：

```text
DPA:
  多路请求流

CP:
  每路请求流内部，把长 prefill sequence 切开

DeepEP:
  MoE 阶段把 token dispatch 到 expert ranks
```

## 4. 启动例子二：fused MoE + interleave CP

DeepSeek V3.2 文档里还有一个 `round-robin-split` 例子，也就是现在的 `interleave`：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --attn-cp-size 8 \
  --max-running-requests 32 \
  --host 0.0.0.0 \
  --port 30000
```

内部并行形态：

```text
tp_size = 8
dp_size = 1
attn_cp_size = 8
attn_tp_size = 1
```

这个模式不走 DPA。文档描述它相比 `zigzag / in-seq-split`：

```text
支持 multi-batch prefill
支持 fused MoE backend
支持 FP8 KV cache
不能和 DP attention 一起开
```

如果没有显式设置：

```text
--moe-a2a-backend deepep
```

那它不是前面第 4、5、6 篇的 DeepEP 主线，而更接近 fused MoE / standard dispatcher 路径。

所以对你的 RL 业务来说，这个例子不是优先生产形态；它更适合理解：

```text
CP 如何服务长 prefill
CP 如何和 fused MoE 组合
```

## 5. 一个 layer 内 CP 通信图

以 `tp8 + dp2 + cp4 + deepep` 为例。

一个请求被路由到：

```text
DP rank0
```

它内部有 4 个 CP rank：

```text
rank0 rank1 rank2 rank3
```

一个长 prompt 被 CP4 切开：

```text
原始 tokens:
  t0 t1 t2 ... t15

CP 后:
  rank0: 一部分 tokens
  rank1: 一部分 tokens
  rank2: 一部分 tokens
  rank3: 一部分 tokens
```

简化 layer 流程：

```text
input_ids / positions
   |
   |  CP split
   |  每个 CP rank 只拿自己的 token shard
   v
embedding / hidden_states
   |
   v
attention prefill
   |
   |  CP 通信：
   |  - KV cache / index / hidden states 可能需要 CP all-gather
   |  - 具体取决于 attention backend 和 cp_strategy
   v
attention output, still CP-sharded
   |
   |  进入 MoE 前：
   |  如果 attn_cp_size > moe_dp_size
   |  SGLang 会让 MoE group 包含 CP partners
   |  并做 moe_cp all-gather
   v
MoE router / topk / DeepEP dispatch
   |
   |  DeepEP all-to-all dispatch/combine
   v
post-MoE hidden
   |
   |  后续层继续按 CP layout 执行
   v
next layer
```

这里最关键的是：

```text
CP 不是让每个 expert 只看一部分 token 后就结束
MoE 前需要把 CP 切开的 token 在 MoE group 内恢复/聚合
```

对应源码里有这个逻辑：

```text
parallel_state.py:
  if attn_cp_size > moe_dp_size:
      _MOE_DP = _ATTN_CP

communicator.py:
  moe_cp allgather
  gather tokens from cp_per_moe CP ranks so each rank holds all tokens for its MoE group
```

## 6. CP 和 MoE 的关键关系

### 6.1 CP 主要影响 attention，不直接切 expert

CP 的并行组叫：

```text
ATTN_CP
```

不是：

```text
MOE_EP
MOE_TP
```

所以 CP 本身不决定：

```text
每张卡有多少 experts
每个 expert 是否做 MoE TP
```

这些仍然由：

```text
ep_size
moe_dp_size
moe_tp_size = tp_size / ep_size / moe_dp_size
```

决定。

### 6.2 CP 会改变 attention TP

公式是：

```text
attn_tp_size = tp_size / dp_size / attn_cp_size
```

所以开 CP 后，attention TP 会变小。

例如：

```text
tp_size = 8
dp_size = 2
attn_cp_size = 4

attn_tp_size = 1
```

这也是为什么 DSA / MLA CP 代码里有注释说要保持：

```text
attn_tp_size == 1
```

因为相关 CP communicator 没有在这条路径里处理 attention TP partial output 的 all-reduce。

### 6.3 CP 和 DeepEP 可以组合，但通常是特定模式

`zigzag / in-seq-split` 路径会把配置收敛到：

```text
moe_a2a_backend = deepep
ep_size = tp_size
moe_dense_tp_size = 1
```

这说明它的主线是：

```text
attention:
  用 CP 拆长 prefill

MoE:
  用 DeepEP 做 token dispatch
```

### 6.4 CP 和 DPA 不是一回事

DPA 是不同请求流：

```text
request A -> dp rank0
request B -> dp rank1
```

CP 是同一个请求内部的长 sequence 切分：

```text
request A 的 long prompt:
  cp rank0 算一部分 token
  cp rank1 算一部分 token
  cp rank2 算一部分 token
  cp rank3 算一部分 token
```

所以：

```text
DPA:
  面向吞吐 / 多请求并发

CP:
  面向长 prefill / 单请求长序列
```

## 7. 约束和容易踩的点

### 7.1 CP 是 prefill 优化，不是 decode 主线

源码里很多判断都是：

```text
forward_batch.forward_mode.is_context_parallel_extend()
```

也就是 extend / prefill 阶段才走 CP。

decode 阶段通常不是 CP 的主要收益点。

### 7.2 `tp_size` 必须能被 `attn_cp_size` 整除

参数检查里有：

```text
tp_size % attn_cp_size == 0
```

如果还开 DPA，还需要：

```text
tp_size % (dp_size * attn_cp_size) == 0
```

否则无法得到整数：

```text
attn_tp_size = tp_size / dp_size / attn_cp_size
```

### 7.3 `attn_cp_size` 和 `moe_dp_size`

SGLang 有这个限制：

```text
if attn_cp_size != moe_dp_size:
  assert moe_dp_size == 1
```

也就是说：

```text
attn_cp_size != moe_dp_size
```

只有在：

```text
moe_dp_size = 1
```

时支持。

### 7.4 Aiter allreduce fusion 不支持 CP

参数检查里有：

```text
Aiter allreduce fusion is not supported with context parallelism
```

所以如果开了相关 fusion，需要关掉。

### 7.5 CP 策略要显式设置

当前新参数里，如果开：

```bash
--enable-prefill-cp
```

就需要配：

```bash
--cp-strategy zigzag
```

或：

```bash
--cp-strategy interleave
```

旧参数会自动映射，但新文档建议直接用新参数。

### 7.6 当前 CP 仍然偏实验性质

DeepSeek V3.2 文档里明确写了：

```text
DSA long sequence context parallel optimization 是 experimental
只在 Hopper machines 验证
```

所以生产使用前应该单独压测：

```text
正确性
prefill latency
decode throughput
显存
和 DeepEP / DPA / chunked prefill 的相互影响
```

## 8. 和前面文档的关系

### 8.1 和第 5 篇 DPA + EP

第 5 篇是：

```text
DPA:
  不同请求可以走不同 DP rank

EP / DeepEP:
  MoE token dispatch 到 experts
```

第 8 篇新增的是：

```text
CP:
  一个请求内部的长 prompt 被多个 CP rank 共同处理
```

### 8.2 和第 6 篇业务配置

第 6 篇业务配置是：

```text
tp_size = 64
dp_size = 8
enable_dp_attention = true
ep_size = 64
moe_a2a_backend = deepep
attn_cp_size = 1
```

如果未来要给它加 CP，理论上你会变成类似：

```text
tp_size = 64
dp_size = 8
attn_cp_size = 8
attn_tp_size = 64 / 8 / 8 = 1
```

但当前 CP 路径是否支持这种 64 卡多机形态，要以具体模型、backend、SGLang 版本和文档为准。不要直接把单机 CP8 的例子外推到 64 卡生产。

### 8.3 和第 7 篇 Hybrid EP + MoE TP

第 7 篇是：

```text
ep_size < tp_size
moe_tp_size > 1
```

它研究的是 expert 权重的二维切分。

第 8 篇 CP 研究的是：

```text
attn_cp_size > 1
```

它研究的是 attention prefill 的 sequence 切分。

二者不是一个维度。

## 9. 源码阅读路径

建议按这个顺序看：

```text
1. 参数
   python/sglang/srt/server_args.py

   重点看：
   --attention-context-parallel-size / --attn-cp-size
   --enable-prefill-cp
   --cp-strategy
   _normalize_context_parallel_aliases()
   _handle_context_parallelism()

2. 并行组
   python/sglang/srt/distributed/parallel_state.py

   重点看：
   attn_tp_size = tp_size // attn_cp_size // attn_dp_size
   _ATTN_CP group
   _ATTN_TP group
   attn_cp_size > moe_dp_size 时 _MOE_DP = _ATTN_CP

3. CP strategy
   python/sglang/srt/layers/cp/base.py
   python/sglang/srt/layers/cp/zigzag.py
   python/sglang/srt/layers/cp/interleave.py

   重点看：
   zigzag 和 interleave 的 token layout

4. CP metadata / all-gather
   python/sglang/srt/layers/utils/cp_utils.py

   重点看：
   prepare_context_parallel_metadata()
   cp_all_gather_reorganized_into_tensor()
   cp_all_gather_reorganized_into_tensor_kv_cache()

5. MoE 前的 CP 聚合
   python/sglang/srt/layers/communicator.py

   重点看：
   moe_cp allgather
   gather tokens from cp_per_moe CP ranks

6. Qwen3 MoE / DeepSeek MoE 模型入口
   python/sglang/srt/models/qwen3_moe.py
   python/sglang/srt/models/deepseek_v2.py
   python/sglang/srt/models/deepseek_v4.py

   重点看：
   forward_batch.attn_cp_metadata
   prepare_context_parallel_metadata()
```

## 10. 一句话总结

CP 的核心是：

```text
把一个长 prefill sequence 按 token 维度切给多个 CP rank
降低单个 rank 的 attention prefill 压力
```

在 MoE 模型里要额外记住：

```text
CP 切的是 attention tokens
EP / DeepEP 切的是 experts / expert dispatch
进入 MoE 前，CP 切开的 token 往往需要在 MoE group 内 all-gather / 重组
```

所以对你的学习路线来说：

```text
短请求高吞吐:
  DPA + DeepEP 是主线

长 prompt prefill:
  在 DPA + DeepEP 基础上再研究 CP
```

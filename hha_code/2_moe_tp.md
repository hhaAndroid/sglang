# 2. MoE 模型里的最简单 TP

本文只讨论 MoE 模型里最简单的 Tensor Parallelism，不讨论 EP、DPA、PP、CP、DeepEP。

建议启动配置：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size N \
  --ep-size 1 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1
```

这个配置下：

```text
tp_size = N
dp_size = 1
pp_size = 1
ep_size = 1
moe_a2a_backend = none

attn_tp_size = N
moe_ep_size = 1
moe_dp_size = 1
moe_tp_size = N
```

一句话总结：

> 最简单 MoE TP：router/gate 是复制的，所有 routed experts 每张卡都有，但每个 expert 内部的 FFN intermediate 维度被 TP 切开，最后通过 all-reduce 把 partial output 求和。

## 0. 如果是 TP16，启动和内部有什么区别

这里按实际常见的两机 `2 x 8` 卡来讲。

### 0.1 两机 2 x 8 卡

如果每台机器 8 卡，总共 16 卡，需要两台机器都启动同一条 server 命令，只是 `--node-rank` 不同，并且指定同一个 `--dist-init-addr`。

node 0：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr $NODE0_IP:50000 \
  --nnodes 2 \
  --node-rank 0
```

node 1：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr $NODE0_IP:50000 \
  --nnodes 2 \
  --node-rank 1
```

这里的 `--tp-size 16` 是全局 TP size，不是每台机器 16。两机 8 卡时，每个 node 会分到 8 个 TP rank。

这里有两个端口概念，不要混：

```text
--host / --port:
  对外 HTTP 服务地址，例如 http://$NODE0_IP:30000

--dist-init-addr:
  torch.distributed 初始化地址，例如 $NODE0_IP:50000
  它不是 HTTP 推理 URL
```

### 0.2 TP16 对外 HTTP URL 是几个

对 RL rollout 或外部 client 来说，纯 TP16 是一个模型实例，所以业务 HTTP URL 只有一个：

```text
base_url = http://$NODE0_IP:30000

OpenAI API:
  http://$NODE0_IP:30000/v1/chat/completions
  http://$NODE0_IP:30000/v1/completions

SGLang native API:
  http://$NODE0_IP:30000/generate
```

不要把请求打到 16 张卡，也不要按两台机器打两个业务 URL。TP16 的 16 张卡共同完成一次 forward，外部看到的是一个 server。

node 1 上虽然也会启动进程，并且可能监听同样的 `--port 30000`，但它不是完整推理入口。源码里 `Engine._launch_subprocesses()` 对 `node_rank >= 1` 的逻辑是：

```text
启动本节点 scheduler / model worker
等待 ready
不启动 tokenizer / detokenizer
启动 dummy health check server
```

这个 dummy server 主要提供：

```text
/ping
/health
/health_generate
```

所以：

```text
node 0:
  http://$NODE0_IP:30000  是业务推理入口

node 1:
  http://$NODE1_IP:30000  只适合健康检查，不适合作为 RL rollout 请求入口
```

如果 RL 侧希望有多个 URL 并发打流量，那不是 TP16 的语义，而是要启动多个模型实例，例如：

```text
2 个 TP8 replica
或者 DP / DPA / router 管理的多副本
```

### 0.3 `dist-init-addr` 是干嘛的

`--dist-init-addr $NODE0_IP:50000` 是 16 个 model worker 初始化 distributed world 的 rendezvous 地址。

在 `model_runner.py` 里，如果配置了 `server_args.dist_init_addr`，会转成：

```python
dist_init_method = "tcp://$NODE0_IP:50000"
```

然后调用：

```python
init_distributed_environment(
    world_size=self.tp_size * self.pp_size,
    rank=self.tp_size * self.pp_rank + self.tp_rank,
    local_rank=self.gpu_id,
    distributed_init_method=dist_init_method,
)
```

对于纯 TP16、PP1：

```text
world_size = tp_size * pp_size = 16 * 1 = 16
global rank = tp_rank
```

`dist-init-addr` 的作用可以理解成：

```text
让 rank 0..15 这 16 个进程找到彼此
建立 torch.distributed WORLD process group
之后再基于 WORLD 创建 TP / MoE TP / attention TP 等子 group
```

它不是请求入口，也不是负载均衡地址。实际 all-reduce/all-gather 通信由 distributed backend 处理，`dist-init-addr` 主要负责初始化阶段的 rendezvous。

实际部署时注意：

```text
$NODE0_IP:
  必须是 node 1 能访问到的 node 0 地址

50000:
  必须是空闲端口
  不要和 --port 30000 复用
  防火墙 / 容器网络要允许两台机器互通
```

### 0.4 16 张卡如何组成一个 TP group

源码里 `engine.py` 会按 node 算本机 rank 范围：

```python
tp_size_per_node = tp_size // nnodes_per_tp_group
tp_rank_range = range(
    tp_size_per_node * (node_rank % nnodes_per_tp_group),
    tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
)
```

纯 TP16、PP1、两节点时：

```text
node 0: tp_rank 0..7
node 1: tp_rank 8..15
```

每个本地 TP rank 会对应一个 scheduler / model worker 进程。`model_runner.py` 用全局 world 初始化：

```python
world_size = self.tp_size * self.pp_size
rank = self.tp_size * self.pp_rank + self.tp_rank
```

所以两机 TP16 的内部并行公式仍然不变：

```text
attn_tp_size = 16
moe_tp_size = 16
```

`parallel_state.py` 里 `initialize_model_parallel()` 会先创建 TP group。纯 TP16、PP1 时：

```text
world_size = 16
tensor_model_parallel_size = 16
num_tensor_model_parallel_groups = world_size // tp_size = 1

TP group:
  [0, 1, 2, ..., 15]

attention TP group:
  等于 TP group

MoE TP group:
  等于 TP group
```

所以两机 16 卡不是 2 个 TP8 group，而是一个跨节点 TP16 group：

```text
TP16 group
├── node 0: rank 0..7
└── node 1: rank 8..15
```

真正变化的是通信域：

```text
两机 TP16:
  all-reduce/all-gather 跨两台机器发生
  attention o_proj 聚合、MoE down_proj 聚合都会跨节点
```

### 0.5 TP16 需要额外注意什么

TP16 的数学逻辑和 TP2/TP8 一样，但更容易触发约束和性能问题。

第一，维度必须能切：

```text
attention:
  num_attention_heads 通常要能被 attn_tp_size=16 整除

MoE:
  moe_intermediate_size 要能被 moe_tp_size=16 整除
```

相关源码：

- `qwen3_5.py` 里 attention 初始化会检查 head 数和 `attn_tp_size`。
- `model_runner.py` 的 `check_quantized_moe_compatibility()` 会检查 `moe_intermediate_size % moe_tp_size`。
- 量化 MoE 还可能要求 `(moe_intermediate_size / moe_tp_size)` 满足 block size 对齐。

第二，TP16 的通信更重：

```text
attention output 聚合:
  16 卡 all-reduce

MoE output 聚合:
  16 卡 moe_tp all-reduce

logits / vocab parallel:
  可能涉及 16 卡 all-gather
```

如果 TP16 跨节点，这些通信会走跨机网络，延迟和带宽压力明显高于节点内通信。

第三，TP16 只是在“算同一个模型实例”：

```text
它不会变成 16 个副本
它不会切 expert 数量
它不会减少 router/topk 复制计算
```

如果目标是吞吐而不是单实例装载，后面要对比的是：

```text
TP16 单实例
vs
TP8 + EP8
vs
TP8 + DPA + EP8
vs
2 个 TP8 replica + router DP
```

## 1. 单层模型切分图

假设：

```text
tp_size = 2
ep_size = 1
moe_a2a_backend = none
```

单个 Transformer layer 可以画成：

```text
              ┌──────────────────────────────┐
              │        input hidden_states    │
              │        [num_tokens, hidden]   │
              └───────────────┬──────────────┘
                              │
          每个 TP rank 都有同一份 hidden_states
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   TP rank 0                                   TP rank 1
        │                                           │
        ▼                                           ▼
┌─────────────────┐                       ┌─────────────────┐
│ qkv_proj shard  │                       │ qkv_proj shard  │
│ heads 0..k      │                       │ heads k..end    │
└────────┬────────┘                       └────────┬────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                       ┌─────────────────┐
│ attention heads │                       │ attention heads │
│ local heads     │                       │ local heads     │
└────────┬────────┘                       └────────┬────────┘
         │                                         │
         ▼                                         ▼
┌─────────────────┐                       ┌─────────────────┐
│ o_proj shard    │                       │ o_proj shard    │
│ partial output  │                       │ partial output  │
└────────┬────────┘                       └────────┬────────┘
         │                                         │
         └────────────── all-reduce ───────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │   attention output / residual│
              └───────────────┬──────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ router gate: ReplicatedLinear │
              │ every rank has full gate      │
              └───────────────┬──────────────┘
                              │
          每个 TP rank 算出相同 topk expert ids
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   TP rank 0                                   TP rank 1
        │                                           │
        ▼                                           ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│ Expert 0..E all present │             │ Expert 0..E all present │
│ but each expert has     │             │ but each expert has     │
│ intermediate shard 0    │             │ intermediate shard 1    │
└────────────┬────────────┘             └────────────┬────────────┘
             │                                       │
             ▼                                       ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│ expert gate/up shard    │             │ expert gate/up shard    │
│ local intermediate      │             │ local intermediate      │
└────────────┬────────────┘             └────────────┬────────────┘
             │                                       │
             ▼                                       ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│ expert down shard       │             │ expert down shard       │
│ partial hidden output   │             │ partial hidden output   │
└────────────┬────────────┘             └────────────┬────────────┘
             │                                       │
             └────────────── all-reduce ─────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │      layer output             │
              │      [num_tokens, hidden]     │
              └──────────────────────────────┘
```

这个图里只有两类核心通信：

```text
1. attention / dense path:
   o_proj 之后 all-reduce

2. MoE expert path:
   expert down_proj 之后 all-reduce
```

没有发生：

```text
没有 expert all-to-all dispatch
没有 DeepEP dispatch/combine
没有跨 DP replica 通信
没有 pipeline stage P2P
```

原因：

```text
ep_size = 1
moe_a2a_backend = none
dp_size = 1
pp_size = 1
```

## 2. 并行组是怎么来的

源码入口：

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/distributed/parallel_state.py`

`model_runner.py` 初始化 distributed 后会调用：

```python
initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    attention_data_parallel_size=self.dp_size,
    pipeline_model_parallel_size=self.pp_size,
    expert_model_parallel_size=self.moe_ep_size,
    attention_context_model_parallel_size=self.attn_cp_size,
    moe_data_model_parallel_size=self.moe_dp_size,
)
```

`parallel_state.py` 里核心公式是：

```python
attn_dp_size = attention_data_parallel_size
attn_cp_size = attention_context_model_parallel_size
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

moe_ep_size = expert_model_parallel_size
moe_dp_size = moe_data_model_parallel_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

只开 TP 时：

```text
attn_dp_size = 1
attn_cp_size = 1
moe_ep_size = 1
moe_dp_size = 1

attn_tp_size = tp_size
moe_tp_size = tp_size
```

所以：

```text
attention 用完整 TP group
MoE expert 内部也用完整 TP group
```

## 3. Attention 侧 TP

Qwen3.5 full attention 层在：

```text
python/sglang/srt/models/qwen3_5.py
```

关键字段：

```python
self.attn_tp_rank = get_parallel().attn_tp_rank
self.attn_tp_size = get_parallel().attn_tp_size
```

只开 TP 时：

```text
attn_tp_size = tp_size
```

然后构造：

```python
self.qkv_proj = QKVParallelLinear(
    ...,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)

self.o_proj = RowParallelLinear(
    ...,
    reduce_results=False,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)
```

直观理解：

```text
qkv_proj:
  column parallel
  每张卡负责一部分 heads / channels

attention:
  每张卡只计算自己的 local heads

o_proj:
  row parallel
  每张卡得到 partial hidden contribution
```

在普通 Transformer TP 里，`o_proj` 之后需要聚合，因为每张卡只算了部分 head 对 hidden 的贡献。

SGLang 里 Qwen3.5 层还有 `LayerCommunicator`，所以聚合可能被融合、延迟或和 residual/norm 路径组合，不一定表现为 `o_proj.forward()` 内部立刻 all-reduce。但语义上，下一个依赖完整 hidden 的模块之前必须完成聚合。

相关源码：

- `python/sglang/srt/layers/linear.py`
  - `QKVParallelLinear`
  - `ColumnParallelLinear`
  - `RowParallelLinear`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/distributed/communication_op.py`

## 4. MoE 侧 TP

Qwen3.5 MoE 使用 `Qwen2MoeSparseMoeBlock`，定义在：

```text
python/sglang/srt/models/qwen3_moe.py
```

初始化时：

```python
self.tp_size = get_parallel().moe_tp_size
self.ep_size = get_parallel().moe_ep_size
```

只开 TP 时：

```text
self.tp_size = tp_size
self.ep_size = 1
```

MoE block 的普通 forward：

```python
router_logits, _ = self.gate(hidden_states)
topk_output = self.topk(hidden_states, router_logits)
final_hidden_states = self.experts(hidden_states, topk_output)

if self.tp_size > 1:
    final_hidden_states = moe_tensor_model_parallel_all_reduce(
        final_hidden_states
    )
```

这里要分清楚三个组件：

```text
gate:
  ReplicatedLinear
  每张 TP 卡都有完整 gate 权重
  每张卡对同一 hidden 算出同样 router_logits

topk:
  每张卡算出同样 topk expert ids

experts:
  FusedMoE
  每张卡都有所有 expert
  但每个 expert 的 intermediate 维度是切开的
```

所以只开 TP 时，token 不需要被发到某个 expert rank，因为所有 rank 都持有所有 expert 的一个 shard。

## 5. FusedMoE 里到底切了什么

源码：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

`FusedMoE.__init__` 里：

```python
self.moe_ep_size = get_parallel().moe_ep_size
self.moe_ep_rank = get_parallel().moe_ep_rank
self.moe_tp_size = get_parallel().moe_tp_size
self.moe_tp_rank = get_parallel().moe_tp_rank

assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
self._num_global_routed = num_experts - num_shared_slots
self._num_local_routed = self._num_global_routed // self.moe_ep_size
self.num_local_experts = self._num_local_routed + num_fused_shared_experts

assert intermediate_size % self.moe_tp_size == 0
self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
```

只开 TP 时：

```text
moe_ep_size = 1
moe_tp_size = tp_size

_num_local_routed = num_experts
num_local_experts = num_experts (+ fused shared experts)
intermediate_size_per_partition = moe_intermediate_size / tp_size
```

也就是说：

```text
expert 数量不切
每张卡都有全部 expert

expert 内部 FFN intermediate 维度切
每张卡只算 intermediate 的一段
```

## 6. 权重切分视角

一个 routed expert 的 FFN 可以简化成：

```text
x -> gate_up_proj -> activation -> down_proj -> y
```

TP=2 时：

```text
gate_up_proj weight:
  rank 0: W_gate_up[:, 0 : inter/2]
  rank 1: W_gate_up[:, inter/2 : inter]

down_proj weight:
  rank 0: W_down[0 : inter/2, :]
  rank 1: W_down[inter/2 : inter, :]

local output:
  y_rank0 = local_down(local_activation_0)
  y_rank1 = local_down(local_activation_1)

global output:
  y = y_rank0 + y_rank1
```

这就是为什么 MoE TP 最后要 all-reduce：

```text
每个 rank 只算了 expert intermediate 的 partial contribution
完整 y 是所有 partial y 的和
```

通信函数在：

```text
python/sglang/srt/distributed/communication_op.py
```

```python
def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_tp_group().all_reduce(input_)
```

## 7. 为什么不能整个 layer 最后只 all-reduce 一次

这个问题非常关键。

不是说物理上必须在某一行立刻 all-reduce，而是语义上必须在下游依赖完整 hidden 之前完成聚合。

一个标准 pre-norm layer 可以简化为：

```text
x
 -> attention
 -> x + attn_out
 -> post_attention_layernorm
 -> router / MoE
 -> x + moe_out
```

TP 下 attention 的 `o_proj` 每个 rank 只算一部分贡献：

```text
attn_out = attn_out_rank0 + attn_out_rank1 + ...
```

如果不先聚合，而是每个 rank 拿自己的 partial output 继续算：

```text
rank0: post_norm(x + attn_out_rank0)
rank1: post_norm(x + attn_out_rank1)
```

这和正确结果不同：

```text
post_norm(x + attn_out_rank0 + attn_out_rank1)
```

因为中间有 residual、RMSNorm/LayerNorm、router topk、activation 这些非线性或依赖完整 hidden 的操作。一般不满足：

```text
MLP(norm(x + a0 + a1)) == MLP(norm(x + a0)) + MLP(norm(x + a1))
```

MoE 里更明显：

```text
router gate 应该基于完整 hidden 选 expert
如果每个 rank 用 partial hidden 选 topk，可能每张卡选出来的 expert 不一致
```

所以通信边界是：

```text
某个模块输出是 TP partial
下游模块需要完整 hidden
=> 必须聚合
```

SGLang 可以优化通信形式：

```text
可以 fuse all-reduce + residual/norm
可以延迟一点执行
可以 reduce-scatter / gather 配合 LayerCommunicator
可以 overlap compute/comm
```

但不能越过会改变语义的点，比如 router/topk 或 norm。

结论：

```text
不能把 attention 的聚合拖到整个 layer 最后，
因为 MoE 的输入依赖完整 attention output。

MoE 之后的聚合可以尝试融合或延迟，
但在下一层需要完整 hidden 前也必须完成。
```

## 8. 和 EP 的区别

只开 TP：

```text
每张卡都有所有 expert
每个 expert 被切 intermediate 维度
通信是 moe_tp all-reduce
没有 token all-to-all
```

开 EP：

```text
每张卡只放一部分 expert
token 根据 topk 被 dispatch 到对应 expert rank
通信是 dispatch/combine，通常 all-to-all
```

所以本篇学习 TP 时先不要管 DeepEP。DeepEP 是“token 去找 expert”；最简单 TP 是“每张卡都有所有 expert 的一片，大家一起算同一个 expert”。

## 9. 对照源码阅读顺序

建议按这个顺序读：

1. `python/sglang/srt/distributed/parallel_state.py`

看：

```python
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

目标：确认只开 TP 时 `attn_tp_size == moe_tp_size == tp_size`。

2. `python/sglang/srt/models/qwen3_5.py`

看 Qwen3.5 decoder layer 如何构造 attention 和 MoE block：

```python
QKVParallelLinear(...)
RowParallelLinear(...)
Qwen2MoeSparseMoeBlock(...)
LayerCommunicator(...)
```

目标：理解 attention 输出为什么要在 MoE 前变成完整 hidden。

3. `python/sglang/srt/models/qwen3_moe.py`

看：

```python
Qwen3MoeSparseMoeBlock.__init__
Qwen3MoeSparseMoeBlock.forward_normal
```

目标：理解 gate/topk/expert/all-reduce 的顺序。

4. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

看：

```python
FusedMoE.__init__
FusedMoE.forward_impl
```

目标：理解 `num_local_experts` 和 `intermediate_size_per_partition`。

5. `python/sglang/srt/layers/linear.py`

看：

```python
ColumnParallelLinear
RowParallelLinear
QKVParallelLinear
```

目标：理解 column parallel / row parallel 的权重切法。

6. `python/sglang/srt/distributed/communication_op.py`

看：

```python
tensor_model_parallel_all_reduce
attention_tensor_model_parallel_all_reduce
moe_tensor_model_parallel_all_reduce
```

目标：区分普通 TP group、attention TP group、MoE TP group。

## 10. 本篇要掌握的检查点

学完这篇，应该能回答：

```text
1. 只开 TP 时，MoE 的 expert 数量有没有被切？
   没有。所有 rank 都有全部 expert。

2. MoE TP 切的是哪一维？
   expert FFN 的 intermediate 维度。

3. gate/router 是切分的还是复制的？
   复制的。ReplicatedLinear。

4. topk 每个 rank 一样吗？
   只开 TP 时应该一样，因为输入 hidden 在 router 前已经是完整且一致的。

5. MoE TP 最后的 all-reduce 在做什么？
   把每个 rank 的 expert partial output 求和。

6. 为什么不能只在整个 layer 最后 all-reduce？
   因为 attention partial output 后面还有 norm/router/topk 等依赖完整 hidden 的操作。

7. TP 和 EP 最大区别是什么？
   TP 切 expert 内部计算维度；EP 切 expert 数量并需要 token dispatch。
```

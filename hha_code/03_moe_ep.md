# 03. MoE 模型里的普通 EP

本文只讨论普通 Expert Parallelism：

```text
moe_a2a_backend = none
ep_size > 1
moe_tp_size = 1
dp_size = 1
pp_size = 1
attn_cp_size = 1
moe_dp_size = 1
```

这里先不讨论 DeepEP / FlashInfer A2A / Mooncake / NIXL / MoRI。那些后端会引入 token dispatch / combine，逻辑比普通 EP 多一层，后面单独学。

## 0. 最容易误解的一点

在 SGLang 里，`ep_size` 不是额外乘到 world size 上的维度。

也就是说，下面不是 8 张卡 TP 再乘 8 张卡 EP：

```text
world_size != tp_size * ep_size
```

普通 SRT model worker 的 distributed world 是：

```text
world_size = tp_size * pp_size
```

在不启用 PP 时：

```text
world_size = tp_size
```

`ep_size` 是在 `tp_size` 这个组里面继续切 MoE experts。源码里：

```python
moe_ep_size = expert_model_parallel_size
moe_dp_size = moe_data_model_parallel_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

所以“只开 EP，不开 MoE TP”的典型配置是：

```text
tp_size = N
ep_size = N
moe_dp_size = 1

attn_tp_size = N
moe_ep_size = N
moe_tp_size = 1
```

这句话很重要：

> SGLang 的“只开 EP”并不等于完全没有 TP；attention 仍然是 TP=N，只是 MoE expert 内部不做 TP，而是按 expert 数量切到不同 rank。


## 1. EP-only 启动必须显式设置 `tp-size`

这是本文最重要的启动规则：

```text
EP-only 场景下，必须显式设置：

--tp-size N
--ep-size N

不能只写 --ep-size N
```

原因是：在 SGLang 里，EP 不是独立拉起额外 worker 的维度。普通 SRT worker 的 distributed world 来自：

```text
world_size = tp_size * pp_size
```

不启用 PP 时：

```text
world_size = tp_size
```

也就是说，`ep_size` 是在 `tp_size` 形成的 world 里面切 expert。没有足够大的 `tp_size`，就没有足够的 rank 给 EP 使用。

CLI 也不会根据 `--ep-size` 自动推导 `--tp-size`。`tp_size` 和 `ep_size` 的默认值都是 1，代码只是分别赋值：

```python
args.tp_size = args.tensor_parallel_size
args.ep_size = args.expert_parallel_size
```

所以如果你以为下面这条命令是在启动 EP8：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --ep-size 8 \
  --moe-a2a-backend none
```

实际并不是。因为你没有写 `--tp-size 8`，所以 `tp_size` 仍然是默认值 1：

```text
tp_size = 1
ep_size = 8
pp_size = 1
world_size = tp_size * pp_size = 1
```

这不是一个符合预期的 EP8。普通 EP 需要 EP ranks 存在于 TP world 里，但这里 world 只有 1 个 rank，不可能承载 8 路 EP。后续可能在参数校验、parallel group 初始化、MoE layer 初始化或 expert 映射处报错；即使没有在最早阶段报错，也一定不是你想要的 EP-only 配置。

正确的 EP-only 写法必须是：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1
```

这时才是本文说的普通 EP-only：

```text
tp_size = 8
ep_size = 8
world_size = 8

attention:
  attn_tp_size = 8

MoE:
  moe_ep_size = 8
  moe_tp_size = 1
```

另外，DeepEP / Mooncake / NIXL / FlashInfer / MoRI 这类 A2A backend 还有反向覆盖逻辑：

```python
self.ep_size = self.tp_size
```

也就是说，如果你漏写 `--tp-size N`，但写了某个 A2A backend，最后反而可能按默认 `tp_size=1` 把 `ep_size` 改回 1。这个坑后面学 A2A EP 时还会再展开。

如果你想要的是：

```text
attention 不做 TP
MoE 只做 EP
```

那不是本文的普通 EP，而是后面要学的：

```text
DPA + EP
```

典型公式会变成：

```text
tp_size = N
dp_size = N
enable_dp_attention = true
ep_size = N

attn_tp_size = 1
moe_ep_size = N
moe_tp_size = 1
```

## 2. 启动命令

### 2.1 单节点 8 卡普通 EP

假设一台机器 8 卡，想让 MoE expert 按 8 路 EP 切开：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

这时：

```text
world_size = 8

attention:
  attn_tp_size = 8

MoE:
  moe_ep_size = 8
  moe_tp_size = 1
```

如果模型有 64 个 routed experts：

```text
每个 EP rank 负责 64 / 8 = 8 个 routed experts
每个 expert 的 intermediate 维度不切
```

### 2.2 两机 2 x 8 卡普通 EP16

两机共 16 卡时：

node 0：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 16 \
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
  --ep-size 16 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr $NODE0_IP:50000 \
  --nnodes 2 \
  --node-rank 1
```

对外业务 URL 仍然只有一个：

```text
base_url = http://$NODE0_IP:30000
```

node 1 不是业务推理入口。它只参与 distributed world 里的 scheduler / model worker，并提供 dummy health check。

## 3. EP 切的是什么

继续假设：

```text
num_experts = 64
num_experts_per_tok = 8
tp_size = 8
ep_size = 8
moe_tp_size = 1
```

EP rank 和 experts 的关系是：

```text
EP rank 0: expert  0..7
EP rank 1: expert  8..15
EP rank 2: expert 16..23
EP rank 3: expert 24..31
EP rank 4: expert 32..39
EP rank 5: expert 40..47
EP rank 6: expert 48..55
EP rank 7: expert 56..63
```

源码里 `FusedMoE` 会算：

```python
self._num_global_routed = num_experts - num_shared_slots
self._num_local_routed = self._num_global_routed // self.moe_ep_size
self.num_local_experts = self._num_local_routed + num_fused_shared_experts
```

权重加载时，会把 global expert id 映射成本 rank 的 local expert id：

```python
start_idx = self.moe_ep_rank * self._num_local_routed
end_idx = start_idx + self._num_local_routed

if start_idx <= expert_id < end_idx:
    return expert_id - start_idx
else:
    return -1
```

如果返回 `-1`，这个 expert 不属于当前 rank，当前 rank 就不加载这份 expert 权重。

## 4. 单层切分图：仅适用于 `moe-a2a-backend=none`

本节这张图只描述普通 EP：

```text
moe_a2a_backend = none
dispatcher = StandardDispatcher
```

它不是 DeepEP / FlashInfer / NIXL / Mooncake / MoRI 的执行图。A2A backend 会走 token dispatch / combine，通信形态不同，后面单独画。

假设只画一个 Transformer layer，并且：

```text
tp_size = 4
ep_size = 4
moe_tp_size = 1
moe_a2a_backend = none
num_experts = 8
top_k = 2
```

expert 分布：

```text
rank 0: expert 0, 1
rank 1: expert 2, 3
rank 2: expert 4, 5
rank 3: expert 6, 7
```

单层执行可以画成：

```text
              ┌──────────────────────────────┐
              │        input hidden_states    │
              │        [num_tokens, hidden]   │
              └───────────────┬──────────────┘
                              │
              每个 TP/EP rank 都有同一批 tokens
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌─────────────────┐                       ┌─────────────────┐
│ attention TP    │        ...            │ attention TP    │
│ rank 0 heads    │                       │ rank 3 heads    │
└────────┬────────┘                       └────────┬────────┘
         │                                         │
         └──────── attention all-reduce ───────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ hidden_states after attention│
              │ 每个 rank 再次拿到完整 hidden │
              └───────────────┬──────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ router gate: ReplicatedLinear │
              │ 每个 rank 算同一份 topk        │
              └───────────────┬──────────────┘
                              │
                              ▼
        topk_ids 示例：token A -> expert 1 + expert 6
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ EP rank 0       │   │ EP rank 1       │   │ EP rank 3       │
│ owns expert 0,1 │   │ owns expert 2,3 │   │ owns expert 6,7 │
│                 │   │                 │   │                 │
│ token A hits e1 │   │ token A no hit  │   │ token A hits e6 │
│ compute e1 part │   │ output is 0     │   │ compute e6 part │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └──────────── MoE EP all-reduce ────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ final MoE output per token   │
              │ e1 contribution + e6 contrib │
              └──────────────────────────────┘
```

这个图里有两个关键点：

```text
1. 普通 EP none 后端不把 token 发送到 expert 所在 rank
2. 每个 rank 都持有同一批 tokens，但只计算本 rank 拥有的 experts
```

最后的 `moe_expert_parallel_all_reduce()` 把不同 EP rank 上的 partial output 求和，得到完整 MoE 输出。

### 4.1 为什么 EP rank 会是同一批 tokens

这里容易把 EP 和 DP 混在一起。

```text
DP / DPA:
  不同 rank 可以处理不同请求 / 不同 tokens

EP:
  不表示不同 rank 处理不同请求
  它表示不同 rank 持有不同 experts
```

在本文的普通 EP-only 配置里：

```text
tp_size = ep_size
dp_size = 1
enable_dp_attention = false
moe_a2a_backend = none
```

这些 rank 首先还是同一个 TP group 的成员。attention 阶段必须协作计算同一批 tokens：

```text
rank 0 算部分 attention heads
rank 1 算部分 attention heads
...
attention all-reduce 后，每个 rank 都拿回同一批 tokens 的完整 hidden_states
```

到 MoE 阶段时，同一组 rank 又被解释成 EP group：

```text
rank 0 持有一部分 experts
rank 1 持有一部分 experts
...
```

但是因为 `moe_a2a_backend=none` 没有 token dispatch，所以 token 不会被重新分发到 expert 所在 rank。于是每个 EP rank 继续持有同一批 tokens，只是：

```text
命中本地 expert:
  计算 contribution

没有命中本地 expert:
  contribution = 0
```

最后通过 EP all-reduce 把这些 contribution 加起来。

所以：

```text
ep8 不等于 8 路不同数据
ep8 只表示 experts 被切成 8 份
```

如果你想让不同 rank 处理不同请求，那是：

```text
DP / DPA
或者多个 server replica
```

如果你想让 MoE 阶段每个 expert rank 只收到自己要处理的 tokens，那是：

```text
DeepEP / FlashInfer / NIXL / Mooncake / MoRI 这类 A2A backend
```

但即使使用 A2A backend，在进入 MoE dispatch 之前，如果没有 DPA/DP，同一个 TP group 里仍然是在服务同一批请求；只是 MoE 阶段会把 token-expert 任务重新分发。

## 5. 通信发生在哪里

普通 EP 的主要通信点：

```text
attention:
  attention output all-reduce
  因为 attention 仍然是 attn_tp_size=N

MoE:
  experts 之后做 EP all-reduce
  把各 EP rank 的 expert 输出求和

logits:
  vocab parallel 场景下可能还有 all-gather / all-reduce
```

普通 EP 的 `none` 后端没有 DeepEP 那种 token all-to-all：

```text
没有 dispatch all-to-all
没有 combine all-to-all
```

因为 `moe_a2a_backend=none` 时，`create_moe_dispatcher()` 使用的是 `StandardDispatcher`。

`StandardDispatcher` 的核心逻辑是把 global expert id 映射成本 rank local expert id：

```text
本 rank 拥有的 expert:
  topk_id -> local_expert_id

本 rank 不拥有的 expert:
  topk_id -> -1
```

MoE runner 看到 `-1` 后会过滤掉这部分 expert 计算。最后再通过 EP all-reduce 合并。

### 5.1 为什么 `none` 后端可以用 all-reduce 代替 all-to-all

MoE 层的数学结果本质上是 top-k experts 输出的加权求和：

```text
moe_output[token] =
  w0 * expert_0(hidden)
+ w1 * expert_1(hidden)
+ ...
```

如果一个 token 选中了 expert 1 和 expert 6：

```text
token A -> expert 1 + expert 6
```

在 `moe_a2a_backend=none` 下，SGLang 不把 token A 发到 expert 1/6 所在 rank，而是让每个 EP rank 都拿到 token A：

```text
EP rank 0 owns expert 0,1:
  计算 expert 1 对 token A 的贡献

EP rank 1 owns expert 2,3:
  token A 没有命中本地 expert，贡献为 0

EP rank 2 owns expert 4,5:
  token A 没有命中本地 expert，贡献为 0

EP rank 3 owns expert 6,7:
  计算 expert 6 对 token A 的贡献
```

每个 rank 先得到同 shape 的 partial output：

```text
rank 0 output[token A] = expert 1 contribution
rank 1 output[token A] = 0
rank 2 output[token A] = 0
rank 3 output[token A] = expert 6 contribution
```

然后做一次 EP all-reduce sum：

```text
all_reduce_sum =
  expert 1 contribution
+ 0
+ 0
+ expert 6 contribution
```

这正好等于 MoE 层需要的完整输出。所以它在数学上成立的前提是：

```text
1. 每个 EP rank 看到同一份 hidden_states
2. router/gate 是复制的，每个 rank 算出同一份 topk
3. 每个 rank 只计算自己拥有的 expert contribution
4. 非本地 expert contribution 视为 0
5. 最后对 hidden-size 维度的输出做 sum all-reduce
```

源码对应关系：

```text
Qwen3MoeSparseMoeBlock.forward_normal()
  gate -> topk -> experts -> moe_expert_parallel_all_reduce()

StandardDispatcher
  本地 expert: global expert id -> local expert id
  非本地 expert: global expert id -> -1
```

所以 `none` 后端不是“不需要通信”，而是把通信放在 expert 计算之后：

```text
none:
  token 不 all-to-all
  output all-reduce

A2A backend:
  token dispatch all-to-all
  output combine
```

两者是不同实现路线：

```text
none 后端:
  简单，依赖少
  所有 EP rank 持有完整 token batch
  MoE 后通信量约是 [num_tokens, hidden_size] 的 all-reduce

A2A 后端:
  token 按 topk 发到 expert 所在 rank
  通信和 topk token dispatch 相关
  更适合大规模 EP，但依赖和调参更复杂
```

## 6. 和 TP-only 的差别

TP-only：

```text
tp_size = N
ep_size = 1
moe_tp_size = N

每张卡都有所有 experts
每个 expert 的 intermediate 维度被切成 N 份
MoE 输出后做 moe_tp all-reduce
```

普通 EP-only：

```text
tp_size = N
ep_size = N
moe_tp_size = 1

每张卡只有一部分 experts
每个 expert 内部 intermediate 维度不切
MoE 输出后做 moe_ep all-reduce
```

对一个 token 来说，如果 topk experts 分布在多个 EP rank：

```text
token 的 expert 0 在 rank 0
token 的 expert 6 在 rank 3

rank 0 算 expert 0 的贡献
rank 3 算 expert 6 的贡献
其它 rank 对这个 token 的 MoE 贡献为 0

EP all-reduce 后，每个 rank 都拿到完整 token output
```

## 7. 再强调：这是 `none` 后端的 all-reduce 路线

因为普通 EP 的 `moe_a2a_backend=none` 没有把 token 聚合到 expert 所在 rank 上完整计算。

每个 EP rank 只知道自己负责的 experts 的输出。对同一个 token：

```text
rank 0:
  只可能算 expert 0..7 的贡献

rank 1:
  只可能算 expert 8..15 的贡献

...
```

但 MoE 层的数学结果是：

```text
moe_output[token] =
  weight_1 * expert_a(hidden)
+ weight_2 * expert_b(hidden)
+ ...
```

如果 topk 里的 experts 分散在多个 EP rank，那么完整结果必须跨 rank 求和。

所以 `none` 后端的通信选择是：

```text
不提前搬 token
各 rank 本地计算自己专家的贡献
最后 all-reduce partial output
```

DeepEP / FlashInfer / NIXL 这类 A2A 后端选择的是另一条路：

```text
先把 token dispatch 到 expert 所在 rank
expert rank 计算
再 combine 回来
```

这两种路径后面要分开学。

## 8. 参数约束

普通 EP 需要满足：

```text
ep_size <= tp_size
num_experts 能被 ep_size 整除
moe_intermediate_size 能被 moe_tp_size 整除
```

如果目标是“只开 EP，不开 MoE TP”，通常设置：

```text
ep_size = tp_size
moe_tp_size = 1
```

但还要注意 attention head 的约束。普通 EP-only 下：

```text
ep_size = tp_size
attn_tp_size = tp_size
```

所以 attention 也会被 `tp_size` 切。以 Qwen3 MoE 为例，Q heads 有硬约束：

```python
assert total_num_heads % attn_tp_size == 0
```

因此：

```text
如果 tp_size > num_attention_heads:
  一定报错

如果 num_attention_heads 不能被 tp_size 整除:
  一定报错
```

KV heads 的规则不完全一样：

```python
if total_num_kv_heads >= attn_tp_size:
    assert total_num_kv_heads % attn_tp_size == 0
else:
    assert attn_tp_size % total_num_kv_heads == 0
```

所以可以记成：

```text
Q heads:
  必须能被 tp_size 整除
  tp_size 超过 Q heads 会报错

KV heads:
  如果 KV heads >= tp_size，则 KV heads 必须能被 tp_size 整除
  如果 tp_size > KV heads，只要 tp_size 能被 KV heads 整除，就会做 KV head 复制/复用，不会因为超过 KV heads 本身报错
```

例如：

```text
num_attention_heads = 16
num_key_value_heads = 8
tp_size = ep_size = 32

结果:
  Q heads: 16 % 32 != 0，报错
```

再比如：

```text
num_attention_heads = 32
num_key_value_heads = 8
tp_size = ep_size = 32

结果:
  Q heads: 32 % 32 == 0，可以
  KV heads: 32 % 8 == 0，可以做 KV head 复制/复用
```

如果设置：

```text
tp_size = 16
ep_size = 8
```

那就不是纯 EP，而是：

```text
moe_tp_size = 16 / 8 = 2
```

也就是 MoE 内部同时有：

```text
EP = 8
MoE TP = 2
```

这个组合并行后面单独学。

## 9. 对 RL rollout 的含义

普通 EP 和 TP-only 一样，对外还是一个模型实例：

```text
RL worker / rollout client 只打 node 0 的 base_url
```

例如：

```text
http://$NODE0_IP:30000/v1/chat/completions
```

不要把请求分别打到每个 EP rank。EP rank 不是独立 replica，它们共同完成同一次 forward。

如果 RL 侧想要多个可并发调度的 URL，要启动多个 server instance，例如：

```text
2 个 EP8 实例
或者 DP / DPA 管理的多副本
```

## 10. 源码阅读顺序

建议按这个顺序看：

```text
python/sglang/srt/server_args.py
  看 ep_size、moe_a2a_backend 的参数和约束

python/sglang/srt/model_executor/model_runner.py
  看 world_size = tp_size * pp_size
  看 initialize_model_parallel() 传入 ep_size

python/sglang/srt/distributed/parallel_state.py
  看 moe_ep_size / moe_tp_size 的公式
  看 MoE EP group 和 MoE TP group 怎么建

python/sglang/srt/models/qwen3_moe.py
  看 Qwen3MoeSparseMoeBlock
  看 gate、topk、experts、post experts all-reduce

python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  看 num_local_experts
  看 _map_global_expert_id_to_local_expert_id()
  看 weight_loader()

python/sglang/srt/layers/moe/token_dispatcher/standard.py
  看普通 EP 如何把非本地 expert 映射成 -1

python/sglang/srt/distributed/communication_op.py
  看 moe_expert_parallel_all_reduce()
```

## 11. 本文结论

普通 EP 的核心逻辑：

```text
attention:
  仍然走 TP

router:
  每个 rank 复制计算

experts:
  按 expert id 切到不同 EP rank
  每个 expert 内部不切

token:
  不做 A2A dispatch
  每个 rank 都看到同一批 tokens

通信:
  attention 后 TP all-reduce
  MoE experts 后 EP all-reduce
```

一句话总结：

> 普通 EP 是“expert 权重分布式 + token 复制 + 输出 all-reduce”，不是“token all-to-all dispatch”。真正的 token dispatch 是 DeepEP / FlashInfer / NIXL 等 A2A 后端要解决的问题。

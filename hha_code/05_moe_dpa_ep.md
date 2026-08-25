# 05. MoE 里的 DPA + EP：不同请求流 + expert 并行

本文讨论 MoE 模型里更接近 RL rollout 使用场景的一种组合：

```text
DPA + EP

DPA = Data Parallel Attention
EP  = Expert Parallel
```

最推荐先学习这个形态：

```text
tp_size = N
dp_size = N
ep_size = N
enable_dp_attention = true
moe_dp_size = 1
moe_tp_size = 1
```

直观理解：

```text
attention:
  每个 DP rank 可以处理不同请求 / token 流

MoE:
  所有 DP rank 共享一个 EP expert group
  expert 权重按 rank 切分
  token 根据 topk 去对应 expert rank 计算
```

这就是前一篇 DeepEP 文档最后引出的区别：

```text
DeepEP-only:
  dp_size = 1
  attention 阶段不是多个 DP 请求流
  MoE dispatch 后 expert compute token 子集才不同

DPA + EP:
  dp_size > 1
  enable_dp_attention = true
  attention 阶段开始就可以是不同 DP rank 处理不同请求
  MoE 阶段再把 token-expert 任务送到 EP expert rank
```

## 0. 先给结论

如果你想让 8 张卡各自承接不同 rollout 请求，同时 MoE experts 做 EP8，最容易理解的启动方式是：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

这里仍然只有一个业务 HTTP URL：

```text
http://$SERVER_IP:30000
```

区别是 server 内部会有 `dp_size=8` 个 DP attention worker。请求进入同一个 HTTP server 后，由 `DataParallelController` 分发到不同 DP rank。

如果你做 RL，想显式控制某个请求路由到哪个 DP rank，可以用：

```text
HTTP header:
  X-Data-Parallel-Rank: 3

request body:
  routed_dp_rank: 3
```

二者用于同一个目的：让请求进入指定 DP rank。`routed_dp_rank` 的合法范围是：

```text
0 <= routed_dp_rank < dp_size
```

## 1. DPA + EP 和前面几篇的关系

### 1.1 TP-only

第 2 篇 TP-only：

```text
tp_size = N
dp_size = 1
ep_size = 1
enable_dp_attention = false
```

attention 和 MoE dense/expert GEMM 都按 TP 切。所有 rank 合起来服务同一个 batch。

### 1.2 EP-only

第 3 篇 EP-only `none`：

```text
tp_size = N
dp_size = 1
ep_size = N
enable_dp_attention = false
moe_a2a_backend = none
```

每个 EP rank 看到同一批 tokens，本地只算自己拥有的 experts，最后 EP all-reduce。

### 1.3 DeepEP-only

第 4 篇 DeepEP：

```text
tp_size = N
dp_size = 1
ep_size = N
enable_dp_attention = false
moe_a2a_backend = deepep
```

attention 阶段仍然不是多 DP 请求流；MoE 阶段用 DeepEP dispatch/combine，让 expert compute 的 token-expert 子集分散到不同 EP rank。

### 1.4 DPA + EP

本文：

```text
tp_size = N
dp_size = N
ep_size = N
enable_dp_attention = true
```

它的核心变化是：

```text
attention 维度:
  从 tensor parallel 变成 data parallel attention

MoE 维度:
  仍然做 expert parallel
```

在 `tp_size = dp_size = ep_size = N` 时：

```text
attn_tp_size = 1
attn_dp_size = N
moe_ep_size = N
moe_tp_size = 1
```

这就是最清楚的学习版本：

```text
attention 不切 head
每个 DP rank 独立处理自己的 tokens
MoE expert 按 EP rank 切
```

## 2. 启动命令

### 2.1 单节点 8 卡：DPA + EP + DeepEP

这是最推荐作为第一个 DPA+EP 学习对象的命令：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

为什么推荐先看这个：

```text
1. attention 阶段是 DPA，每个 DP rank 可以承接不同请求
2. MoE 阶段是 EP，experts 按 rank 切分
3. MoE token 通信是 DeepEP dispatch/combine，语义最符合“token 去 expert 所在 rank”
4. attn_tp_size = 1，不会遇到 attention head 被 tp_size 切不开的问题
```

如果想指定本地 expert GEMM backend，可以加：

```bash
--moe-runner-backend deep_gemm
```

完整写法：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

注意 DeepEP 后端会在参数处理阶段强制：

```text
ep_size = tp_size
```

所以 DPA+EP+DeepEP 最自然的学习配置就是：

```text
tp_size = dp_size = ep_size
```

不要把 DeepEP 当成可以任意设置 `tp_size=16, ep_size=8` 的 backend；这种 hybrid EP + MoE TP 组合应该先放到 `none` 或其它支持该形态的 backend 下理解。

### 2.2 单节点 8 卡：DPA + EP + none

也可以写：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend none \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

但它和 DeepEP 的 MoE 通信路线不一样。

`none` 后端下，SGLang 的 sparse MoE `mlp_mode` 会走 `FULL` 路线：

```text
attention 后每个 DP rank 有自己的 local tokens
进入 MoE 前通过 DP gather 得到全局 token buffer
每个 EP rank 只计算本地 experts
MoE 输出再通过 scatter / reduce-scatter 回到各自 DP rank
```

所以：

```text
DPA + EP + none:
  更接近 gather / all-reduce / reduce-scatter 路线

DPA + EP + deepep:
  更接近 token-expert dispatch / combine 路线
```

学习时建议：

```text
先用 DPA + EP + DeepEP 理解“不同请求流 + expert dispatch”
再回头看 DPA + EP + none 的 gather / reduce-scatter 优化
```

### 2.3 两机 2 x 8 卡：DPA + EP16 + DeepEP

node 0：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --dp-size 16 \
  --enable-dp-attention \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
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
  --dp-size 16 \
  --enable-dp-attention \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr $NODE0_IP:50000 \
  --nnodes 2 \
  --node-rank 1
```

业务请求仍然建议发给 node 0：

```text
base_url = http://$NODE0_IP:30000
```

`--dist-init-addr` 仍然是给 server 内部分布式进程建组用的，不是业务请求地址。

## 3. 参数公式

源码里的 DPA rank 公式可以先记成：

```text
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

MoE 公式是：

```text
moe_ep_size = ep_size
moe_dp_size = moe_dp_size
moe_tp_size = tp_size // moe_ep_size // moe_dp_size
```

本文最常见配置：

```text
tp_size = 8
dp_size = 8
ep_size = 8
attn_cp_size = 1
moe_dp_size = 1
```

代入：

```text
attn_dp_size = 8
attn_tp_size = 8 // 8 // 1 = 1

moe_ep_size = 8
moe_tp_size = 8 // 8 // 1 = 1
```

所以每个 rank 的角色是：

```text
rank 0:
  attn_dp_rank = 0
  attn_tp_rank = 0
  moe_ep_rank = 0

rank 1:
  attn_dp_rank = 1
  attn_tp_rank = 0
  moe_ep_rank = 1

rank 2:
  attn_dp_rank = 2
  attn_tp_rank = 0
  moe_ep_rank = 2

...

rank 7:
  attn_dp_rank = 7
  attn_tp_rank = 0
  moe_ep_rank = 7
```

注意这里的 `tp_size=8` 不是说 attention 还在 TP8 切 head。因为打开 DPA 后：

```text
真正给 attention 用的是 attn_tp_size
```

而不是原始 `tp_size`。

在 `tp_size=dp_size=8` 时：

```text
attn_tp_size = 1
```

这也解释了一个非常重要的好处：

```text
DPA + EP 可以绕开 EP-only 里 attention head 不够 tp_size 切的问题。
```

比如某个模型 Q head 数是 8。如果 EP-only 写：

```text
tp_size = 16
ep_size = 16
enable_dp_attention = false
```

attention 实际 `attn_tp_size=16`，Q head 不够切，会报错。

但 DPA+EP16 写：

```text
tp_size = 16
dp_size = 16
ep_size = 16
enable_dp_attention = true
```

attention 实际：

```text
attn_tp_size = 1
```

因此 attention 不再按 16 切 head。

## 4. rank 分组图

假设：

```text
tp_size = 4
dp_size = 4
ep_size = 4
enable_dp_attention = true
attn_cp_size = 1
moe_dp_size = 1
```

### 4.1 attention 视角

```text
attn_dp_size = 4
attn_tp_size = 1
```

分组是：

```text
DP rank 0:
  rank 0
  attn_tp_group = [0]

DP rank 1:
  rank 1
  attn_tp_group = [1]

DP rank 2:
  rank 2
  attn_tp_group = [2]

DP rank 3:
  rank 3
  attn_tp_group = [3]
```

也就是说 attention 阶段每个 rank 可以独立处理自己的请求：

```text
rank 0: request A, request E
rank 1: request B
rank 2: request C, request F
rank 3: request D
```

### 4.2 MoE 视角

```text
moe_ep_size = 4
moe_tp_size = 1
```

EP group 是：

```text
moe_ep_group = [0, 1, 2, 3]
```

如果有 8 个 routed experts：

```text
rank 0 owns expert 0,1
rank 1 owns expert 2,3
rank 2 owns expert 4,5
rank 3 owns expert 6,7
```

所以同一个物理 rank 有两种身份：

```text
rank 2:
  attention 里是 DP rank 2
  MoE 里是 EP rank 2，拥有 expert 4,5
```

这就是 DPA+EP 的核心：

```text
同一组 GPU
attention 按 DP 解释
MoE 按 EP 解释
```

## 5. 单层执行图：DPA + EP + DeepEP

假设：

```text
tp_size = 4
dp_size = 4
ep_size = 4
moe_a2a_backend = deepep
top_k = 2
num_experts = 8
```

外部请求进入一个 HTTP URL：

```text
HTTP
  |
  v
DataParallelController
  |
  +--> DP rank 0 / GPU0: tokens from req A
  +--> DP rank 1 / GPU1: tokens from req B
  +--> DP rank 2 / GPU2: tokens from req C
  +--> DP rank 3 / GPU3: tokens from req D
```

一个 layer 内部：

```text
                 DPA + EP + DeepEP, one MoE layer

rank0 / dp0 / ep0      rank1 / dp1 / ep1      rank2 / dp2 / ep2      rank3 / dp3 / ep3
tokens A               tokens B               tokens C               tokens D
   |                      |                      |                      |
   | attention             | attention             | attention             | attention
   | attn_tp_size=1        | attn_tp_size=1        | attn_tp_size=1        | attn_tp_size=1
   v                      v                      v                      v
hidden A               hidden B               hidden C               hidden D
   |                      |                      |                      |
   | router/topk           | router/topk           | router/topk           | router/topk
   v                      v                      v                      v
A -> e1,e6             B -> e2,e7             C -> e4,e5             D -> e0,e3
   |                      |                      |                      |
   +---------- DeepEP dispatch across EP group [0,1,2,3] --------------+
                          |
                          v
rank0 owns e0,e1      rank1 owns e2,e3      rank2 owns e4,e5      rank3 owns e6,e7
compute:
  D/e0, A/e1          B/e2, D/e3            C/e4, C/e5            A/e6, B/e7
                          |
   +---------- DeepEP combine back to original DP token layout --------+
   |                      |                      |                      |
   v                      v                      v                      v
MoE out A              MoE out B              MoE out C              MoE out D
   |                      |                      |                      |
next layer local       next layer local       next layer local       next layer local
```

这张图里有两个“不同”：

```text
attention 阶段不同:
  rank0/1/2/3 本来就处理不同请求 tokens

expert compute 阶段不同:
  DeepEP dispatch 后，每个 EP rank 收到的是按 expert 聚合后的 token-expert 任务
```

这和 DeepEP-only 不一样。DeepEP-only 的 attention 阶段不是 `tokens A/B/C/D` 分别在不同 DP rank 上；它只有 `dp_size=1`。

## 6. 通信点在哪里

还是以：

```text
tp_size = dp_size = ep_size = 4
attn_tp_size = 1
moe_a2a_backend = deepep
```

为例。

### 6.1 请求分发

```text
HTTP server / tokenizer
  -> DataParallelController
  -> 某个 DP rank scheduler
```

这是 server 内部的请求路由，不是 GPU collective。

和 RL 最相关的是：

```text
不指定 routed_dp_rank:
  controller 按 load balance 策略选择 DP rank

指定 routed_dp_rank:
  请求直接发给指定 DP rank
```

### 6.2 attention 阶段

在 `attn_tp_size=1` 时：

```text
QKV / attention / o_proj 都在本 rank 本地完成
没有 attention TP all-reduce
```

如果不是 `tp_size=dp_size`，比如：

```text
tp_size = 16
dp_size = 8
```

则：

```text
attn_tp_size = 16 // 8 = 2
```

每个 DP rank 内部还有 attention TP2，这时 attention 里仍然会有 TP 相关通信。

### 6.3 MoE 阶段

DeepEP 后端：

```text
router/topk
  -> DeepEP dispatch
  -> local expert compute
  -> DeepEP combine
```

通信发生在：

```text
dispatch:
  token-expert 任务发给 expert 所在 EP rank

combine:
  expert output 回到原 token / DP rank 布局
```

这里不再是 03 文档 `none` 后端那种“所有 EP rank 都有同一批 tokens，然后 output all-reduce”。

### 6.4 layer 输出

DeepEP combine 后，token 回到原来的 DP rank：

```text
rank0 继续拿 tokens A
rank1 继续拿 tokens B
rank2 继续拿 tokens C
rank3 继续拿 tokens D
```

所以下一层 attention 仍然可以保持 DPA 语义。

## 7. DPA + EP + none 的通信图

`none` 后端也可以和 DPA+EP 组合，但它不是 token-expert A2A。

假设：

```text
tp_size = 4
dp_size = 4
ep_size = 4
moe_a2a_backend = none
```

attention 后：

```text
rank0: hidden A
rank1: hidden B
rank2: hidden C
rank3: hidden D
```

因为 sparse MoE 的 `mlp_mode` 会走 `FULL`，进入 MoE 前会把 DP tokens gather 成全局 token buffer：

```text
DP gather:

rank0: [A, B, C, D]
rank1: [A, B, C, D]
rank2: [A, B, C, D]
rank3: [A, B, C, D]
```

然后普通 EP 计算：

```text
rank0 owns e0,e1:
  对 [A,B,C,D] 中命中 e0/e1 的 token 算 contribution

rank1 owns e2,e3:
  对 [A,B,C,D] 中命中 e2/e3 的 token 算 contribution

rank2 owns e4,e5:
  对 [A,B,C,D] 中命中 e4/e5 的 token 算 contribution

rank3 owns e6,e7:
  对 [A,B,C,D] 中命中 e6/e7 的 token 算 contribution
```

MoE 输出再回到各自 DP rank：

```text
scatter / reduce-scatter:

rank0: MoE out A
rank1: MoE out B
rank2: MoE out C
rank3: MoE out D
```

可以记成：

```text
DPA + EP + none:
  attention 是 DP
  MoE 前 gather
  expert 本地算 contribution
  MoE 后 scatter / reduce-scatter

DPA + EP + deepep:
  attention 是 DP
  MoE 内 DeepEP dispatch 到 expert rank
  expert 本地算收到的 token-expert
  DeepEP combine 回原 DP rank
```

所以前者不是“每个 EP rank 从一开始就只处理自己不同 tokens”。它为了普通 EP 语义，需要在 MoE 边界做 token buffer 的重排和通信。

## 8. HTTP 请求到底发给谁

这是 RL 场景最容易混的点。

即使：

```text
dp_size = 8
tp_size = 8
ep_size = 8
```

对外业务 URL 仍然通常只有一个：

```text
http://$SERVER_IP:30000
```

你不是把请求分别发给 8 个 GPU 端口。SGLang 内部会启动 `DataParallelController`，它再把请求派发给 DP workers。

默认路由策略由 load balance 参数控制。源码里常见策略包括：

```text
round_robin
follow_bootstrap_room
total_requests
total_tokens
auto
```

对 RL 来说，最重要的是你可以显式指定 DP rank：

### 8.1 OpenAI API header

```bash
curl http://$SERVER_IP:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Data-Parallel-Rank: 3' \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 16
  }'
```

### 8.2 request body

```json
{
  "model": "your-model",
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 16,
  "routed_dp_rank": 3
}
```

如果同时存在 header 和 body，OpenAI serving 入口会优先用 header 解析出来的 DP rank。

如果 rank 越界：

```text
routed_dp_rank < 0
routed_dp_rank >= dp_size
```

会被校验拦住。

## 9. 为什么 DPA+EP 适合 RL rollout

RL rollout 关心的是：

```text
1. 大量独立请求
2. 每个请求生成长度不同
3. 希望 GPU 之间能自然分摊请求
4. MoE expert 又必须跨卡切分才能放得下 / 跑得快
```

EP-only 的问题是：

```text
dp_size = 1
所有 rank 合起来服务同一个 batch
不同 GPU 不是独立请求 worker
```

DPA+EP 的好处是：

```text
attention:
  请求可以按 DP rank 分流

MoE:
  experts 仍然按 EP rank 切

HTTP:
  仍然一个 server URL，RL 侧不用维护每张 GPU 的独立 endpoint

路由:
  可以交给 SGLang load balance
  也可以用 routed_dp_rank 显式控制
```

## 10. 常见误区

### 10.1 `dp_size=N` 不是自动等于 DPA

必须写：

```bash
--enable-dp-attention
```

否则 `dp_size` 走的是另一套 DP server 组织方式，不是本文讲的 attention DP carve-out。

本文讨论的是：

```text
enable_dp_attention = true
```

### 10.2 `tp_size=N` 仍然要写

DPA+EP 里 `tp_size` 仍然是内部总并行组大小。典型写法是：

```bash
--tp-size N \
--dp-size N \
--enable-dp-attention \
--ep-size N
```

不能只写：

```bash
--dp-size N \
--ep-size N
```

因为 `tp_size` 默认是 1，很多分组公式都会不符合预期。

还要注意，打开 `--enable-dp-attention` 后，`dp_size` 不是再额外乘出一份 GPU 数。最常见的：

```text
tp_size = 8
dp_size = 8
```

表示这 8 个 TP ranks 被重新解释成 8 个 attention DP ranks，而不是需要 `8 * 8 = 64` 张卡。

### 10.3 `ep_size` 和 `dp_size` 不一定永远相等，但先学相等

源码允许更复杂组合，例如：

```text
tp_size = 16
dp_size = 8
ep_size = 8
```

这时：

```text
attn_tp_size = 2
attn_dp_size = 8
moe_ep_size = 8
moe_tp_size = 2
```

这已经变成：

```text
attention:
  每个 DP rank 内还有 TP2

MoE:
  expert parallel 之外还有 MoE TP2
```

学习顺序建议先不要跳到这里。先把：

```text
tp_size = dp_size = ep_size
attn_tp_size = 1
moe_tp_size = 1
```

学清楚。

这里还要加一个 backend 限制：

```text
DeepEP 会强制 ep_size = tp_size
```

所以 `tp_size=16, dp_size=8, ep_size=8` 这种例子不要放在 DeepEP 语境下理解。它更适合用来理解 `none` 后端或其它允许 hybrid EP + MoE TP 的后端。

### 10.4 DPA+EP 不等于多个独立 server

DPA+EP 是一个 SGLang server 内部的多个 DP attention workers。对外不是 8 个 base URL。

你可以把它理解成：

```text
一个 HTTP 入口
内部多个 DP worker
MoE expert 跨这些 worker/rank 做 EP
```

## 11. 源码阅读顺序

建议按这个顺序看：

```text
python/sglang/srt/server_args.py
  看 enable_dp_attention / dp_size / ep_size 的参数处理
  看 enable_dp_attention 时 tp_size % dp_size 的约束

python/sglang/srt/layers/dp_attention.py
  看 compute_dp_attention_world_info()
  看 attn_dp_size / attn_tp_size / attn_dp_rank 的计算

python/sglang/srt/distributed/parallel_state.py
  看 initialize_model_parallel()
  看 ATTN_TP / MOE_EP / MOE_TP group 如何建

python/sglang/srt/managers/data_parallel_controller.py
  看 DataParallelController 如何启动 DP attention schedulers
  看 routed_dp_rank 如何直接路由到指定 worker

python/sglang/srt/entrypoints/openai/serving_base.py
  看 X-Data-Parallel-Rank header 如何解析

python/sglang/srt/managers/tokenizer_manager.py
  看 routed_dp_rank 的范围校验

python/sglang/srt/layers/communicator.py
  看 LayerScatterModes
  看 none 后端为什么 sparse MoE mlp_mode 是 FULL
  看非 none 后端为什么 mlp_mode 是 SCATTERED

python/sglang/srt/models/qwen3_moe.py
  看 Qwen3MoeAttention 用 attn_tp_size 切 attention
  看 Qwen3MoeSparseMoeBlock 里 normal / deepep 两条 forward

python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  看 DPA+EP+DeepEP 的 dispatch / combine
```

## 12. 最终记忆

可以把本文压缩成几句话：

```text
DPA+EP = 同一批 GPU 在不同模块里按不同维度解释。

attention:
  按 DP 解释
  不同 DP rank 可以处理不同请求

MoE:
  按 EP 解释
  experts 切到不同 rank

DeepEP backend:
  token-expert dispatch 到 expert rank
  combine 回原 DP token 布局

none backend:
  MoE 边界用 gather / scatter / reduce-scatter 类通信维持普通 EP 语义
```

对 RL rollout 最重要的一句：

```text
DPA+EP 仍然是一个业务 base_url，但内部可以把请求分到多个 DP rank，并且 MoE experts 仍然跨 rank 并行。
```


## 13. DeepEP 下 `tp/dp/ep/enable-dp-attention` 组合表

这一节只讨论：

```text
--moe-a2a-backend deepep
pp_size = 1
```

先给 DeepEP 下的硬规则。

### 13.1 DeepEP 的通用规则

规则 1：DeepEP 会强制：

```text
effective_ep_size = tp_size
```

所以你写：

```bash
--tp-size 8 --ep-size 4 --moe-a2a-backend deepep
```

最终也会被调整成：

```text
effective_ep_size = 8
```

因此 DeepEP 语境下，`--ep-size` 更像是“显式表达意图”。建议写成和 `tp_size` 一样，不要写不同值。

规则 2：如果：

```text
dp_size = 1
```

那么：

```text
enable_dp_attention = false
```

即使你手动写了 `--enable-dp-attention`，`dp_size=1` 也没有 DPA 意义。

规则 3：如果开启：

```bash
--enable-dp-attention
```

则必须满足：

```text
tp_size % dp_size == 0
```

并且：

```text
attn_dp_size = dp_size
attn_tp_size = tp_size // dp_size
```

规则 4：不开 DPA 时，`dp_size` 是“模型副本数”；开 DPA 时，`dp_size` 是“同一个 TP group 里的 attention DP 数”。

这是最容易混的地方：

```text
不开 enable-dp-attention:
  总 GPU 数 = tp_size * dp_size
  每个 DP 副本内部各自有一个 DeepEP group

开启 enable-dp-attention:
  总 GPU 数 = tp_size
  dp_size 是把这 tp_size 个 ranks 重新解释成 attention DP
```

### 13.2 组合总表

| `tp_size` | `dp_size` | `enable_dp_attention` | 用户写的 `ep_size` | DeepEP 实际 `ep_size` | 实际 GPU 数 | attention 形态 | MoE 形态 | 含义 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N` | `1` | `false` | 建议 `N` | `N` | `N` | `attn_tp_size=N`, `attn_dp_size=1` | `moe_ep_size=N`, `moe_tp_size=1` | DeepEP-only。所有 rank 组成一个模型实例，MoE 用 DeepEP dispatch/combine |
| `N` | `1` | `true` | 建议 `N` | `N` | `N` | 实际会退化成不开 DPA | `moe_ep_size=N` | 没必要这么写。`dp_size=1` 时 DPA 没意义 |
| `N` | `M>1` | `false` | 建议 `N` | `N` | `N * M` | 每个副本内部 `attn_tp_size=N` | 每个副本内部 `moe_ep_size=N` | 普通 DP 多副本。不是 DPA；是起 `M` 个 DeepEP 模型副本 |
| `N` | `M>1` 且 `N % M == 0` | `true` | 建议 `N` | `N` | `N` | `attn_dp_size=M`, `attn_tp_size=N/M` | `moe_ep_size=N`, `moe_tp_size=1` | DPA + DeepEP。同一组 `N` 卡里，attention 分成 `M` 个 DP 流，MoE 用 EP `N` |
| `N` | `M>1` 且 `N % M != 0` | `true` | 任意 | `N` | 启动报错 | 不合法 | 不合法 | DPA 要求 `tp_size % dp_size == 0` |
| `N` | 任意 | 任意 | `K != N` | `N` | 取决于是否 DPA | 取决于是否 DPA | `moe_ep_size=N` | DeepEP 会覆盖 `ep_size`，不要依赖 `K` |

这个表里最重要的是两行：

```text
不开 DPA:
  dp_size 会乘出多个模型副本

开 DPA:
  dp_size 不再乘 GPU，而是在 tp_size 内部切 attention DP
```

### 13.3 典型例子

#### 13.3.1 8 卡，DeepEP-only

```bash
--tp-size 8 \
--dp-size 1 \
--ep-size 8 \
--moe-a2a-backend deepep
```

实际含义：

```text
GPU 数:
  8

attention:
  attn_tp_size = 8
  attn_dp_size = 1

MoE:
  moe_ep_size = 8
  moe_tp_size = 1

请求流:
  不是多个 attention DP rank
```

这是第 4 篇主要讲的 DeepEP-only。

#### 13.3.2 8 卡，DPA + DeepEP8

```bash
--tp-size 8 \
--dp-size 8 \
--enable-dp-attention \
--ep-size 8 \
--moe-a2a-backend deepep
```

实际含义：

```text
GPU 数:
  8

attention:
  attn_dp_size = 8
  attn_tp_size = 1

MoE:
  moe_ep_size = 8
  moe_tp_size = 1

请求流:
  8 个 attention DP rank
```

这是最适合先学的 DPA+EP 形态。

#### 13.3.3 16 卡，两个 DeepEP8 副本

```bash
--tp-size 8 \
--dp-size 2 \
--ep-size 8 \
--moe-a2a-backend deepep
```

注意这里没有：

```bash
--enable-dp-attention
```

实际含义：

```text
GPU 数:
  8 * 2 = 16

模型副本:
  2 个

每个副本:
  attention TP8
  MoE DeepEP8

请求流:
  controller 可以把请求分给两个 DP 副本

expert:
  两个副本之间不共享 experts
  每个副本各自有一套 expert 权重
```

这适合“卡很多，想多起几个完整模型副本”的场景，不是本文重点的 DPA+EP。


#### 13.3.4 16 卡，DPA8 + DeepEP16

```bash
--tp-size 16 \
--dp-size 8 \
--enable-dp-attention \
--ep-size 16 \
--moe-a2a-backend deepep
```

实际含义：

```text
GPU 数:
  16

attention:
  attn_dp_size = 8
  attn_tp_size = 2

MoE:
  moe_ep_size = 16
  moe_tp_size = 1

请求流:
  8 个 attention DP rank
  每个 DP rank 内 attention 还有 TP2
```

这个组合是合法的，因为：

```text
16 % 8 == 0
```

但它比 `tp=dp=ep` 难理解，因为 attention 里还有 TP2。建议学完 `tp8/dp8/ep8` 后再看。

#### 13.3.5 16 卡，DPA16 + DeepEP16

```bash
--tp-size 16 \
--dp-size 16 \
--enable-dp-attention \
--ep-size 16 \
--moe-a2a-backend deepep
```

实际含义：

```text
GPU 数:
  16

attention:
  attn_dp_size = 16
  attn_tp_size = 1

MoE:
  moe_ep_size = 16
  moe_tp_size = 1

请求流:
  16 个 attention DP rank
```

这是 16 卡上最直观的 DPA+DeepEP。

#### 13.3.6 容易误解的写法：`tp8 dp8` 但不开 DPA

```bash
--tp-size 8 \
--dp-size 8 \
--ep-size 8 \
--moe-a2a-backend deepep
```

因为没有：

```bash
--enable-dp-attention
```

实际含义不是 8 卡 DPA+EP，而是：

```text
GPU 数:
  8 * 8 = 64

模型副本:
  8 个 DeepEP8 副本
```

所以如果你的机器只有 8 张卡，这种写法不是你想要的。

#### 13.3.7 容易误解的写法：DeepEP 下 `ep_size != tp_size`

```bash
--tp-size 16 \
--dp-size 8 \
--enable-dp-attention \
--ep-size 8 \
--moe-a2a-backend deepep
```

这不是：

```text
MoE EP8
```

因为 DeepEP 会强制：

```text
effective_ep_size = tp_size = 16
```

所以实际更接近：

```text
attention:
  attn_dp_size = 8
  attn_tp_size = 2

MoE:
  moe_ep_size = 16
  moe_tp_size = 1
```

如果你真的想研究：

```text
tp_size = 16
ep_size = 8
moe_tp_size = 2
```

那就不是 DeepEP 的学习路径，应该回到 `none` 或其它支持 hybrid EP + MoE TP 的 backend。

#### 13.3.8 真实业务例子：64 卡 engine，DPA8 + DeepEP64

你给的业务参数是：

```bash
--rollout-num-gpus-per-engine 64
--sglang-mem-fraction-static 0.7
--sglang-ep-size 64

# dp attention
--sglang-enable-dp-attention
--sglang-dp-size 8
--sglang-moe-dense-tp-size 1
--sglang-enable-dp-lm-head

# enable deepep for sglang
--sglang-moe-a2a-backend deepep
--sglang-deepep-mode auto
```

这里我按常见上层 rollout 框架的含义理解：

```text
--rollout-num-gpus-per-engine 64
  对应一个 SGLang engine 使用 64 张 GPU
  等价理解成 SGLang server 内部 tp_size = 64

--sglang-ep-size 64
  对应 SGLang --ep-size 64

--sglang-dp-size 8
  对应 SGLang --dp-size 8

--sglang-enable-dp-attention
  对应 SGLang --enable-dp-attention
```

所以这个组合可以翻译成 SGLang 内部并行参数：

```text
tp_size = 64
dp_size = 8
enable_dp_attention = true
ep_size = 64
moe_a2a_backend = deepep
deepep_mode = auto
moe_dense_tp_size = 1
enable_dp_lm_head = true
```

先看核心并行公式：

```text
attn_dp_size = dp_size = 8
attn_tp_size = tp_size // dp_size = 64 // 8 = 8

DeepEP 强制:
  moe_ep_size = ep_size = tp_size = 64

默认 moe_dp_size = 1，所以:
  moe_tp_size = tp_size // moe_ep_size // moe_dp_size
              = 64 // 64 // 1
              = 1
```

最终含义：

```text
GPU 数:
  64

attention:
  8 个 attention DP rank
  每个 attention DP rank 内部还有 attention TP8

MoE:
  DeepEP64
  64 个 ranks 共同组成一个 expert parallel group
  moe_tp_size = 1

请求流:
  不是 64 个独立 DP 请求流
  是 8 个 attention DP 请求流
```

rank 可以这样理解：

```text
DP attention group 0:
  ranks [0..7]
  attention TP8

DP attention group 1:
  ranks [8..15]
  attention TP8

...

DP attention group 7:
  ranks [56..63]
  attention TP8

MoE EP group:
  ranks [0..63]
  DeepEP64
```

也就是说，请求进入后：

```text
DataParallelController
  -> 选择 8 个 DP rank 之一
  -> 该 DP rank 内部用 8 张卡做 attention TP
  -> 到 MoE 时，token-expert 任务通过 DeepEP 在 64 张卡的 expert group 里 dispatch/combine
```

单层图可以简化成：

```text
HTTP request
  |
  v
DataParallelController
  |
  +--> attn DP0: ranks 0..7     attention TP8
  +--> attn DP1: ranks 8..15    attention TP8
  +--> ...
  +--> attn DP7: ranks 56..63   attention TP8
             |
             v
        router / topk
             |
             v
DeepEP dispatch across ranks 0..63
             |
             v
local expert compute on owner ranks
             |
             v
DeepEP combine back to original attention DP group
```

这个配置和 `tp64/dp64/ep64` 的区别很大：

```text
tp64 dp64 enable-dp-attention:
  attn_dp_size = 64
  attn_tp_size = 1
  64 个 attention DP 请求流

你的配置 tp64 dp8 enable-dp-attention:
  attn_dp_size = 8
  attn_tp_size = 8
  8 个 attention DP 请求流
  每个请求流内部 attention 仍然 TP8
```

为什么业务上可能这样配：

```text
1. 64 卡全部参与一个超大 MoE expert pool，MoE 用 DeepEP64
2. attention 不做 TP64，而是拆成 8 个 DP 流，每个流 TP8
3. 比 DeepEP-only 更适合 RL 多请求 rollout
4. 比 DPA64 更保守，因为每个 attention DP 流仍有 TP8，可以承载更大的 dense/attention 权重
```

这几个附加参数也要分开看：

```text
--sglang-mem-fraction-static 0.7
  内存比例参数，不改变并行拓扑。

--sglang-moe-dense-tp-size 1
  对应 --moe-dense-tp-size 1。
  含义是 MoE 模型里的 dense MLP 层 TP size 设为 1。
  在 DPA 场景下，它通常用于让 dense MLP 层更偏 DP 化，避免大 TP size 下 dense MLP 权重维度太小或 kernel 不合适。

--sglang-enable-dp-lm-head
  对应 --enable-dp-lm-head。
  它必须配合 --enable-dp-attention 使用。
  作用是让 LM head / vocab parallel 使用 attention TP group，减少 DP attention 下跨 DP group 的 logits all-gather 压力。

--sglang-deepep-mode auto
  prefill / extend 倾向 normal
  decode 倾向 low_latency
```

对 RL 请求路由来说，最重要的是：

```text
routed_dp_rank 的范围是 [0, 8)
```

也就是：

```text
routed_dp_rank = 0..7
```

不是：

```text
0..63
```

因为这里的请求级 DP 是：

```text
attn_dp_size = dp_size = 8
```

一句话总结这个业务配置：

```text
这是一个 64 卡 SGLang engine：
  attention 被切成 DPA8，每个 DP 流内部 TP8；
  MoE experts 用 DeepEP64；
  对 RL 来说有 8 个可路由的 DP 请求流，而不是 64 个。
```

### 13.4 选择建议

如果目标是 RL rollout，且你假设 DeepEP 一定开，优先考虑：

```bash
--tp-size N \
--dp-size N \
--enable-dp-attention \
--ep-size N \
--moe-a2a-backend deepep
```

它的好处是：

```text
1. 总 GPU 数就是 N，不会被 dp_size 再乘一次
2. attention 不切 head，attn_tp_size = 1
3. 每个 DP rank 可以承接不同请求
4. MoE experts 用 DeepEP 在 N 张卡上切分
```

如果你有更多卡，想多个完整模型副本提高吞吐，再考虑：

```bash
--tp-size N \
--dp-size M \
--ep-size N \
--moe-a2a-backend deepep
```

并且不加 `--enable-dp-attention`。这时需要：

```text
N * M 张 GPU
```

一句话总结：

```text
想让 dp_size 不额外乘 GPU:
  加 --enable-dp-attention

想让 dp_size 表示多个完整模型副本:
  不加 --enable-dp-attention

DeepEP 下:
  ep_size 永远按 tp_size 理解
```

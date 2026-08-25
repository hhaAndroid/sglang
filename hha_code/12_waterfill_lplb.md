# 12. Waterfill 和 LPLB：DeepEP MoE 的运行时负载均衡

本文基于 LMSYS 文章：

```text
Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB
https://www.lmsys.org/blog/2026-06-26-waterfill-lplb/
```

文章讲的是 SGLang 在 DeepEP MoE 推理里新增的两类 dispatch-time load balancing：

```text
Waterfill:
  针对 shared expert
  把 shared expert 当成可调度的额外 expert slot
  尽量 dispatch 到当前更空的 EP rank

LPLB:
  针对 EPLB 产生的 redundant expert replicas
  用每层每 batch 的线性规划，决定 replicated logical expert 的 token 应该怎么分到不同 physical copies
```

一句话总结：

> EPLB 偏静态 placement，Waterfill / LPLB 偏运行时 dispatch。它们不改变 logical expert 选择，只改变哪个 physical rank / physical replica 执行这份计算。

## 0. 先用 EP=4 建立直觉

假设：

```text
ep_size = 4

rank0:
  expert group 0

rank1:
  expert group 1

rank2:
  expert group 2

rank3:
  expert group 3
```

输入可以理解成 4 路数据，每个 rank 都有一批本地 tokens。进入 MoE 层后，router/topk 先为每个 token 选择 routed experts，然后 DeepEP 做：

```text
dispatch:
  按 token 选中的 expert，把 token 发到对应 EP rank

per-rank MoE compute:
  每个 rank 只计算自己负责的 experts 收到的 tokens

combine:
  把 expert 输出收回来，恢复成原来的 token shape
```

这一步会产生不平衡。比如 routed experts 造成的负载是：

```text
rank0 routed load = 1000
rank1 routed load = 800
rank2 routed load = 400
rank3 routed load = 300
```

整个 EP group 的 MoE latency 往往被 rank0 这种最忙 rank 拖住。

旧的 shared expert 可以先理解成一条本地 dense branch：

```text
rank0 本地 tokens -> rank0 算 shared expert
rank1 本地 tokens -> rank1 算 shared expert
rank2 本地 tokens -> rank2 算 shared expert
rank3 本地 tokens -> rank3 算 shared expert
```

也就是说，shared expert 不参与 routed expert 的 DeepEP dispatch。这样 rank0 已经因为 routed experts 很忙了，还要继续算自己的 shared expert；rank2 / rank3 比较空，却没办法帮 rank0 分担 shared expert 工作。

Waterfill 的直观目标就是改变 shared expert 这一路的 token-to-rank 分配：

```text
Waterfill 前:
  shared expert tokens 固定由本地 rank 计算

Waterfill 后:
  shared expert tokens 可以被路由到更空的 EP rank 计算
```

例如本来 shared expert 也会让 rank0 多算很多 token，rank1 算相对少一些；Waterfill 会强行调整 shared expert 的 token 路由，让总负载更接近：

```text
rank0 ~= 900
rank1 ~= 900
rank2 ~= 900
rank3 ~= 900
```

这个例子里的数字只是帮助理解，不是实际算法的精确输出。真实实现还要考虑通信约束：shared expert token 不一定能随便发到任意 rank，通常会优先选择这个 token 的 routed experts 已经访问过的 ranks，避免额外扩大 all-to-all 通信。

所以 Waterfill 可以这样记：

> routed experts 决定基础负载；Waterfill 用 shared expert 这部分每个 token 都必须执行的 dense 工作去补低谷，让每个 EP rank 的总 MoE 工作量尽量均衡。

要做到这一点，前提是 shared expert 必须融合进 DeepEP 的 MoE dispatch 里。如果 shared expert 仍然是独立本地 branch，DeepEP 的 dispatch/combine 根本看不到这部分工作，也就没法调度它。

因此现在的做法相当于：

```text
router/topk:
  仍然只决定 routed experts

系统额外追加:
  一个 shared expert slot

shared expert slot 的特殊性:
  每个 token 都必须经过它
  它不是由 router 选择性激活的
  Waterfill 只决定它由哪个 physical EP rank 执行

DeepEP:
  routed expert slots + shared expert slot
  一起 dispatch / compute / combine
```

这也是 Waterfill 必须依赖 shared expert fusion 的原因。

## 1. 先给结论

DeepEP MoE 的性能瓶颈经常不是单个 expert 算不快，而是：

```text
router/topk 选 expert 不均匀
某些 expert 或某些 rank 收到更多 token
EP group 里其他 rank 算完后等待最忙的 rank
整层 MoE latency 被长尾 rank 决定
```

静态 EPLB 可以通过移动 expert placement 和复制热点 expert 改善长期分布，但它解决不了每个 batch 的残余不均衡：

```text
训练/采样统计分布 != 当前线上 batch 分布
当前 batch 可能集中命中特定 expert
prefill / decode / idle rank 在 DP attention 下也可能看到不同 token 分布
```

Waterfill 和 LPLB 正是补这个缺口：

| 方法 | 处理对象 | 决策内容 | 依赖 | 成本 |
| --- | --- | --- | --- | --- |
| Waterfill | shared expert | 每个 token 的 shared expert slot 发到哪个 rank | shared expert fusion + DeepEP | 很低 |
| LPLB | redundant routed experts | replicated logical expert 的 token 分到哪些 physical copies | EPLB redundant experts | all-reduce + LP solve |

## 2. 背景：DeepEP MoE 为什么还会不均衡

MoE 模型里每个 token 通常会经过：

```text
router/topk:
  token -> 选择若干 routed experts

DeepEP dispatch:
  token 按 expert 所在 EP rank 发过去

expert compute:
  每个 rank 只算自己收到的 token/expert

combine:
  把 expert 输出合并回来
```

在 DeepSeek-V3/R1 这类模型里，还会有 shared expert：

```text
routed expert:
  稀疏
  不同 token 选不同 experts
  负载取决于 router 分布

shared expert:
  稠密
  每个 token 都要过 shared expert
  每个 batch 都有这份稳定工作量

redundant expert:
  EPLB 复制出来的热点 logical expert physical copies
  给运行时 dispatch 提供多个合法目的地
```

不均衡的本质是：

```text
logical expert 选择是模型语义的一部分
physical expert / physical rank 是部署实现的一部分
```

只要不改变 logical expert，系统就有机会在 physical 层面做负载均衡。

## 3. Waterfill：把 shared expert 也纳入 DeepEP dispatch

### 2.1 它解决什么问题

如果 shared expert 总是在本 rank 本地计算，那么每个 rank 都固定承担 shared expert 工作：

```text
rank A routed expert 已经很忙
rank A 仍然要算本地 shared expert

rank B routed expert 比较空
rank B 不能帮 rank A 分担 shared expert
```

这样 shared expert 没有缓解 routed expert 造成的长尾，反而会叠加到已经忙的 rank 上。

Waterfill 的做法是：

```text
先看 routed experts 已经给每个 EP rank 带来了多少负载
再把 shared expert 的工作量分配给相对更空的 rank
```

直观上就是把 shared expert 这桶水倒进 rank load 的低洼处，让各 rank 的总工作量更平。

### 2.2 算法流程

文章给的 Waterfill 高层流程是：

```text
1. 统计 routed expert 已经落到每个 EP rank 的 load

2. 把这个 load 当成每个 rank 当前的负载分数
   dynamic 模式下，会先做一次 EP group collective
   这样每个 rank 可以看到全局 routed-load vector

3. 对每个参与 token 增加一个 shared expert slot
   设:
     L_r = rank r 当前负载
     N   = 需要放置的 shared expert slots 数量
     R   = EP group size

   目标 waterline:
     H = ceil((sum_r L_r + N) / R)

4. 每个 rank 的 slack:
     S_r = max(H - L_r, 0)

5. 对每个 token，按 slack 比例选择 shared expert 的目标 rank
   同时保留一个小的 local-rank preference
   如果候选 rank 都没有 slack，则 fallback 到更轻的候选 rank
```

这里的核心不是精确求全局最优，而是以很低成本把 shared expert 负载从高负载 rank 挪到低负载 rank。

### 2.3 为什么不能随便发到任意 rank

理论上，如果每个 token 的 shared expert 都可以发到任意 EP rank，平衡空间最大。

但在 GPU MoE serving 里，通信常常比多算一点更贵。任意 rank 发送会扩大 all-to-all 目的地，可能增加通信成本。

因此文章强调了一个通信约束：

```text
保守模式:
  shared expert 优先放到这个 token 的 routed experts 已经访问过的 ranks
  source/local rank 作为 fallback

all-rank 模式:
  可以给 Waterfill 更多自由度
  但可能增加每 token 的 dispatch destination
```

所以 Waterfill 的设计取舍是：

```text
既要填平 rank 负载低洼
又不能为了平衡引入过高通信开销
```

## 4. shared expert fusion 是 Waterfill 的机制基础

Waterfill 能落地，需要先让 shared expert 进入 DeepEP MoE layout。

原先 shared expert 如果走独立路径，Waterfill 一旦选择非本地 rank，就会变成：

```text
DeepEP routed experts dispatch
再从 dispatched layout 里抽 shared expert tokens
单独 launch shared expert compute
再做 layout conversion / combine
```

这样开销会很高。

shared expert fusion 的思路是：

```text
把 shared expert 表示成 DeepEP MoE layout 里的额外 expert slot
TopK 输出从原 routed top-k 追加一列 shared expert
每个 EP rank 在 physical expert id layout 里预留一个 shared expert slot
routed experts 和 shared expert 共享同一套 DeepEP dispatch / grouped-GEMM / combine 流程
```

所以文章里把 Waterfill 拆成两步 PR：

```text
#20089:
  把 shared expert 融入 DeepEP MoE path
  先用固定 local assignment

#19290:
  在 fusion 基础上加入 Waterfill
  把 fixed assignment 替换成 load-aware assignment
```

理解重点：

> shared expert fusion 不是最终负载均衡算法，它是让 shared expert 变成 DeepEP 可调度对象的前置机制。

## 5. LPLB：用线性规划分配 redundant expert traffic

### 4.1 它解决什么问题

EPLB 会把热点 logical expert 复制成多个 physical copies：

```text
logical expert 7
  -> physical copy on rank 1
  -> physical copy on rank 5
  -> physical copy on rank 9
```

默认的 dynamic dispatch 可以把 token 均匀随机分到这些 copies。

问题是，均匀分配只在一个前提下合理：

```text
EPLB placement 使用的离线统计分布
约等于
当前 live batch 的真实分布
```

现实里这个前提经常不成立：

```text
当前 batch 比 calibration 数据更偏
线上数据分布漂移
rebalance 周期较长
某一层当前 batch 正好集中命中特定专家
```

于是即使 logical expert 有多个 copies，平均切分也可能让某些 rank 继续过载。

LPLB 的目标是：

```text
每层、每 batch 看当前实际 per-expert token counts
决定每个 replicated logical expert 的 token 应该按什么比例分到各 physical copies
最小化最大 rank load
```

它不搬权重，不改 router/topk，只是在合法 replicas 之间决定流量比例。

### 4.2 LP formulation

LPLB 把问题建成一个小线性规划。

目标：

```text
minimize M

M = 所有 rank 中的最大负载
```

约束大致是：

```text
rank-load constraints:
  每个 rank:
    来自 redundant copies 的可调度负载
  + 来自 single-copy experts 的固定负载
  + slack
  = M

redundant-expert conservation:
  对每个 replicated logical expert:
    x_1 + x_2 + ... + x_n = L

  其中:
    x_i = 分给第 i 个 physical copy 的负载
    L   = 当前 batch 中这个 logical expert 的总 observed load
```

变量包括：

```text
每个 replicated expert copy 的分配负载
每个 rank 到峰值 M 的 slack
峰值 M
```

single-copy experts 不作为变量，只作为固定负载加入约束。

这能把 LP 规模控制在：

```text
redundant experts 数量 + ranks 数量
```

而不是完整 expert 数量。

### 4.3 offline / online 拆分

文章强调 constraint matrix 被拆成两部分：

```text
offline:
  physical copy -> logical expert 映射
  rank 拥有哪些 replicated copies
  slack / -M 结构列
  这些只依赖 expert placement
  启动时或 EPLB rebalance 后预计算

online:
  当前 batch observed redundant-expert loads
  当前 batch per-rank single-copy loads
  这些作为 right-hand side 每 batch 更新
```

这样每 batch 不需要重新搭完整问题，只需要更新 RHS 并求解。

文章还提到 Big-M auxiliary column：

```text
用于保证 solve feasible
在 objective 里给重罚
求解器会尽量把它压到 0
```

### 4.4 全局统计和求解

DP attention 场景下，同一个 step 里不同 EP rank 可能处于不同 forward mode：

```text
rank 0 prefill
rank 1 decode
rank 2 idle
...
```

所以单个 rank 不能代表全局 token distribution。

LPLB 的做法是：

```text
1. 每个 rank 本地统计 tokens per logical expert
2. EP group 做一次 all-reduce
   idle rank 贡献 0
3. 每个 rank 得到相同的 global per-expert distribution
4. 每个 rank 独立求同一个 LP
5. 不需要额外 broadcast LP result
```

文章里说 LP 在 GPU 上用 fused IPM kernel 求解，基于 `cuSOLVERDx` / `cuBLASDx`，并按 layer matrix shape 预编译，避免第一个真实请求承受 JIT 成本。

每 batch 关键路径被压成三类 CUDA kernel：

```text
build right-hand side
solve IPM
extract per-copy split / log2phy_prob
```

### 4.5 LP 输出如何进入 token dispatch

LP 返回的是每个 replicated logical expert 在各 physical copies 上的负载分配。

系统会把它归一化成：

```text
log2phy_prob:
  logical expert -> physical copies probability
```

dispatch 时：

```text
token routed 到 replicated logical expert:
  按 log2phy_prob 采样一个 physical copy

token routed 到 single-copy logical expert:
  仍然映射到唯一 physical location
```

所以 LPLB 可以看成 dynamic policy 的替代：

```text
dynamic:
  replicated copies 均匀随机

lp / LPLB:
  replicated copies 按 LP 求出的 load-optimal distribution 随机
```

## 6. Waterfill 和 LPLB 的关系

它们不是互斥关系，而是作用在两个不同负载来源上：

```text
Waterfill:
  shared expert 是 dense 的
  每个 token 都有 shared expert 工作
  通过移动 shared expert slot 平衡 rank load

LPLB:
  routed experts 是 sparse 的
  只有被 EPLB replicated 的 logical experts 才有多个 physical choices
  通过 LP 改变 redundant copies 的流量比例
```

可以这样理解：

```text
EPLB:
  先给 hot experts 摆好位置和 replicas

Waterfill:
  利用 shared expert fusion，把 shared expert 的 dense 工作放到更空 rank

LPLB:
  利用 EPLB redundant replicas，把 routed expert 的 sparse 工作放到更空 rank
```

准确性上，两者都不改变模型语义：

```text
Waterfill:
  routed top-k 不变
  shared expert 仍然是同一个 shared expert
  只变执行它的 physical EP rank

LPLB:
  logical top-k 不变
  selected logical expert 不变
  只在这个 logical expert 的 identical physical replicas 里选择一个
```

## 7. 什么时候 LPLB 最有用

文章给出的判断很实用：

```text
极大规模、非常多样的 serving:
  batch 内部分布可能已经比较平均
  残余不均衡少
  LPLB 收益可能有限

极窄、几乎不变的 serving:
  静态 EPLB calibration 已经很贴近 live traffic
  even split 也接近最优
  LPLB 收益可能有限

中等规模、主题相关但不完全固定的 serving:
  batch 会有结构性偏斜
  但偏斜又不是 EPLB 离线 placement 总能预测
  LPLB 最容易带来收益
```

换句话说，LPLB 是在“有残余不均衡，并且 redundant copies 真的给了选择空间”时最有价值。

## 8. 评测数据怎么读

### 7.1 DeepSeek-V3/R1-style workload

文章配置：

```text
Model:
  DeepSeek-V3 FP8

Hardware:
  2 个 Hopper GPU nodes
  16 GPUs total

Parallelism/backend:
  TP16
  DP16
  EP16
  DP attention
  DeepEP normal mode

Datasets:
  MMLU
  GPQA
  GSM8K

Benchmark shape:
  batch_size=1000
  concurrency=256
  request_rate=inf
  max_tokens=1
```

Waterfill 的结果比较稳定：

```text
MMLU:
  +1.48% ~ +3.40%

GPQA:
  +2.14% ~ +4.66%

GSM8K:
  +3.19% ~ +4.45%
```

LPLB 要分情况：

```text
red0:
  没有 redundant replicas
  LP 没有调度空间
  只剩 all-reduce / solve overhead
  MMLU / GPQA / GSM8K 都是负收益

red16 / red32:
  有 replicated experts 可选
  收益变正
  最高 GSM8K red32: +7.34%
```

所以这组数据最重要的结论不是“LPLB 永远提升”，而是：

> LPLB 必须和有实际 redundant experts 的 EPLB placement 配套使用。

### 7.2 DeepSeek V4 Flash

文章还测了 DeepSeek V4 Flash FP8：

```text
Hardware:
  2 个 Hopper GPU nodes

Dataset:
  MMLU-style 14,042 prompts

Benchmark shape:
  batch=512
  concurrency=128
  max_tokens=1
  2 warmup rounds
  4 measured rounds
```

Waterfill 在 DeepSeek V4 Flash 上也有正收益：

```text
No EPLB:
  45,951 -> 47,876 tok/s
  +4.19%

Static EPLB, red0:
  49,253 -> 51,677 tok/s
  +4.92%

Static EPLB, red16:
  50,006 -> 51,655 tok/s
  +3.30%

Static EPLB, red32:
  50,167 -> 51,813 tok/s
  +3.28%
```

这部分主要说明：

```text
Waterfill 不只适用于 DeepSeek-V3/R1 的普通 TopK 路径
DeepSeek V4 的 HashTopK 路径也可以支持
```

但文章也提醒，DeepSeek V4 的 batch/concurrency 配置不同，不能直接和 V3/R1 的吞吐数字横向比较。

## 9. 如何启用

### 8.1 Waterfill

文章给的代表性启动方式：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-V3 \
    --tp 16 \
    --dp-size 16 \
    --nnodes 2 \
    --node-rank ${NODE_RANK} \
    --dist-init-addr ${HEAD_NODE_IP}:${PORT} \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --enable-dp-attention \
    --enable-deepep-waterfill \
    --init-expert-location /path/to/expert_distribution.pt
```

关键参数：

```text
--moe-a2a-backend deepep:
  使用 DeepEP token dispatch / combine

--enable-deepep-waterfill:
  开启 shared expert fusion + Waterfill dispatch

--init-expert-location:
  可选，用已收集的 expert distribution / placement 信息初始化 expert location
```

当前仓库里 `ServerArgs` 对 `--enable-deepep-waterfill` 的描述是：

```text
启用 DeepEP Waterfill
把 shared expert 作为额外 routed expert dispatch 到较低负载 EP rank
自动设置 moe_a2a_backend=deepep
隐式启用 shared-expert fusion
支持 deepep-mode auto / normal / low_latency
```

源码入口：

```text
python/sglang/srt/server_args.py
  enable_deepep_waterfill
  _handle_a2a_moe()

python/sglang/srt/model_executor/model_runner.py
  _prepare_moe_topk()

python/sglang/srt/layers/moe/deepep_waterfill.py
  DeepEPWaterfillBalancer

python/sglang/srt/layers/moe/topk.py
python/sglang/srt/layers/moe/hash_topk.py
  _apply_deepep_waterfill()
```

### 8.2 LPLB

文章给的代表性启动方式：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-R1 \
    --tp 16 \
    --dp-size 16 \
    --ep-size 16 \
    --nnodes 2 \
    --node-rank ${NODE_RANK} \
    --dist-init-addr ${HEAD_NODE_IP}:${PORT} \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --enable-dp-attention \
    --ep-num-redundant-experts 16 \
    --ep-dispatch-algorithm lp \
    --init-expert-location /path/to/expert_stats.pt
```

关键参数：

```text
--ep-dispatch-algorithm lp:
  选择 LPLB 线性规划 dispatcher
  替代 static 或 uniform-random dynamic

--ep-num-redundant-experts:
  创建 redundant physical replicas
  没有 redundant experts，LPLB 没有可调度空间

--init-expert-location:
  加载 static EPLB placement
  其中包含 physical-to-logical map 和 redundant slots
  replica 数量必须和 ep-num-redundant-experts 一致
```

当前仓库里 `ep_dispatch_algorithm` 支持：

```text
static
dynamic
fake
lp
```

源码入口：

```text
python/sglang/srt/server_args.py
  ep_num_redundant_experts
  ep_dispatch_algorithm
  init_expert_location
  enable_eplb
  _handle_eplb_and_dispatch()

python/sglang/srt/model_executor/model_runner.py
  _init_lplb_solvers()

python/sglang/srt/eplb/lplb_solver.py
  LPLBSolver

python/sglang/srt/eplb/expert_location_dispatch.py
  _topk_ids_logical_to_physical_probability()

python/sglang/jit_kernel/lplb/
  cuda_solver.py
  torch_solver.py
  csrc/lplb/*.cuh

python/sglang/srt/layers/moe/topk.py
python/sglang/srt/layers/moe/hash_topk.py
  在 topk_ids 转 physical ids 前调用 LPLBSolver.solve()
```

## 10. 和前面 EPLB 文档的关系

可以把几篇文档串起来：

```text
03_moe_ep:
  解释 SGLang 里的 EP 不是额外 world_size 维度

04_moe_deepep_backend:
  解释 DeepEP 是 token dispatch + expert compute + combine 的 A2A 后端

09_moe_eplb_outline:
  解释 EPLB / redundant experts / logical-to-physical placement

12_waterfill_lplb:
  解释在 DeepEP + EPLB 基础上，如何进一步做运行时 dispatch load balance
```

更完整的生产链路可以理解成：

```text
先有 EP:
  expert 切到不同 rank

再有 DeepEP:
  token 按 expert 位置 dispatch，而不是每 rank 都复制全部 token

再有 EPLB:
  按统计信息调整 expert placement，并为热点 expert 加 redundant copies

再有 Waterfill:
  shared expert 也进入 DeepEP dispatch，被调度到较空 rank

再有 LPLB:
  当前 batch 中 replicated experts 的 token 按 LP 结果分给不同 physical copies
```

## 11. 实践注意点

### 10.1 Waterfill

Waterfill 适合关注 shared expert 带来的 dense 负载：

```text
DeepSeek-V3/R1:
  普通 TopK path

DeepSeek V4:
  HashTopK path 也需要追加和 remap shared expert slot
```

启动时要注意：

```text
--enable-deepep-waterfill 会要求 DeepEP backend
如果未显式设置，当前源码会把 moe_a2a_backend override 到 deepep

Waterfill 需要 shared expert fusion
如果用户设置了 disable_shared_experts_fusion，当前源码会覆盖回 False
```

### 10.2 LPLB

LPLB 不是单独开一个参数就必然有收益。

它至少依赖：

```text
DeepEP MoE path
EPLB placement / init_expert_location
ep_num_redundant_experts > 0
ep_dispatch_algorithm = lp
```

并且要理解它的额外成本：

```text
每层每 batch:
  per-expert count
  EP all-reduce
  LP solve
  log2phy_prob extraction
  probability-based dispatch
```

所以 red0 场景下开 LPLB 通常是不合理的：

```text
没有 redundant copies
没有优化空间
只剩调度开销
```

### 10.3 语义不变不等于 bitwise 完全一致

文章说 Waterfill 和 LPLB preserve model semantics，原因是 logical expert choice 不变。

但实践中仍要注意：

```text
不同 rank / different dispatch order
可能影响浮点归约顺序、kernel 路径、随机采样路径
```

通常应该按 serving 质量和统计误差理解，而不是要求所有输出 bitwise identical。

## 12. 最终理解

Waterfill 和 LPLB 的共同目标是：

```text
不要让 EP group 的慢 rank 决定整层 MoE 的尾延迟
```

但二者的抓手不同：

```text
Waterfill:
  用低成本 heuristic 移动 shared expert work
  主要吃 shared expert dense 负载的平衡红利

LPLB:
  用更重的 LP 求解移动 redundant routed expert traffic
  主要吃 replicated experts 提供的 dispatch choice 红利
```

如果只记一句：

> Waterfill 让 shared expert 也变成 DeepEP 可调度负载；LPLB 让 redundant expert copies 的分流不再是均匀随机，而是按当前 batch 的最小最大负载目标动态决定。

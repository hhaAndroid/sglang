# DWDP 论文原理与 SGLang 实现技术报告

论文：

```text
DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72
arXiv:2604.01621v2
https://arxiv.org/abs/2604.01621v2
版本：v2，2026-05-12
作者：NVIDIA
```

对应 SGLang 实现：

```text
PR: https://github.com/sgl-project/sglang/pull/29778
合入提交: 37a830b667
标题: [Feature] Add DWDP (Distributed Weight Data Parallelism) for MoE prefill
```

本文同时覆盖两个层次：

```text
论文层：
  解释为什么要用 weight prefetch 替代 token all-to-all，
  以及 TensorRT-LLM 论文实现中的 TensorList groupedGEMM 和 TDM copy。

SGLang 层：
  解释 PR #29778 如何用 CUDA VMM composite virtual address、
  peer handle exchange、双缓冲和 CUDA event 把 DWDP 接入现有 FusedMoE。
```

需要始终记住：

> 论文描述的 DWDP 算法，与 SGLang PR #29778 的具体工程实现不是完全相同的。SGLang 保留了核心的异步 weight pull 思路，但 split-weight 和 copy contention 的处理方式与论文实现存在重要差异。

本地截图：

```text
hha_code/img/dwdp-1.png  # Figure 1: DEP 负载不均衡带来的同步等待
hha_code/img/dwdp-2.png  # Figure 2: DWDP group size 4 的整体执行方式
```

## 0. 一句话结论

DWDP 不是传统意义上的 TP / EP / PP 的又一种切分方式，而是一种更激进的 MoE 推理并行范式：

```text
传统 DEP:
  Attention 做数据并行
  MoE 做专家并行
  每层 MoE 前后通过 all-to-all 同步所有 rank

DWDP:
  每个 rank 仍然像独立 DP worker 一样跑完整请求
  attention 权重每卡全量复制
  MoE expert 权重跨 GPU 分布存放
  执行某层 MoE 前，把本 rank 缺失的远端 expert 权重异步拉到本地
  每个 rank 独立推进，不再在层边界做 collective 同步
```

它用“远端权重预取”替代“跨 rank token dispatch / combine”。收益来自两个地方：

```text
1. 去掉 DEP/EP 里的 all-to-all collective 和层级同步等待
2. 负载不均衡时，每个 rank 可以按自己的请求进度独立前进，不被最慢 rank 拖住
```

SGLang PR 的核心工程技巧可以再压缩成一句话：

```text
用 CUDA VMM 把“本地常驻专家物理页 + 远端专家双缓冲页”映射成
FusedMoE 看起来连续的完整 expert tensor，因而不修改现有 Triton MoE kernel。
```

代价也很明确：

```text
1. 需要非常高的 GPU peer-to-peer 带宽，例如 GB200 NVL72 / NVLink domain
2. 需要足够大的 prefill/context compute window 来隐藏远端 expert 权重预取
3. runtime、kernel、copy scheduling 都要配合，否则会被 D2D merge、copy engine 争用、功耗降频吃掉收益
```

所以 DWDP 最适合：

```text
大 MoE 模型
prefill/context 阶段
长上下文或较大 max_num_tokens
单 NVLink domain 内高带宽 P2P
线上请求长度和 expert routing 明显不均衡
PD 分离中可单独优化 context server 的场景
```

它不太适合：

```text
decode 阶段
短 prompt / 小 batch / 小 MNT
跨节点低带宽互联
dense 模型
TTFT 极度敏感且无法接受 context GPU 缩容导致排队增加的服务形态
```

## 1. 论文要解决的核心问题

### 1.1 传统模型并行都有层级同步

现在的大模型推理常用几类并行：

```text
TP:
  hidden dimension / matrix 切分
  每层需要 all-reduce / all-gather 等 collective

EP:
  expert 按 rank 切分
  token 需要 all-to-all dispatch 到 expert 所在 rank
  算完后再 all-to-all combine 回来

PP:
  layer 按 stage 切分
  stage 间有 pipeline 通信和调度依赖

DEP:
  attention 数据并行 + MoE expert parallel
  attention 每个 rank 独立算自己的 token
  MoE 通过 all-to-all 跨 rank 发 token
```

这些方法切分对象不同，但都有一个共同点：

```text
每层或每个阶段存在跨 rank 协调点。
```

只要有 collective，就会遇到一个基本问题：

```text
collective 的完成时间由最慢 rank 决定。
```

### 1.2 线上推理里的 rank 负载天然不均衡

论文 Figure 1，也就是本地截图 [dwdp-1.png](img/dwdp-1.png)，把这个问题画得很清楚。

在 DEP 中，每个 rank 的 attention 是 DP 形式：

```text
rank0: attention 处理自己的请求
rank1: attention 处理自己的请求
rank2: attention 处理自己的请求
rank3: attention 处理自己的请求
```

如果不同 rank 分到的请求输入长度不同：

```text
rank0/1/2: shorter ISL
rank3: longer ISL
```

那么 rank3 的 attention 会更慢。到了第一个 all-to-all，其他 rank 不能继续独立推进，只能等 rank3。

进入 MoE 后又会出现第二类不均衡：

```text
某些 expert 更热
某些 rank 收到更多 token
某些 rank 的 MoE grouped GEMM 更重
```

MoE 结束后的第二个 all-to-all 又会把这种不均衡暴露成等待。

论文给出的 Figure 1(b) 中，在 DeepSeek-R1 on GB200，ISL/OSL = 8K/1，input ratio = 0.8 的配置下，同步等待已经能达到约 12% 的 iteration latency。这个数值不是极端 case，而是论文认为线上常见不均衡程度下会出现的可观损耗。

### 1.3 调度和负载均衡只能缓解，不能消灭同步点

可以做很多传统优化：

```text
cache-aware scheduling
load-aware scheduling
expert load balancing
expert replica / redundant expert
batching 策略优化
```

这些方法能减少不均衡，但只要底层并行策略仍然需要 collective，那么：

```text
慢 rank 仍然会把局部抖动传播成全局等待。
```

DWDP 的立场是：

```text
不要只优化不均衡本身，而是让 rank 之间不再因为每层 collective 被迫同步。
```

## 2. DWDP 的核心思想

论文 Figure 2，也就是本地截图 [dwdp-2.png](img/dwdp-2.png)，展示的是 DWDP group size = 4 的执行方式。

### 2.1 权重怎么放

对一个 MoE 模型，DWDP 认为：

```text
attention / dense 部分:
  占模型总权重比例相对小
  每张 GPU 全量复制

MoE expert 权重:
  占模型总权重比例大
  在 DWDP group 内按 expert 分布到不同 GPU
```

以 DWDP4 为例：

```text
每个 rank 常驻:
  100% attention / dense 权重
  约 1/4 local experts

每个 rank 不常驻:
  约 3/4 remote experts
```

当 rank 要执行某一层 MoE 时，它需要该层所有必要 expert 权重。缺失的 remote experts 会从 peer GPUs 拉取到本地临时 buffer。

这里有一个非常关键的显存边界：DWDP 不是把全模型所有层的 remote experts 一次性拉到单卡上，而是按 MoE layer 管理临时权重。

```text
不是:
  rank 持有 attention/dense 全量
  + 所有层的 local experts
  + 所有层的 remote experts

而是:
  rank 常驻 attention/dense 全量
  + 所有层中属于本 rank 的 local experts
  + 当前 MoE layer 缺失的 remote experts 临时 buffer
  + 下一 MoE layer 预取用的 remote experts 临时 buffer
```

所以 DWDP 的峰值显存不是“完整 MoE 模型复制到每张卡”，但也不是没有额外显存压力。以 DWDP4 为例，某个 MoE layer 执行时，本 rank 可能临时持有：

```text
该层 1/4 local experts
+ 该层 3/4 remote experts
= 该层完整 expert 权重
```

如果同时做 double buffering，还可能在预取下一层时额外持有下一层的 remote expert buffer。因此更保守的峰值心智模型是：

```text
常驻:
  全量 attention/dense weights
  + 全部 MoE layers 的 local expert weights

临时:
  当前层 remote expert weights
  + 下一层 remote expert weights
  + activation / KV / workspace / communication buffers
```

这也是 DWDP 强依赖 GB200 NVL72 这类大显存、高带宽平台的原因之一。论文验证的是 DeepSeek-R1 NVFP4 checkpoint；低精度 MoE 权重显著降低了临时 remote buffer 的显存压力。

这和 EP 的方向正好相反：

```text
EP:
  权重固定在 expert 所在 rank
  token 被发到权重所在位置

DWDP:
  token 留在本 rank
  权重被拉到 token 所在位置
```

可以用一句话记：

```text
EP 是 move activation/token。
DWDP 是 move weight。
```

### 2.2 为什么叫 Distributed Weight Data Parallelism

它仍然保留 data parallel 的执行形态：

```text
每个 rank 都可以独立接收请求
每个 rank 都跑完整前向逻辑
每个 rank 自己返回结果
rank 之间没有每层 all-to-all / all-reduce 的强同步
```

但模型权重不是纯 DP 那样每卡全量复制，而是：

```text
MoE expert weights distributed across peer GPUs
```

所以名字里的三个词对应：

```text
Distributed Weight:
  MoE expert 权重分布式存放

Data Parallelism:
  每个 rank 仍然像 DP worker 独立处理请求

Distributed Weight Data Parallelism:
  用分布式权重存储支撑 DP 式异步推理
```

### 2.3 执行流水

以第 `l` 层和第 `l+1` 层为例，DWDP 的理想流水是：

```text
执行 layer l 的 MoE
执行 layer l+1 的 attention
同时异步预取 layer l+1 的 remote experts

等到 layer l+1 要执行 MoE 时:
  如果预取完成，直接执行
  如果没完成，只等待本 rank 自己缺的权重
```

论文里的描述是：

```text
prefetch(layer l+1 remote experts)
  overlap with MoE(layer l) + Attention(layer l+1)
```

这两个 compute block 共同构成隐藏远端权重搬运的窗口。

实现上需要 double buffering：

```text
buffer A:
  当前层 MoE 使用的 remote expert 权重

buffer B:
  下一层异步预取进来的 remote expert 权重

层推进后交换角色
```

这也解释了“某个请求或 batch 命中很多 experts 会不会 OOM”的边界：

```text
单个 token:
  MoE top-k 通常是固定小数，不会激活所有 experts

一个 batch:
  多个 token 合起来可能覆盖某层大量甚至全部 experts

DWDP 的保守实现:
  可能直接为当前层准备所有缺失 remote experts
  而不是只拉本 batch 实际命中的 expert 子集
```

因此 OOM 风险主要不来自“一个 token 需要所有 experts”，而来自：

```text
每层 remote expert buffer 太大
double buffering 叠加下一层预取
context batch / MNT 带来的 activation 和 workspace 太大
模型量化精度较高
DWDP group size 使 remote 比例过高
```

论文没有给出一个通用显存公式，也没有把 OOM 边界作为主实验维度；它的设计默认目标平台能容纳“一层级别的 remote expert 临时展开 + double buffer”。

一个更精确的实现还可以考虑“按本 batch 实际命中的 expert 子集预取”，但这会引入额外问题：

```text
需要更动态的 expert prefetch plan
预取粒度更碎，copy scheduling 更复杂
不同 batch 的命中集合变化会增加 runtime 开销
可能削弱连续大块 P2P copy 的带宽效率
```

论文当前更像是选择了工程上更稳定的 layer-wise split-weight 管理方式。

### 2.4 为什么不用 NCCL all-gather 拉权重

一个直觉做法是：

```text
用 NCCL all-gather 把 expert weights 聚起来
```

论文明确避开这个方向，因为 all-gather 本身就是 collective，会重新引入同步。DWDP 的目标是移除 collective synchronization，所以采用：

```text
cudaMemcpyAsync peer-to-peer pull
copy engine 驱动
serial P2P pulls
```

关键点：

```text
1. copy engine 搬运，不占 SM
2. 每个 rank 自己发起 pull
3. 不要求所有 rank 在同一层、同一时刻一起进入 collective
4. rank 之间进度可以自然错开
```

这也是 DWDP 的工程本质：

```text
把 collective communication 问题改造成每个 rank 的异步 P2P DMA 预取问题。
```

## 3. 和 DEP / EP 的本质差异

### 3.1 数据流差异

DEP/EP 的 MoE 数据流：

```text
每个 rank 本地 token
  -> router/topk
  -> all-to-all dispatch token 到 expert rank
  -> expert rank 做 MoE compute
  -> all-to-all combine 回 token 原 rank
```

DWDP 的 MoE 数据流：

```text
每个 rank 本地 token
  -> router/topk
  -> 本地执行 MoE
     本地 expert 直接用
     remote expert 先被 P2P 拉到本地 buffer
  -> 输出仍在本 rank
```

### 3.2 同步模型差异

DEP/EP：

```text
所有 rank 必须在 all-to-all 汇合
某个 rank attention 慢，会让其他 rank 等
某个 rank MoE token load 高，会让其他 rank 等
```

DWDP：

```text
没有 layer-wise collective
rank 只等自己需要的远端权重
不同 rank 可以处在不同 layer / 不同请求进度
```

DWDP 不是完全没有等待。它仍然可能等待：

```text
remote expert prefetch 没完成
copy engine 争用导致预取变慢
通信与计算重叠导致 kernel 变慢
```

但这种等待是本 rank 局部等待，不再是 collective 导致的全局等待。

### 3.3 资源使用差异

DEP/EP 的成本：

```text
activation/token 通信
all-to-all latency
同步等待
expert load imbalance
```

DWDP 的成本：

```text
MoE 权重 P2P 传输
remote expert 临时 buffer
double buffering
copy engine / NVLink / HBM / L2 / power contention
GroupedGEMM kernel 需要支持 split-weight
```

所以 DWDP 把瓶颈从：

```text
跨 rank activation collective
```

转移到：

```text
跨 GPU weight prefetch + kernel/runtime 协同
```

## 4. 初步性能模型：什么时候 DWDP 会赢

论文用 layer-wise roofline 模型分析 DWDP4 对 DEP4。

可以把单层粗略写成：

```text
DEP layer time:
  compute_time + expert_parallel_all_to_all_time + sync_wait

DWDP layer time:
  max(compute_window, remote_weight_prefetch_time) + residual_interference
```

这里的 compute window 主要来自：

```text
MoE(layer l) + Attention(layer l+1)
```

如果：

```text
compute_window >= remote_weight_prefetch_time
```

那么远端权重预取基本可以被隐藏，DWDP 主要收益就是省掉 DEP 的 all-to-all 和同步等待。

如果：

```text
compute_window < remote_weight_prefetch_time
```

那么 rank 在进入 MoE 前要等权重，DWDP 的优势会被削弱，甚至不如 DEP。

论文 Figure 3 的一个重要结论：

```text
batch size = 1 时，DeepSeek-R1 context phase 在 GB200 上，
DWDP4 大约到 16K tokens 左右开始明显优于 DEP4。
```

但这个 16K 不是普适阈值。论文后续实验说明：

```text
如果 runtime 能形成更大的 effective batch
或者 max_num_tokens 足够大
即使单请求 ISL 更短，也可能让 DWDP 受益
```

这对系统实现很关键：

```text
DWDP 的收益不只看单条请求长度，还看 serving runtime 每次 context forward 能聚合多少 token。
```

## 5. 实现挑战一：split-weight merge overhead

### 5.1 问题来源

DWDP 下，一层 MoE 的权重天然分成：

```text
local experts:
  常驻本 GPU

remote experts:
  当前层执行前从 peer GPU 拉到临时 buffer
```

很多现有 MoE groupedGEMM kernel 假设：

```text
所有 expert weights 在一个连续 buffer 里
```

最朴素实现会在 MoE kernel 前做一次 D2D copy：

```text
local experts + remote experts
  -> merge 到 contiguous weight buffer
  -> groupedGEMM
```

这会产生额外 device-to-device copy，占 HBM 带宽和时间。

论文 Table 1 中，baseline DWDP 相比 DEP 虽然去掉了 communication 和 sync cost，但出现了：

```text
D2D Copy: 34.00
P2P Copy: 429.00  # 被异步 overlap，不直接算 critical path 同类项目
```

其中 D2D merge 是明确的 critical path overhead。

### 5.2 论文优化：GroupedGEMM 直接吃多个 weight buffer

论文的解决方式：

```text
扩展 MoE groupedGEMM kernel
让 kernel 接收 TensorList-based inputs
在 kernel 内根据 expert id 选择 local buffer 或 remote buffer
避免预先 merge 成一个 contiguous buffer
```

也就是从：

```text
pre-merge weights -> groupedGEMM(contiguous)
```

改成：

```text
groupedGEMM(tensor_list_of_weight_buffers)
```

这个优化的意义很直接：

```text
省掉 D2D merge copy
减少 HBM 额外读写
让 DWDP 的 weight split 不泄漏成 MoE kernel 前的固定开销
```

论文报告：

```text
split-weight merge elimination 可以让 DWDP baseline 额外提升约 3% TPS/GPU
且 groupedGEMM 本身没有明显 regression
```

### 5.3 SGLang 采用了另一条路：VMM composite virtual address

论文的 TensorRT-LLM 实现选择修改 groupedGEMM，让 kernel 直接接收多个 weight buffer；但这不是消除 merge copy 的唯一方法。

SGLang PR #29778 使用 CUDA Virtual Memory Management（VMM）构造一个 composite virtual address：

```text
同一个逻辑 expert tensor 的虚拟地址空间：

低 expert id                                                     高 expert id
┌────────────────────┬────────────────────┬────────────────────┐
│ remote pre region  │ local expert pages │ remote post region │
│ 由 page pool 支撑   │ 由本 rank handle 支撑│ 由 page pool 支撑   │
└────────────────────┴────────────────────┴────────────────────┘

FusedMoE 看到的仍然是：
  weight.shape == [num_global_experts, ...]
  地址连续
```

于是 SGLang 不需要让 Triton grouped MoE kernel 理解 TensorList，也不需要在 kernel launch 前把 local/remote 两部分 D2D merge 到第三个 buffer：

```text
论文方案：
  多个物理 buffer
  -> kernel 接收 TensorList
  -> kernel 内按 expert id 选 buffer

SGLang 方案：
  多个物理 allocation
  -> CUDA VMM 映射到连续 virtual address
  -> kernel 继续读取普通连续 tensor
```

因此更准确的结论不是“MoE kernel 必须修改”，而是：

```text
系统必须提供一种零 merge-copy 的 split-weight 消费机制。

它可以是：
  1. 论文的 multi-buffer / TensorList kernel；或
  2. SGLang 的 VMM composite VA；或
  3. 其他能够把逻辑 expert id 映射到正确物理地址的 kernel/runtime 协同方案。
```

两条路线各有代价：

```text
TensorList kernel:
  优点：物理布局直观，placement 更灵活
  代价：需要修改并维护每种 MoE/量化 kernel

VMM composite VA:
  优点：复用现有连续 tensor kernel，接入面较小
  代价：需要处理 VMM granularity、页边界、handle 生命周期、
       DLPack/raw pointer tensor、平台兼容和异常清理
```

SGLang 的 VMM 实现只把主 expert weights `w13_weight`、`w2_weight` 放进 composite VA；量化 scale、alpha、bias 等按 expert 切分的小 tensor 在初始化时通过 `all_gather` 复制成全量。这是可行的，因为这些 side tensor 相对主权重较小，但它也意味着初始化阶段并不是“零 collective”。

## 6. 实现挑战二：异步 P2P copy 的 many-to-one 争用

### 6.1 问题来源

DWDP 中每个 rank 都会拉其他 rank 的 remote experts。以 DWDP4 为例：

```text
rank0 拉 rank1/2/3
rank1 拉 rank2/3/0
rank2 拉 rank3/0/1
rank3 拉 rank0/1/2
```

如果多个 destination rank 同时从同一个 source rank 拉权重：

```text
rank0 -> pull from rank2
rank1 -> pull from rank2
rank3 -> pull from rank2
```

那么 source rank2 的 copy engine / memory path 会成为 many-to-one 争用点。论文 Figure 4 的 Nsight trace 显示，这种 source-side serialization 会拉长通信窗口，导致下一段 compute 前出现 bubble。

这类争用不需要极端调度才会出现。论文用随机模型说明：

```text
DWDP4:
  contention level 1: 44.44%
  contention level 2: 44.44%
  contention level 3: 11.11%

DWDP8:
  contention level 1: 39.66%
  contention level 2: 39.66%
  contention level 3: 16.52%
  contention level 4: 3.67%
```

group 越大，高阶争用概率越高。

### 6.2 论文优化：copy with time-division multiplexing

论文的优化不是让所有 rank 同步调度 copy，而是把每个大块 remote weight transfer 切成小 slice，然后 round-robin 交错发起：

```text
原始:
  pull(peer0, entire shard)
  pull(peer1, entire shard)
  pull(peer2, entire shard)

TDM slicing:
  pull(peer0, slice0)
  pull(peer1, slice0)
  pull(peer2, slice0)
  pull(peer0, slice1)
  pull(peer1, slice1)
  pull(peer2, slice1)
  ...
```

论文 Listing 1 的伪代码核心就是：

```text
for each parameter:
  for offset in 0..M step slice_size:
    for peer in round_robin(remote_peers):
      append copy(dst(peer, offset), src(peer, offset), chunk)
```

它的直觉是：

```text
不要让一个 destination 用一个大 DMA request 长时间占住 source-side copy path。
把大请求切片后，让不同 destination 在 slice 粒度上交替前进。
```

论文使用的一个实验配置是 1MB slice。Table 4 说明，TDM 对 compute window 短的场景帮助最大：

```text
ISL ratio 0.5, MNT 16384:
  DEP: 1.000
  DWDP + merge elimination: 0.995
  Full DWDP: 1.081

ISL ratio 0.5, MNT 32768:
  DEP: 1.000
  DWDP + merge elimination: 1.140
  Full DWDP: 1.139
```

解释：

```text
MNT 小 / 平均 ISL 小:
  compute window 短
  copy 抖动更容易暴露成 bubble
  TDM 更有价值

MNT 大:
  compute window 足够长
  baseline 已能隐藏大部分 copy
  TDM 额外收益变小
```

## 7. 实现挑战三：通信计算重叠会触发硬件干扰

DWDP 的理想假设是：

```text
P2P copy 由 copy engine 负责，不占 SM
因此可以和 attention/MoE compute 完美 overlap
```

论文 Appendix A 说明这个假设并不完全成立。

即使 copy engine 不占 SM，P2P 权重搬运仍然要经过：

```text
source GPU:
  DRAM -> L2 -> NoC -> NVLink

destination GPU:
  NVLink -> NoC -> L2/DRAM
```

所以它会和 compute kernel 共享：

```text
NoC
L2
HBM bandwidth
power budget
```

### 7.1 memory-bound kernel 的 HBM/L2 争用

对量化、copy、elementwise 这类 memory-bound kernel，NVLink traffic 会消耗 HBM bandwidth。

论文给出 Blackwell 上的粗略上界：

```text
HBM peak bandwidth: 约 8 TB/s
NVLink 5 aggregate read/write bandwidth: 约 1800 GB/s

1800 GB/s / 8 TB/s ~= 22.5%
```

所以如果 memory-bound kernel 和满速 NVLink copy 完全重叠，理论上可能有约 22.5% 级别的 slowdown 上界。

论文 Table 1 里 Others 从 DEP4 的 241.69 增加到 DWDP4 的 284.32，约 17.6% slowdown，和这个解释一致。

### 7.2 compute-intensive attention 的主要瓶颈是功耗降频

更有意思的是 attention。论文认为对 DeepSeek-R1 context attention 这种 compute-intensive kernel，主要 slowdown 不是 HBM/L2/NVLink 饱和，而是：

```text
power-induced frequency throttling
```

也就是：

```text
attention 本身已经吃掉很高功耗
P2P copy 同时运行又增加功耗
总功耗超过 TDP/TGP power cap
GPU 触发 DVFS 降频
attention kernel 变慢
```

论文 Table 7 比较三种模式：

```text
Intermittent Compute:
  normalized kernel time 1.000
  normalized GPU frequency 1.000

Long-Duration Overlap:
  normalized kernel time 1.049
  normalized GPU frequency 0.963

Short-Duration Overlap:
  normalized kernel time 1.226
  normalized GPU frequency 0.798
```

这个结果说明：

```text
attention slowdown 和 GPU frequency drop 高度相关
memory utilization 只有 45%-50% 左右，并没有接近饱和
NVLink throughput 也不是主要抖动源
```

对 DWDP 的关键启发：

```text
copy 越细粒度、越频繁地插入高功耗 compute 区间，
越可能制造持续 power pressure，
让 compute kernel 降频。
```

这和 TDM slice 优化之间存在张力：

```text
slice 太大:
  many-to-one 争用更明显
  copy bubble 更大

slice 太小:
  copy 更频繁
  runtime overhead 和 power interference 可能增加
```

因此 slice size 不是越小越好，需要按硬件和 workload 调参。

## 8. 实验结果解读

### 8.1 实验环境

论文实验条件：

```text
硬件:
  GB200 NVL72

框架:
  TensorRT-LLM
  基于 commit 3a89495
  upstream integration 在 TensorRT-LLM PR #12136 中推进

模型:
  DeepSeek-R1
  NVFP4 checkpoint
  MoE weights 使用 NVFP4
  attention 使用 FP8 KV cache

服务架构:
  disaggregated serving
  DWDP 应用于 context server
  generation server 配置保持不变
```

这个设置很重要。论文不是说 DWDP 在任意 LLM serving 中都提升 8.8%，而是：

```text
在 GB200 NVL72 + DeepSeek-R1 + PD 分离 + context server 的特定高带宽场景下，
DWDP 改善 context GPU 使用效率，并在端到端 Pareto 上带来收益。
```

### 8.2 context-only 结果

论文 Table 3 的核心结论：

```text
固定 MNT=32768，ISL 从 1K 到 32K:
  TPS/GPU speedup 约 1.09-1.11
  TTFT speedup 约 1.11-1.27

固定 ISL=8192，MNT 从 16384 到 32768:
  MNT=16384: TPS/GPU 1.01, TTFT 1.07
  MNT=32768: TPS/GPU 1.10, TTFT 1.16

固定 ISL=16384，输入长度标准差增大:
  STD 0: TPS/GPU 1.09
  STD 4096: TPS/GPU 1.15
```

这些结果符合 DWDP 的设计预期：

```text
MNT 越大:
  compute window 越大
  越能隐藏 remote weight prefetch

负载越不均衡:
  DEP collective sync 等待越多
  DWDP 的异步独立推进越有优势
```

DWDP3 和 DWDP4 的 TPS/GPU speedup 接近：

```text
DWDP3: 1.093
DWDP4: 1.091
```

但 DWDP3 TTFT speedup 为 0.86，论文解释可能是 context-side aggregate throughput 下降导致 queueing delay 增加。

这说明 DWDP 的一个实际价值是：

```text
支持更细粒度的 context GPU 数量配置，
但 GPU 数量缩得太激进会伤 TTFT。
```

### 8.3 end-to-end 结果

端到端实验设置：

```text
dataset:
  SemiAnalysis

最大 input length:
  8K

output length:
  1K

input lengths:
  6.4K 到 8K
  input ratio = 0.8

目标:
  在类似 TPS/user 下，提高 output TPS/GPU
```

论文 Table 5：

```text
TPS/user 20-30:
  TPS/user speedup 1.15
  TPS/GPU speedup 1.10

TPS/user 40-50:
  TPS/user speedup 1.16
  TPS/GPU speedup 1.08

TPS/user 60-70:
  TPS/user speedup 1.00
  TPS/GPU speedup 1.10

TPS/user 80-90:
  TPS/user speedup 1.00
  TPS/GPU speedup 1.06

TPS/user 170-180:
  TPS/user speedup 1.00
  TPS/GPU speedup 0.97
```

摘要里的 8.8% 可以理解为：

```text
在 20-100 TPS/user serving range，
DWDP 在相近 TPS/user 下平均提升 end-to-end output TPS/GPU 约 8.8%。
```

论文的解释非常务实：

```text
多数 baseline Pareto 点是 generation-bottlenecked。
只加速 context 不一定直接提高端到端总吞吐。
DWDP 的主要收益是减少 context GPU 需求。
同样 generation 配置下，用更少 context GPU 维持类似 TPS/user，
于是 output TPS/GPU 更高。
```

### 8.4 TTFT 代价

论文 Table 6 显示 DWDP 的 TTFT 会变差：

```text
TPS/user 20-30:
  baseline TTFT 2538 ms
  DWDP TTFT 8314 ms

TPS/user 40-50:
  baseline TTFT 1919 ms
  DWDP TTFT 7012 ms

TPS/user 60-70:
  baseline TTFT 965 ms
  DWDP TTFT 1640 ms

TPS/user 80-90:
  baseline TTFT 1669 ms
  DWDP TTFT 2280 ms

TPS/user 170-180:
  baseline TTFT 494 ms
  DWDP TTFT 660 ms
```

这点非常关键。DWDP 的端到端收益不是“无条件更低延迟”，而是：

```text
更高 output TPS/GPU / 更省 context GPU
但某些 Pareto 点上 TTFT 明显上升
```

低 TPS/user 区间 TTFT 变差尤其明显，原因不是 DWDP 单卡 context 效率低，而是：

```text
为了提高 GPU 效率，DWDP 点通常减少了 context GPU 数
context stage aggregate service rate 下降
context 和 generation 的 rate matching 变差
请求排队时间增加
```

所以实际部署时不能只看 TPS/GPU，需要同时约束：

```text
TTFT SLO
TPS/user SLO
output TPS/GPU
context/generation 资源配比
queueing delay
```

## 9. DWDP 的适用边界

### 9.1 适合场景

DWDP 适合以下条件同时成立的场景：

```text
模型:
  大 MoE 模型
  MoE expert 权重占大头
  attention/dense 权重可以承受全量复制
  每卡显存能容纳全量 dense/attention + local experts + 1-2 层 remote expert 临时 buffer

硬件:
  单个高带宽 GPU domain
  任意 GPU peer-to-peer bandwidth 足够高
  copy engine 和 NVLink 能支撑高频远端权重拉取

workload:
  context/prefill 为主要可优化阶段
  prompt 较长，或 MNT 较大，能形成足够 compute window
  请求长度、KV cache 命中、expert routing 存在明显不均衡

服务架构:
  PD disaggregation
  可以只优化 context server
  generation server 保持固定
  能接受通过调 context GPU 数量来找 Pareto 点
```

### 9.2 不适合场景

DWDP 不适合以下情况：

```text
decode-heavy:
  decode 每步 token 数小
  单层 compute window 很短
  远端权重预取难隐藏

低带宽互联:
  跨节点 IB / Ethernet 拉 expert weights
  权重搬运成本过高

小模型或 dense 模型:
  expert 权重不是主要内存压力
  复制完整权重可能更简单

显存余量不足:
  当前层 remote experts + 下一层 prefetch buffer 放不下
  或者 activation / KV / workspace 已经把显存打满
  此时 DWDP 会把 EP 通信瓶颈换成 OOM 风险

严格低 TTFT:
  如果为了提高 TPS/GPU 缩 context GPUs，
  排队可能让 TTFT 显著变差

既不能改 kernel、也不能使用 VMM 等地址拼接机制:
  groupedGEMM 只能吃单个 contiguous physical buffer
  必须 D2D merge，则收益会被侵蚀
```

### 9.3 一个判断公式

可以用下面的工程判断式：

```text
DWDP 是否值得尝试
  ~= DEP_sync_overhead + DEP_all_to_all_cost
     > exposed_remote_prefetch_cost + overlap_interference + extra_runtime_overhead
```

更具体：

```text
如果:
  1. DEP 下同步等待明显
  2. all-to-all 在 critical path 上占比高
  3. context compute window 足够隐藏 P2P weight copy
  4. kernel 能直接消费 split weights，或者 VMM 能把 split weights
     暴露成连续逻辑 tensor
  5. 每卡显存能承受 layer-wise remote expert buffer 和 double buffering
  6. TTFT SLO 允许重新配 context/generation 资源

那么:
  DWDP 有机会改善 TPS/GPU Pareto
```

## 10. 从 SGLang 角度理解 DWDP

这篇论文对 SGLang/DeepEP 系统有三个层面的启发。

### 10.1 DWDP 和 DeepEP 是两条不同路线

DeepEP / EP 路线：

```text
expert 权重固定在 EP rank
dispatch token 到对应 rank
优化目标:
  all-to-all 更快
  token dispatch 更均衡
  expert load balance 更好
  combine 更低开销
```

DWDP 路线：

```text
token 不 dispatch 到远端 expert
remote expert weight 被拉到本地
优化目标:
  P2P prefetch 可隐藏
  split-weight kernel 无额外 merge
  copy scheduling 避免 source contention
  避免 communication-compute power interference
```

两者不是小改参数的关系，而是 MoE 执行方向的反转：

```text
DeepEP:
  weight stationary, token moves

DWDP:
  token stationary, weight moves
```

### 10.2 和 PD 分离天然相关

SGLang 里的 PD 分离可以理解为：

```text
prefill/context server:
  长 prompt
  大 context batch
  TTFT 关键

decode/generation server:
  每步少量 token
  output TPS 关键
```

DWDP 论文只把 DWDP 放在 context server，因为：

```text
context 阶段有更大的 compute window
decode 阶段很难隐藏整层 remote expert weight prefetch
```

这和 PD 架构高度匹配：

```text
context side:
  可以用 DWDP 减少 EP collective 等待
  可以用更少 GPU 达到类似 context service rate

decode side:
  继续使用传统 TP/EP/DeepEP 路线
  避免每 token decode 时拉大块 expert 权重
```

### 10.3 和 Waterfill / LPLB 的关系

之前 DeepEP 方向的 Waterfill / LPLB 解决的是：

```text
在 EP token dispatch 框架内，
更好地把 token 分配到 physical expert / rank，
减少 MoE compute 长尾。
```

DWDP 解决的是：

```text
直接绕开 EP dispatch/combine collective，
避免每层全 rank 同步。
```

可以理解为两个层级：

```text
Waterfill / LPLB:
  在 collective 存在的前提下减少不均衡

DWDP:
  尝试移除 collective，让不均衡不再变成全局等待
```

如果未来系统同时支持两类策略，选择可能取决于：

```text
硬件互联是否足够强
context compute window 是否足够大
TTFT vs TPS/GPU 哪个更重要
是否能改 kernel 和 runtime
是否已有成熟 DeepEP load balancing 能把同步等待压低
```

## 11. SGLang PR #29778 的实现核心逻辑

这一节不再描述“如果要实现 DWDP 应该有什么”，而是沿着 SGLang 的真实代码执行路径说明它已经做了什么。

分析对象：

```text
PR #29778 merge commit: 37a830b667
当前工作树：包含该 PR 之后的 lazy import 修复和 CUDA VMM 公共组件重构
```

原 PR 自带 `python/sglang/srt/layers/moe/dwdp/vmm.py`；当前主线已经把大部分通用 VMM primitive 重构到 `python/sglang/srt/cuda_vmm_utils.py`。这只是代码归属变化，不改变下面描述的核心机制。

### 11.1 代码模块地图

| 模块 | 作用 |
|---|---|
| [`server_args.py`](../python/sglang/srt/server_args.py) | 校验 DWDP 约束，并强制切换 DP attention、DP LM head、EP layout 等配置 |
| [`model_runner.py`](../python/sglang/srt/model_executor/model_runner.py) | 模型加载后初始化 `DwdpManager`；每次 forward 前启动前两层预取 |
| [`dwdp_manager.py`](../python/sglang/srt/layers/moe/dwdp/dwdp_manager.py) | 总控：发现 MoE 层、建立 layout/transport/buffer、绑定新权重、清理资源 |
| [`layout.py`](../python/sglang/srt/layers/moe/dwdp/layout.py) | expert ownership、weight shape、页对齐 composite VA 布局 |
| [`transport.py`](../python/sglang/srt/layers/moe/dwdp/transport.py) | 把本地权重搬进可共享 VMM allocation，交换/import peer handles，建立 peer tensor view |
| [`page_pool.py`](../python/sglang/srt/layers/moe/dwdp/page_pool.py) | 为 remote expert 区域提供两组 ping-pong 物理页 |
| [`weight_buffer.py`](../python/sglang/srt/layers/moe/dwdp/weight_buffer.py) | 把 local handle 与 remote page pool 拼成每层的完整逻辑 expert tensor |
| [`weight_manager.py`](../python/sglang/srt/layers/moe/dwdp/weight_manager.py) | copy stream、prefetch/consume event、两层 ahead 预取协议 |
| [`fused_moe_triton/layer.py`](../python/sglang/srt/layers/moe/fused_moe_triton/layer.py) | 把原 EP FusedMoE 改绑成“本地拥有全部 expert”的逻辑视图，并插入 wait/record hook |
| [`dp_attn.py`](../python/sglang/srt/managers/scheduler_components/dp_attn.py) | 允许 DWDP rank 独立运行，空闲 rank 不再构造同步用 idle batch |

### 11.2 参数解析阶段：把一个 TP world 改造成 DWDP/DP world

用户入口是：

```bash
python -m sglang.launch_server \
  --tp 4 \
  --dwdp-size 4 \
  --disaggregation-mode prefill \
  ...
```

当前 SGLang v1 实现要求：

```text
dwdp_size > 1
dwdp_size == tp_size
disaggregation_mode in {null, prefill}
pp_size == 1
enable_eplb == false
speculative_algorithm is None
enable_two_batch_overlap == false
```

启用后 `_handle_dwdp()` 强制设置：

```text
dp_size = dwdp_size
enable_dp_attention = true
enable_dp_attention_local_control_broadcast = true
enable_dp_lm_head = true
moe_dense_tp_size = 1
ep_size = dwdp_size
moe_ep_size = dwdp_size
moe_dp_size = 1
moe_a2a_backend = none
SGLANG_SCHEDULER_SKIP_ALL_GATHER = true
disable_cuda_graph = true
```

这里容易产生一个误解：启动参数仍然写 `--tp 4`，并不代表 DWDP forward 中 attention 和 MoE 仍按普通 TP4 执行。

更准确的理解是：

```text
--tp 4:
  先创建 4 个 model worker/rank 和对应 process groups

DWDP 配置重写：
  attention 变成 4 路数据并行
  dense/shared 部分在每个 rank 上完整执行
  routed expert 在加载阶段按 EP4 分片
  DWDP setup 后每个 rank 逻辑上重新拥有全部 routed experts
```

为什么先设 `ep_size=4`？因为 checkpoint load 阶段不能把所有专家完整加载到每张卡，否则失去分布式权重存储的意义。SGLang 先复用已有 EP loader，让每个 rank 只加载本地专家；加载完成后再把这些 shard 转换成 DWDP 可共享的 VMM allocation。

standalone 的 `disaggregation_mode=null` 虽然允许启动，但代码会明确警告：decode 每一步都会重新获取所有 remote expert weights，性能很差。因此真正目标是 PD 分离中的 prefill worker。

### 11.3 初始化时机：先正常加载模型，再改造 expert storage

`ModelRunner` 的顺序可以概括为：

```text
1. 根据 EP layout 创建 FusedMoE parameter
2. checkpoint loader 把每个 rank 的 local expert shard 加载进 parameter
3. ModelRunner.maybe_init_dwdp()
4. DwdpManager.setup(model)
5. 把 FusedMoE weight parameter 重新绑定到 composite VMM tensor
```

当前 `maybe_init_dwdp()` 在确认不是 draft worker 且 `dwdp_size > 1` 后才 lazy import CUDA-only DWDP 模块。这个 lazy import 是 PR 合入后的修复：原始 PR 在文件顶层 import `DwdpManager`，曾导致没有 `cuda-python` 的 CPU/XPU/NPU 环境即使不启用 DWDP 也 import 失败。

`DwdpManager.setup()` 首先扫描 transformer layers：

```python
decoder = model.model if hasattr(model, "model") else model
for layer_idx, layer in enumerate(decoder.layers):
    experts = first module in layer that isinstance(FusedMoE)
```

然后检查：

```text
至少找到一个 FusedMoE
所有 MoE layer 的 num_global_routed_experts 相同
num_routed_experts % dwdp_size == 0
```

这一点与论文的 placement 能力不同。论文强调 DWDP 可通过冗余 placement 支持“不整除”；SGLang v1 明确要求整除，也不支持一层多个独立 `FusedMoE` 实例或动态 expert placement。

SGLang 当前只把两个主权重交给 VMM：

```text
w13_weight
w2_weight
```

对每个 `(layer_idx, weight_name)` 构造 `WeightSpec`：

```text
chunk_shape = 当前 rank 的 EP shard shape
full_shape  = [num_global_experts] + chunk_shape[1:]
expert_bytes = product(full_shape[1:]) * dtype_size
```

### 11.4 expert ownership：SGLang v1 是连续、等分、静态分片

当 `E` 个 experts 被 `N` 个 DWDP ranks 等分时：

```text
experts_per_rank = E / N

rank r owns:
  [r * experts_per_rank, (r + 1) * experts_per_rank)
```

代码中的 `num_prefetch_experts` 公式看起来更一般：

```text
ceil((E - E/N) / (N - 1))
```

但因为 setup 已经要求 `E % N == 0`，它会化简为 `E/N`。因此当前 `peer_ranges` 实质上就是普通连续 EP shard。

例如 E=8、N=4：

```text
rank0 owns experts [0, 2)
rank1 owns experts [2, 4)
rank2 owns experts [4, 6)
rank3 owns experts [6, 8)
```

没有论文所说的 redundant expert placement，也没有 EPLB permutation；这正是启用 DWDP 时禁止 EPLB 的原因之一。

### 11.5 Transport phase 1：把 PyTorch local shard 搬进可共享 VMM allocation

对每一层的 `w13_weight` 和 `w2_weight`，`_copy_local_weights_to_handles()` 执行：

```text
1. 根据 local expert 的全局 byte range 计算 page-aligned 范围
2. cuMemCreate/VmmReservation 创建可共享物理 allocation
3. 临时 reserve 一段 VA 并 map allocation
4. cuMemcpyDtoD 把原 PyTorch parameter shard 搬进 allocation
5. synchronize，关闭临时 VA mapping，但保留 allocation handle
6. param.untyped_storage().resize_(0) 释放原 PyTorch storage
7. torch.cuda.empty_cache()
```

这里需要区分三个概念：

```text
physical allocation:
  真正占 GPU 显存，保存本 rank 的 local expert shard

shareable handle:
  其他进程/rank 用来 import 这块 physical allocation 的句柄

virtual address mapping:
  某个进程把 allocation 映射到自己 CUDA VA space 后得到的可访问地址
```

原 parameter storage 被释放后，本地专家的唯一真实数据就在 VMM physical allocation 中。后面本 rank 的完整逻辑 tensor 和 peer rank 的只读 view，都会映射同一个 allocation，而不是再复制一份常驻权重。

### 11.6 Transport phase 2：跨 rank 交换 handle，并建立 peer views

DWDPTransport 对每个本地 allocation 导出 shareable handle：

```text
支持 FABRIC/IMEX：
  导出 CUDA fabric handle
  通过 CPU process group all_gather_object 交换

不支持 FABRIC：
  导出 POSIX file descriptor
  通过 pidfd/Unix 进程机制交换 fd
```

每个 rank 随后 import 其他 `N-1` 个 rank 的每层权重 handle，为它们 reserve/map 一段本进程 VA，并创建形状为：

```text
[num_peer_experts, ...]
```

的 `peer_tensor`。

这些 `peer_tensor` 的含义不是“remote weights 已经复制到本地”：

```text
peer_tensor:
  本进程 CUDA VA 中的一个 view
  背后映射的是 peer rank 的物理显存

对 peer_tensor 做 src -> local dst copy：
  才会发生真正的 NVLink/P2P weight pull
```

所有 handle import 完成后初始化阶段会执行一次 group barrier。由此可见，DWDP 的“无同步”只成立于 steady-state forward critical path，而不是模型初始化全过程。

### 11.7 Composite VA：怎样让 split weights 看起来是连续 tensor

假设某权重每个 expert 占 `expert_bytes`，本 rank 拥有 expert 区间 `[local_start, local_end)`：

```text
local_start_bytes = local_start * expert_bytes
local_end_bytes   = local_end   * expert_bytes

page_start = align_down(local_start_bytes, vmm_granularity)
page_end   = align_up(local_end_bytes, vmm_granularity)
```

本地 VMM handle 实际覆盖 `[page_start, page_end)`，而不是精确的 expert byte boundary，因为 `cuMemMap` 必须按 allocation granularity 映射。

`WeightBuffer` 为每个 `(layer, weight)` reserve 一段足够容纳全部 experts 的连续虚拟地址，然后这样映射：

```text
VA base
│
├─ pre region
│    映射 page_pool[slot]，保存 local_start 之前的 remote experts
│
├─ local/MNNVL region
│    映射本 rank 永久持有的 shareable expert handle
│
└─ post region
     映射同一个 page_pool[slot] 的后续页，
     保存 local_end 之后的 remote experts
```

最后从带 `pre_padding` 的地址创建：

```python
full_tensor.shape = [num_global_experts, ...]
```

于是对 MoE kernel 而言：

```text
full_tensor[0]
full_tensor[1]
...
full_tensor[E-1]
```

都是普通连续寻址。kernel 不知道其中某些地址映射到常驻 local handle，另一些地址映射到可反复覆盖的 page pool。

#### 页边界为什么需要特殊处理

如果 `expert_bytes` 不是 VMM page size 的整数倍，本地 handle 的首尾页可能同时包含少量相邻 remote expert bytes：

```text
page_start < local_start_bytes
page_end   > local_end_bytes
```

这些 leading/trailing edge bytes 不能映射到另一个 allocation，因为同一 VA page 不能被拆成两个物理来源。SGLang 选择让整页归本地 handle，然后在 setup 时通过 `_fill_edge_bytes()` 从 peer view 把边缘相邻 expert 的数据复制进本地 handle 的空白区域。

这个细节很重要：VMM composite VA 不是简单把三个 tensor `cat` 到地址空间；它必须解决物理页粒度与 expert byte boundary 不重合的问题。

### 11.8 PagePool：所有层共享两组 remote physical buffers

如果每层都永久分配自己的 remote region，DWDP 会接近在每张卡复制完整 MoE 权重，失去显存优势。

SGLang 为所有层只创建两个物理 page slots：

```text
slot 0: 偶数序号 MoE layer 使用
slot 1: 奇数序号 MoE layer 使用
```

这里的“偶数/奇数”是 MoE layer 在 `_moe_layer_indices` 中的序号，而不一定是 transformer 的原始 layer id 奇偶。

每个 slot 的物理大小是该 slot 所有候选层中 remote region 最大值：

```text
slot_size[s] = max over layer assigned to s (
    sum over weight names (pre_size + post_size)
)
```

不同层各自保留独立的完整 virtual address reservation，但同一 parity 的 layer 把 remote VA region 映射到同一组 page-pool physical handles。于是：

```text
layer0 remote VA ─┐
layer2 remote VA ─┼─> same slot0 physical pages
layer4 remote VA ─┘

layer1 remote VA ─┐
layer3 remote VA ─┼─> same slot1 physical pages
layer5 remote VA ─┘
```

这就是“双缓冲”真正节省显存的原因：VA reservation 可以很多，但 remote physical allocation 只有两套。

### 11.9 把 FusedMoE 从 EP shard 重新绑定成完整逻辑专家集

composite full tensor 建立后，`bind_full_expert_weights()` 修改 `FusedMoE`：

```text
moe_ep_size = 1
moe_ep_rank = 0
num_local_routed_experts = num_global_routed_experts
dispatcher.local_expert_mapping = None
dispatcher.expert_mask_gpu = None
w13_weight/w2_weight -> composite full tensors
```

这一步的语义是：

```text
加载阶段：
  这个 FusedMoE 物理上是 EP shard，只拥有 E/N experts

DWDP setup 后：
  这个 FusedMoE 逻辑上是单 rank full-expert MoE，拥有 E experts
  但只有 E/N experts 永久驻留，其余地址由当前双缓冲内容支撑
```

除 `w13_weight/w2_weight` 外，`named_per_expert_tensors()` 会扫描 FusedMoE 直接持有、dim0 等于 local expert 数的 parameter/buffer/tensor，例如 scale、alpha、bias，并在 setup 阶段：

```text
dist.all_gather(local side tensor shards)
torch.cat -> full side tensor
replace_expert_tensor()
```

所以 steady-state forward 不需要为这些小 metadata 做 prefetch；代价是它们全量复制到每个 rank，并且初始化有 collective。

### 11.10 每次 forward 的双缓冲事件协议

`DWDPWeightManager` 创建：

```text
1 个独立 CUDA copy stream
2 个 prefetch_events
2 个 consume_events
```

初始化时先在当前 compute stream 上 record 两个 `consume_event`，表示两个空 slot 都可以被第一次写入。

每个 forward 开始前，`ModelRunner` 调用：

```text
prefetch_first_layers()
  prefetch(first MoE layer)
  prefetch(second MoE layer)
```

`prefetch_layer(layer)` 的顺序：

```text
copy stream wait consume_event[slot]
  # 防止覆盖 compute 仍在读取的旧 layer，解决 WAR hazard

把所有 remote w13/w2 slices 从 peer views copy 到 composite tensor remote region

record prefetch_event[slot] on copy stream
```

进入某个 `FusedMoE` 时：

```text
compute stream wait prefetch_event[slot]
run dispatcher.dispatch
run_moe_core 读取完整逻辑 expert weights
record consume_event[slot] on compute stream
触发 layer + 2 的 prefetch，复用相同 slot
dispatcher.combine
```

`consume_event` 在 `run_moe_core` 之后、`combine` 之前 record 是安全的，因为 combine 消费的是 MoE activation output，不再读取 expert weight。

一个简化 timeline：

```text
forward start:
  copy stream:  prefetch L0(slot0) -> prefetch L1(slot1)

compute stream: wait L0 -> MoE L0 ----------------------->
copy stream:                     wait consume(slot0)
                                 prefetch L2(slot0) ----->

compute stream:        attention L1 -> wait L1 -> MoE L1 ---------->
copy stream:                                      prefetch L3(slot1) ->

compute stream:                  attention L2 -> wait L2 -> MoE L2
```

严格来说，论文写的是“prefetch L+1 与 MoE L + Attention L+1 overlap”；SGLang 的事件实现经过启动灌管后，在完成 MoE L 的 weight consumption 时触发 L+2，给它留下 layer L 的尾部、layer L+1 的后续计算以及 layer L+2 attention 等窗口。二者是同一个双缓冲 pipeline 思想，但触发点和编号表达不同。

### 11.11 SGLang 当前会拉所有 remote experts，不看本 batch 的 top-k 命中

`_prefetch_layer_per_slice()` 遍历：

```text
for weight_name in [w13_weight, w2_weight]:
  for remote region in [before local range, after local range]:
    根据 peer_ranges 切到各 owner
    dst_slice.copy_(peer_view[owner slice])
```

它发生在 router/top-k 之前，因此不知道当前 batch 实际命中了哪些 experts。结果是每个活跃 rank、每个 MoE layer 都会拉取全部缺失主权重：

```text
remote_weight_bytes_per_layer_per_rank
  ~= (N - 1) / N * total_expert_weight_bytes_of_layer
```

好处：

```text
copy plan 静态
大块连续传输
MoE kernel 可以直接访问任意 expert
事件协议简单稳定
```

坏处：

```text
通信量与实际激活 expert 数无关
decode 或小 batch 时极不经济
DWDP group 越大，remote 比例越接近 100%
```

### 11.12 SGLang PR 没有实现论文 v2 的 1MB TDM round-robin copy

这是阅读论文和代码时最容易误判的地方。

论文 full DWDP：

```text
for offset in fixed-size slices, e.g. 1MB:
  for peer in round-robin order:
    enqueue one slice
```

SGLang PR #29778：

```text
for weight name:
  for contiguous remote region:
    找到 owner peer
    对该 peer 的连续 expert range 发起一次 tensor copy_
    然后继续下一个 owner
```

代码里没有：

```text
固定 1MB slice size
按 slice offset 外层遍历
按 peer round-robin 交错 copy
per-destination pending slice queue
```

因此 SGLang 已实现论文的核心异步 P2P pull 和 double buffering，但没有实现论文 4.3 节用于缓解 many-to-one source contention 的 TDM 优化。PR 自己在 4×B200 上的结果不能被解释成“已经包含论文 full DWDP 的全部优化”。

### 11.13 为了真正独立推进，SGLang 还改了 scheduler/communicator

只去掉 MoE A2A 不够。如果 scheduler 仍为了 DP/MLP 同步而 all-gather batch metadata，或者让空闲 rank 构造 idle forward，rank 仍不能真正独立。

SGLang 做了以下配套改动：

```text
LayerCommunicator:
  DWDP sparse MLP 使用 SCATTERED mode
  token 保持在自己的 DP rank，不恢复成 TP full layout

MoE post-processing:
  skip post-experts all-reduce
  因为每个 rank 已独立得到自己 token 的完整 MoE 输出

Scheduler DP attention adapter:
  SCHEDULER_SKIP_ALL_GATHER=true
  DWDP 下即便 dp_size > 1，也不为其他 rank 的工作构造 idle batch

LM head:
  enable_dp_lm_head=true
  每个 rank 对自己的 token 独立完成 logits
```

所以“零跨 rank 同步”是 transport、MoE layout、layer communicator、scheduler、LM head 一起形成的，不是只有 `weight_manager.py` 一处改动。

### 11.14 模型适配：为什么 PR 还改了 GPT-OSS 和 MiMo-V2

#### GPT-OSS

GPT-OSS 的 sparse MoE block 原本按普通/DeepEP 路径分支。DWDP 新增 `forward_dwdp()`：

```text
本 rank 计算 router logits/top-k
直接调用已经绑定 full expert tensor 的 self.experts
不执行 token all-to-all
处理 hidden size padding/unpadding
```

#### MiMo-V2

DWDP 强制 attention data parallel，使 effective attention TP size 可能小于 checkpoint fused-QKV 的 TP interleave size。PR 因此扩展了 MiMo QKV loader：

```text
允许 effective_attn_tp_size 是 checkpoint_tp 的因数
合并多个 checkpoint QKV shards
必要时 de-interleave Q/K/V
对 block-quantized scale_inv 延迟处理
dequantize -> merge/de-interleave -> requantize
```

这部分并不是 DWDP 算法本身，而是“把 MiMo-V2 从 attention TP 切到 attention DP”所需的模型加载兼容。它显著增加了 PR review surface，也说明当前 DWDP 并不能对任意模型无条件开启；模型的 attention/dense 权重加载路径必须支持 DP replication。

### 11.15 论文方案与 SGLang PR 的对照表

| 维度 | 论文 v2 / TensorRT-LLM | SGLang PR #29778 |
|---|---|---|
| 核心语义 | token 留在 rank，remote expert weights 按层拉取 | 相同 |
| attention | 每 rank 全量复制、数据并行 | 强制 DP attention |
| split-weight 消费 | CuTeDSL groupedGEMM 接收 TensorList | CUDA VMM 拼成连续逻辑 tensor，复用 Triton FusedMoE |
| merge D2D copy | 通过 multi-buffer kernel 消除 | 通过 composite VA 消除 |
| remote buffer | layer-wise double buffering | 两套共享 PagePool physical pages |
| copy transport | copy-engine `cudaMemcpyAsync` P2P pull | peer view 到本地 remote slice 的异步 CUDA tensor `copy_` |
| many-to-one 缓解 | 固定小 slice + peer round-robin TDM | 未实现；按 weight/连续 peer range 顺序 copy |
| expert placement | 可不整除，可冗余放置 | 要求专家数整除 `dwdp_size`，连续静态分片 |
| DWDP group | 可作为独立资源配置维度 | 要求 `dwdp_size == tp_size` |
| kernel | 论文修改 groupedGEMM | 主 MoE kernel 基本不变，只插 wait/record hook |
| 初始化 collective | 不是论文重点 | handle metadata exchange、side tensor all-gather、barrier |
| 验证模型/硬件 | DeepSeek-R1 NVFP4 / GB200 NVL72 | GPT-OSS-120B、MiMo-V2.5 / B200 |

这张表可以形成一个非常重要的判断：

```text
SGLang 不是逐行复刻论文 TensorRT-LLM 实现，
而是复用论文的并行范式，再用更适合现有 SGLang Triton MoE 的 VMM 方案落地。
```

### 11.16 SGLang 实现的显存模型

记：

```text
W_dense:
  attention、dense/shared、embedding/head 等需要复制的权重

W_moe_total:
  所有层 routed expert 主权重总大小

R_l:
  第 l 个 MoE layer 在本 rank 上缺失的 remote 主权重大小

S_side:
  全量复制的 expert scale/bias/alpha 等 side tensors
```

忽略页对齐和 workspace 后，每 rank 的 persistent/temporary 权重显存近似：

```text
M_rank
  ~= W_dense
   + W_moe_total / N
   + max(R_l assigned to slot0)
   + max(R_l assigned to slot1)
   + S_side
```

不是：

```text
W_dense + W_moe_total
```

也不完全等于简单的：

```text
W_dense + W_moe_total/N + 2 * average(R_l)
```

因为 PagePool 分别按两个 slot 中最大的 remote layout 分配，并且还存在：

```text
VMM allocation granularity
PagePool 8 * granularity 的 page size
每个 weight pre/post region 的 alignment/padding
local handle 首尾 edge bytes
peer VA reservations 本身
activation/KV/workspace
```

VA reservation 消耗地址空间但不等价于同量物理显存；真正主要的 temporary physical memory 是两套 PagePool pages。

### 11.17 性能结果不要把论文和 PR 混用

论文结果：

```text
DeepSeek-R1 NVFP4
GB200 NVL72
TensorRT-LLM
context-only + end-to-end PD
包含 TensorList merge elimination；部分实验包含 1MB TDM
```

SGLang PR 描述中的结果：

```text
4 × B200
GPT-OSS-120B
prefill-only
DWDP4 vs DEP4

MNT=16K/32K、ISL=4K~32K:
  报告约 1.16x~1.92x

bench_serving saturation, concurrency=128, ISL=8K:
  DWDP4 506K tok/s
  DEP4  329K tok/s
  约 1.54x
```

两组数字不能直接横向比较，因为：

```text
模型不同
硬件不同
baseline backend/配置可能不同
论文包含端到端 generation 资源配比和 TTFT
PR 结果重点是 SGLang prefill throughput
```

### 11.18 当前 SGLang v1 的明确边界和代码风险

功能边界：

```text
只支持 CUDA/NVIDIA VMM
只支持识别到的 Triton FusedMoE
只处理 w13_weight/w2_weight 主权重
要求所有 MoE 层 routed expert 数一致
要求 expert 数整除 group size
要求 dwdp_size == tp_size
不支持 PP、EPLB、speculative decoding、TBO、CUDA graph
推荐仅用于 PD prefill
```

测试边界：

```text
PR 主要是两条昂贵的 B200 PD e2e accuracy 测试
缺少 layout/page-edge 单元测试
缺少事件协议和重复 setup/cleanup 测试
缺少 DEP vs DWDP logits/token-level 等价测试
缺少 standalone decode 性能/正确性覆盖
非 CUDA import 回归在合入后才通过 lazy import 修复
```

资源生命周期上还有一个值得继续核查的点：本地 expert allocation handle 在创建时以 retained handle 保存到 `MnnvlHandleSet`，但当前 `DWDPTransport.release()` 只显式释放 imported peer handles，没有遍历释放本地 `_handle_set.handles`。普通 server 进程退出时 CUDA context 会整体回收，因此不容易暴露；进程内 teardown/re-init 或未来模型热重载时可能形成 VMM physical allocation 泄漏。

初始化异常路径也缺少完整事务式 rollback：handle 创建、fd exchange、peer import、VA mapping 任一步失败，都可能留下 fd/handle/VA，并让其他 rank 停在 collective/barrier。这不否定核心设计，但说明当前实现更接近面向固定部署的 v1，而不是已经覆盖复杂恢复场景的通用 runtime。

## 12. 工程实现需要哪些模块

如果要在一个推理框架里实现 DWDP，至少需要这些组件。

### 12.1 权重 placement 和 metadata

需要描述：

```text
DWDP group
rank -> local expert list
logical expert -> owner rank(s)
每层 expert weight 的 local pointer
每层 remote expert 的 peer pointer
量化 scale / metadata 的 peer pointer
冗余 expert placement
```

DWDP 的一个优点是 placement 约束比 EP 弱：

```text
expert 数不一定要被 DWDP group size 整除
可以允许少量 redundant placement
可以按单 rank 粒度扩缩 context server
```

### 12.2 异步预取 runtime

需要管理：

```text
prefetch stream
compute stream
cuda events
double buffers
per-layer prefetch plan
remote pointer validity
buffer lifetime
```

执行逻辑大致是：

```text
for layer in layers:
  wait prefetch(layer) if needed
  launch attention / MoE compute
  issue prefetch(layer + 1) early enough
  release old remote buffer after MoE consumes it
```

真正难点是：

```text
不同 rank 进度不同
不能假设所有 rank 在同一层
不能用 collective event 作为同步
每个 rank 的 prefetch plan 必须自洽
```

### 12.3 split-weight 消费机制

系统必须让 MoE compute 在不做额外 D2D merge 的情况下访问 local/remote split weights。可以选择两类机制。

论文的 multi-buffer kernel 至少要支持：

```text
多个 expert weight buffers
expert id 到 buffer pointer 的映射
local/remote expert 混排
量化格式 metadata
不引入明显 address calculation overhead
```

SGLang 的 VMM 路线则需要支持：

```text
shareable CUDA physical allocation
page-aligned composite VA layout
local handle 与双缓冲 page pool 的混合映射
从 raw pointer 构造正确 dtype/shape 的 torch tensor
页边界 edge bytes 修复
handle、VA、tensor view 的生命周期管理
```

如果两类机制都不具备，runtime 被迫 merge：

```text
D2D merge copy
  -> HBM 带宽浪费
  -> critical path 增加
  -> DWDP 收益下降
```

### 12.4 copy scheduling

需要避免：

```text
多个 destination 同时从同一个 source 拉大块权重
source-side copy engine 串行化
通信窗口被拉长
compute bubble 暴露
```

可能策略：

```text
slice-based TDM
round-robin peers
按 source rank 建队列
限制同一 source 的大块并发
根据 layer compute window 调 slice size
```

slice size 需要调：

```text
太大:
  争用和 bubble 更明显

太小:
  DMA request 过碎
  launch/scheduling overhead 增加
  power interference 可能加剧
```

### 12.5 性能监控

DWDP 调优不能只看 TPS，需要同时看：

```text
P2P copy duration
copy overlap ratio
prefetch wait time
copy engine utilization
NVLink throughput
HBM bandwidth
GPU frequency
power cap / TGP
attention kernel time
MoE groupedGEMM time
queueing time
TTFT
TPS/user
TPS/GPU
```

特别是 Appendix A 说明：

```text
如果 attention 变慢，不要只怀疑 NVLink 或 HBM。
GPU frequency / power throttle 可能才是主因。
```

## 13. 论文亮点

### 13.1 抓住了 serving 中的真实不均衡

很多并行论文默认各 rank workload 平衡，但线上 serving 里：

```text
prompt length 不同
KV cache hit rate 不同
expert routing 不同
batch 内请求动态进出
PD context/generation rate matching 不稳定
```

DWDP 把“同步放大局部不均衡”作为核心问题，这是很实际的系统视角。

### 13.2 把 MoE 权重搬运变成可 overlap 的 runtime 问题

传统直觉认为：

```text
MoE expert 权重大，搬权重一定很贵
```

DWDP 的反驳是：

```text
在 NVL72 这种高带宽域内，
context 阶段每层 compute window 足够大时，
远端 expert 权重预取可以被隐藏。
```

这不是普适结论，但在 GB200 NVL72 + 大 MoE context serving 上是有工程意义的。

### 13.3 没停留在概念，分析了实际 overhead

论文没有只报一个高层 speedup，而是拆了几个很真实的工程问题：

```text
split-weight merge D2D copy
many-to-one source contention
copy 与 compute 的 memory/power interference
TTFT 变差
context GPU 缩容后的 queueing
```

这些部分比“提出一种新并行策略”本身更有参考价值。

## 14. 论文不足和需要谨慎的地方

### 14.1 硬件依赖很强

DWDP 的前提是高带宽 peer GPU 通信。论文结论不能直接外推到：

```text
PCIe-only 多卡
跨节点 IB
普通以太网络
非全互联 NVLink 拓扑
```

如果远端权重预取不能稳定隐藏，DWDP 会变成“每层拉权重”的高开销策略。

### 14.2 端到端收益伴随 TTFT 代价

论文摘要强调 8.8% output TPS/GPU 提升，但 Table 6 显示 TTFT 在多个区间变差明显。

所以更准确的表述是：

```text
DWDP 改善了 serving efficiency frontier 的一部分，
尤其是 context GPU 效率，
但不是无条件降低用户延迟。
```

如果业务强约束 TTFT，不能只按 TPS/GPU 选点。

### 14.3 主要验证在 DeepSeek-R1 + NVFP4 + TensorRT-LLM + GB200 NVL72

论文没有充分证明：

```text
不同 MoE 架构
不同 expert 数 / top-k
不同量化格式
不同上下文分布
不同调度器
不同 GPU 代际
```

下也能得到相同收益。

### 14.4 DWDP group size 扩大后的复杂度

论文分析了更大 group 的 contention 概率，但主要端到端收益并不等于 group 越大越好。

group size 扩大意味着：

```text
每 rank local expert 更少
remote expert 更多
P2P peer 更多
copy plan 更复杂
source contention 概率上升
buffer 和 metadata 管理更复杂
```

实际可用 group size 需要结合模型权重、GPU 内存、互联拓扑、MNT、TTFT SLO 一起搜索。

### 14.5 显存峰值和 OOM 边界没有系统展开

DWDP 的显存安全性依赖一个隐含条件：

```text
单卡能放下:
  全量 attention/dense weights
  + 所有层的 local expert weights
  + 当前层 remote expert buffer
  + 下一层 prefetch buffer
  + activation / KV / workspace / communication buffers
```

论文通过 layer-wise prefetch/release 避免了“所有层 remote experts 一次性常驻”，但并没有给出通用显存上界公式，也没有系统扫描不同 group size、不同精度、不同 batch/MNT 下的 OOM 边界。

这一点对落地很关键。DWDP 的可行性不能只看通信是否能隐藏，还要先验证：

```text
remote expert 临时展开是否能放下
double buffering 是否能放下
prefill activation/workspace 是否还留有余量
```

如果这三点不成立，DWDP 可能还没进入性能收益区间就先遇到显存瓶颈。

## 15. 一个简化 mental model

可以用下面的模型快速判断 DWDP：

```text
传统 EP/DEP:
  每层大家集合:
    谁慢等谁
    token 来回跑

DWDP:
  大家各跑各的:
    token 不动
    权重提前搬过来
```

收益来自：

```text
少等人
少做 collective
context 阶段有时间偷偷搬权重
```

风险来自：

```text
搬权重没藏住
搬权重影响计算
copy engine / NVLink / HBM / power 被打满
为了省 GPU 缩 context server 后排队变长
```

## 16. 对后续学习和实现的建议

如果继续研究 DWDP，建议按这个顺序深入：

```text
1. 先复现实验里的 profiling breakdown
   看 DEP 中 sync cost / all-to-all cost 到底占多少

2. 再看目标硬件上的 P2P bandwidth 和 copy engine 行为
   重点不是 peak bandwidth，而是 overlap compute 时的 effective bandwidth

3. 单独验证 split-weight 消费路径
   TensorList 路线要验证 groupedGEMM 多 buffer 的索引成本
   VMM 路线要验证页对齐、composite VA、edge bytes 和 handle 生命周期
   两条路线都要确认没有重新引入 D2D merge copy

4. 做 prefetch wait time trace
   看 remote weight copy 是否真的隐藏在 compute window 后面

5. 做 power/frequency profiling
   attention slowdown 可能来自 DVFS，而不是传统 bandwidth bottleneck

6. 放到 PD 端到端系统中重新找 Pareto
   不能只优化 context-only TPS/GPU
   必须同时看 TTFT、TPS/user、output TPS/GPU、context GPU 数量
```

## 17. 最终评价

DWDP 的价值不在于它“替代 EP”，而在于它指出了一个在 NVL72 这类新硬件上变得可行的方向：

```text
当 GPU 间带宽足够高、模型是大 MoE、prefill compute window 足够大时，
把 expert 权重按需拉到本地，
可能比每层同步 dispatch token 更高效。
```

它本质上是一次系统设计取舍：

```text
用更多 runtime/kernel/communication 复杂度
换取去 collective、去全局同步、提升 context GPU efficiency。
```

对 SGLang 这类服务系统来说，最值得吸收的是三个思想：

```text
1. 不均衡的真正成本不是局部慢，而是 collective 把局部慢变成全局等。

2. 在 PD 分离下，context server 和 decode server 可以使用完全不同的并行策略；
   context 阶段可以尝试更激进的异步权重预取，decode 阶段继续走低延迟 token dispatch。

3. MoE 系统优化不能只做调度或只做 kernel；
   真正有效的方案需要 placement、runtime prefetch、copy scheduling、kernel layout、power profiling 一起设计。
```

如果用一句工程化的判断收尾：

```text
DWDP 适合拿来优化“高带宽单机/单域 MoE prefill 集群的 GPU 效率”，
不适合直接当成“所有 MoE 推理都该换的默认并行策略”。
```

对当前 SGLang 实现再补一句：

```text
PR #29778 已经验证了“VMM composite VA + 双缓冲 P2P pull”可以在不重写
Triton MoE kernel 的情况下落地 DWDP；但它仍是范围受限的 v1，尚未包含论文
的灵活/冗余 expert placement 和 1MB TDM contention mitigation。
```

## 18. 后续提问索引与资料

后续可以按下面的主题直接提问：

| 想问的问题 | 建议先看 |
|---|---|
| 为什么 DEP 的不均衡会变成全局等待 | 第 1、3 节 |
| DWDP 什么时候比 DEP 快 | 第 4、8、9 节 |
| 论文 TensorList groupedGEMM 怎么消除 merge | 第 5.1、5.2 节 |
| SGLang 为什么不需要改 groupedGEMM | 第 5.3、11.7、11.9 节 |
| CUDA VMM、physical handle、VA mapping 的关系 | 第 11.5～11.7 节 |
| PagePool 为什么只要两个 slot | 第 11.8、11.10 节 |
| CUDA event 怎样避免 buffer 被提前覆盖 | 第 11.10 节 |
| SGLang 是否按 top-k 只拉命中 expert | 第 11.11 节 |
| 论文的 TDM 在 SGLang 中实现了吗 | 第 6.2、11.12 节 |
| 为什么必须放在 prefill/context | 第 2.3、4、9、10.2 节 |
| 显存怎么估算 | 第 2.1、11.16、14.5 节 |
| 当前代码限制和潜在问题 | 第 11.18 节 |

主要资料：

- 论文 v2：[arXiv 2604.01621v2](https://arxiv.org/abs/2604.01621v2)
- 论文 HTML 全文：[DWDP HTML](https://arxiv.org/html/2604.01621v2)
- SGLang 实现：[PR #29778](https://github.com/sgl-project/sglang/pull/29778)
- SGLang 合入提交：`37a830b667098b41f355dfde58518154971efb32`
- 非 CUDA lazy-import 修复：[PR #31919](https://github.com/sgl-project/sglang/pull/31919)
- 论文对应 TensorRT-LLM 实现：[TensorRT-LLM PR #12136](https://github.com/NVIDIA/TensorRT-LLM/pull/12136)

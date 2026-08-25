# 06. 业务配置拆解：64 卡 engine，DPA8 + DeepEP64 的 SGLang 全流程

本文只分析你给的这组真实业务配置。

前面几篇文档讲的是通用概念：

```text
02:
  TP

03:
  EP none backend

04:
  DeepEP backend

05:
  DPA + EP 组合关系
```

这一篇不再泛讲所有组合，而是把一个具体配置从启动参数一路追到源码执行路径。

## 0. 业务参数

你给的参数是：

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

这里有一层上层 rollout 框架参数到 SGLang server 参数的映射。本文按下面这个假设来解释：

```text
--rollout-num-gpus-per-engine 64
  表示一个 SGLang engine 使用 64 张 GPU
  等价理解成 SGLang 内部 tp_size = 64

--sglang-ep-size 64
  对应 SGLang --ep-size 64

--sglang-dp-size 8
  对应 SGLang --dp-size 8

--sglang-enable-dp-attention
  对应 SGLang --enable-dp-attention

--sglang-moe-a2a-backend deepep
  对应 SGLang --moe-a2a-backend deepep

--sglang-deepep-mode auto
  对应 SGLang --deepep-mode auto

--sglang-moe-dense-tp-size 1
  对应 SGLang --moe-dense-tp-size 1

--sglang-enable-dp-lm-head
  对应 SGLang --enable-dp-lm-head
```

所以本文分析的 SGLang 侧核心配置是：

```text
tp_size = 64
dp_size = 8
enable_dp_attention = true
ep_size = 64
moe_a2a_backend = deepep
deepep_mode = auto
moe_dense_tp_size = 1
enable_dp_lm_head = true
pp_size = 1
```

`--sglang-mem-fraction-static 0.7` 是内存比例参数，会影响 KV cache / 静态内存预算，不改变本文讨论的并行拓扑。

## 1. 最终并行形态

先直接给结果。

这不是：

```text
64 个请求 DP 流
```

而是：

```text
64 张 GPU 组成一个 SGLang engine
8 个 attention DP 请求流
每个 attention DP 流内部有 TP8
MoE 使用 DeepEP64
```

代入公式：

```text
attn_dp_size = dp_size = 8
attn_tp_size = tp_size // dp_size = 64 // 8 = 8

DeepEP 强制:
  moe_ep_size = ep_size = tp_size = 64

默认:
  moe_dp_size = 1

所以:
  moe_tp_size = tp_size // moe_ep_size // moe_dp_size
              = 64 // 64 // 1
              = 1
```

最终：

```text
attention:
  DPA8
  每个 DP 内 attention TP8

MoE:
  DeepEP64
  moe_tp_size = 1

请求路由:
  routed_dp_rank = 0..7
```

特别注意：

```text
routed_dp_rank 不是 0..63
```

因为请求级 DP 是：

```text
attn_dp_size = dp_size = 8
```

## 2. rank 分组

### 2.1 全局 64-rank group，不是 attention TP64

这个配置里最容易误解的是 `tp_size=64`。

它不是说 attention 一定用 TP64。打开 DPA 后，attention 真正使用的是：

```text
attn_tp_size = tp_size // dp_size = 64 // 8 = 8
```

但 SGLang 仍然会用 `tp_size=64` 建一个全局 64-rank model-parallel group。源码里这个 group 仍然叫 TP group：

```text
TP group = [0, 1, 2, ..., 63]
```

更准确的理解是：

```text
TP group / get_tp_group():
  64-rank 的全局通信 root group
  DeepEP 会使用它
  不等于 attention TP64

ATTN_TP group:
  attention 真正用的 TP group
  本配置下 size = 8
```

由于 `pp_size=1`，分布式 world size 基本就是：

```text
world_size = tp_size * pp_size = 64
```

如果是多机，例如 8 机 x 8 卡，仍然是同一个 64-rank world，只是每台机器启动自己负责的 local ranks。

### 2.2 attention DP / TP group

源码里 DPA 的 rank 公式在：

```text
python/sglang/srt/layers/dp_attention.py
  compute_dp_attention_world_info()
```

核心公式：

```python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

本配置：

```text
tp_size = 64
dp_size = 8
attn_cp_size = 1
```

所以：

```text
attn_dp_size = 8
attn_tp_size = 8
attn_dp_rank = tp_rank // 8
attn_tp_rank = tp_rank % 8
```

因此 attention 分组是：

```text
attention DP0:
  ranks [0..7]
  attn_tp_rank = 0..7

attention DP1:
  ranks [8..15]
  attn_tp_rank = 0..7

attention DP2:
  ranks [16..23]
  attn_tp_rank = 0..7

attention DP3:
  ranks [24..31]
  attn_tp_rank = 0..7

attention DP4:
  ranks [32..39]
  attn_tp_rank = 0..7

attention DP5:
  ranks [40..47]
  attn_tp_rank = 0..7

attention DP6:
  ranks [48..55]
  attn_tp_rank = 0..7

attention DP7:
  ranks [56..63]
  attn_tp_rank = 0..7
```

这意味着：

```text
一个请求被路由到 DP3:
  它的 attention 计算发生在 ranks [24..31] 这 8 张卡上

一个请求被路由到 DP7:
  它的 attention 计算发生在 ranks [56..63] 这 8 张卡上
```

### 2.3 MoE EP group

DeepEP 在参数处理阶段会强制：

```text
ep_size = tp_size
```

源码位置：

```text
python/sglang/srt/server_args.py
  _handle_a2a_moe()
```

逻辑是：

```python
if self.moe_a2a_backend == "deepep":
    self.ep_size = self.tp_size
```

本配置本来就写了：

```text
ep_size = 64
tp_size = 64
```

所以不会改变数值。

在：

```text
python/sglang/srt/distributed/parallel_state.py
  initialize_model_parallel()
```

里有：

```text
moe_ep_size = expert_model_parallel_size
moe_dp_size = moe_data_model_parallel_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

代入：

```text
moe_ep_size = 64
moe_dp_size = 1
moe_tp_size = 64 // 64 // 1 = 1
```

因为：

```text
moe_ep_size == tensor_model_parallel_size
```

所以 MoE EP group 就是整个 64-rank global TP/root group：

```text
MoE EP group = [0, 1, 2, ..., 63]
```

也就是说：

```text
attention:
  一个请求只在某个 DP group 的 8 张卡上做 attention

MoE:
  router/topk 后，token-expert 任务可以通过 DeepEP 发到 64 张卡中的 expert owner rank
```

这就是这个配置的核心。

## 3. 启动阶段源码流程

### 3.1 HTTP server 和 engine

启动入口在：

```text
python/sglang/srt/entrypoints/http_server.py
  launch_server()
```

注释里写得很清楚，SRT server 包含：

```text
HTTP server
TokenizerManager
Scheduler subprocess
DetokenizerManager
```

对于普通非 DP 情况，Engine 会直接启动 scheduler processes。对于本文配置：

```text
dp_size = 8 > 1
```

所以会进入 DP controller 路线。

源码位置：

```text
python/sglang/srt/entrypoints/engine.py
  Engine._launch_scheduler_processes()
```

关键分支：

```python
if server_args.dp_size == 1:
    # Launch tensor parallel scheduler processes
else:
    # Launch the data parallel controller
    proc = mp.Process(
        target=run_data_parallel_controller_process,
        ...
    )
```

本文配置会走：

```text
启动一个 DataParallelController 进程
```

然后由 DataParallelController 再启动实际的 scheduler worker 进程。

### 3.2 DataParallelController 选择 DPA 路线

源码位置：

```text
python/sglang/srt/managers/data_parallel_controller.py
  DataParallelController.__init__()
```

关键逻辑：

```python
if server_args.enable_dp_attention:
    self.launch_dp_attention_schedulers(server_args, port_args)
else:
    self.launch_dp_schedulers(server_args, port_args)
```

本文配置：

```text
enable_dp_attention = true
```

所以走：

```text
launch_dp_attention_schedulers()
```

这点非常重要。

如果没有 `--enable-dp-attention`，`dp_size=8` 会变成：

```text
8 个完整 DeepEP64 副本
需要 64 * 8 = 512 张 GPU
```

在你这个：

```text
--rollout-num-gpus-per-engine 64
```

的语境下，这显然不是想要的资源模型；要让 64 张卡内部形成 8 个请求级 DP 流，就必须开启 DPA。

但现在开启 DPA 后，`dp_size=8` 是在一个 64-rank global TP/root group 里切出 8 个 attention DP groups；每个 attention DP group 内部才是 attention TP8。

### 3.3 DPA scheduler 进程如何启动

源码位置：

```text
python/sglang/srt/managers/data_parallel_controller.py
  launch_dp_attention_schedulers()
  launch_tensor_parallel_group()
```

`launch_dp_attention_schedulers()` 做两件事：

```text
1. node0 上给 dp_rank=0..7 预分配 worker ports
2. 调用 launch_tensor_parallel_group() 启动 64 个 TP ranks
```

在 `launch_tensor_parallel_group()` 里，每个 `tp_rank` 都会重新计算自己的 `dp_rank`：

```python
_, _, dp_rank, _ = compute_dp_attention_world_info(
    server_args.enable_dp_attention,
    tp_rank,
    server_args.tp_size,
    server_args.dp_size,
    server_args.attn_cp_size,
)
```

代入本文配置：

```text
tp_rank 0..7:
  dp_rank = 0

tp_rank 8..15:
  dp_rank = 1

...

tp_rank 56..63:
  dp_rank = 7
```

然后每个 rank 会启动一个 scheduler process：

```python
proc = mp.Process(
    target=self.run_scheduler_process_func,
    args=(
        server_args,
        rank_port_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        writer,
    ),
)
```

更贴近真实部署的情况是 8 节点 x 8 卡。可以理解成：

```text
node 0:
  local GPUs 0..7
  global tp_rank [0..7]
  attention DP0

node 1:
  local GPUs 0..7
  global tp_rank [8..15]
  attention DP1

node 2:
  local GPUs 0..7
  global tp_rank [16..23]
  attention DP2

...

node 7:
  local GPUs 0..7
  global tp_rank [56..63]
  attention DP7
```

整个 8 节点共同组成一个：

```text
world_size = 64
global TP/root group = [0..63]
MoE EP group = [0..63]
```

node 0 通常还会承担对外 HTTP server / TokenizerManager / DataParallelController 的入口角色；其它 node 主要启动自己负责的 scheduler worker ranks，并加入同一个分布式 world。

### 3.4 进程到底跑在哪个节点

这里单独展开，因为它对理解服务形态很重要。

先给结论：

```text
HTTP server:
  只在 node0 暴露业务 URL

TokenizerManager:
  只在 node0

DetokenizerManager:
  只在 node0

DataParallelController:
  每个 node 都会启动 1 个进程
  但只有 node0 的 DataParallelController 进入 event_loop() 做请求分发

Scheduler worker:
  每个 node 启动 8 个
  8 个节点总共 64 个
```

这里的 scheduler worker 是一个进程，不是 scheduler 进程再加一个 model worker 进程。

一个 scheduler worker 进程内部大致是：

```text
run_scheduler_process()
  -> Scheduler
  -> TpModelWorker
  -> ModelRunner
```

也就是说：

```text
Scheduler:
  负责接请求、组 batch、调度 decode/prefill

TpModelWorker:
  Scheduler 内部持有的对象

ModelRunner:
  TpModelWorker 内部持有的对象
  真正执行 model forward
```

它们在同一个 scheduler worker 进程里。

源码位置：

```text
python/sglang/srt/entrypoints/engine.py
  Engine._launch_scheduler_processes()

python/sglang/srt/managers/data_parallel_controller.py
  run_data_parallel_controller_process()
  DataParallelController.__init__()
```

为什么 DataParallelController 每个节点都有一个？

因为每个节点都要启动自己本地的 scheduler worker ranks：

```text
node 0:
  启动 tp_rank [0..7]

node 1:
  启动 tp_rank [8..15]

...

node 7:
  启动 tp_rank [56..63]
```

这些本地 worker 是由本节点的 DataParallelController 拉起来的。

但是请求分发只在 node0 的 DataParallelController 上发生。源码里：

```python
if server_args.node_rank == 0:
    controller.event_loop()
```

也就是说：

```text
node0 DataParallelController:
  1. 启动 node0 本地 8 个 scheduler worker
  2. 接收 TokenizerManager 发来的请求
  3. 根据 routed_dp_rank / load balance 分发到 DP worker

node1..node7 DataParallelController:
  1. 启动本节点本地 8 个 scheduler worker
  2. 不接收 HTTP / TokenizerManager 请求
  3. 不进入 controller.event_loop()
  4. 等待本节点 worker 进程
```

非 node0 的主进程也不会启动 TokenizerManager / DetokenizerManager。源码里：

```text
if server_args.node_rank >= 1:
  non-zero rank nodes do not need to run tokenizer or detokenizer
```

所以 8 节点 x 8 卡时，可以粗略记成：

| 节点 | HTTP server | TokenizerManager | DetokenizerManager | DataParallelController | Scheduler worker |
| --- | --- | --- | --- | --- | --- |
| node0 | 1 | 1 | 1 | 1，负责 dispatch | 8 个，tp_rank 0..7 |
| node1 | dummy health check | 0 | 0 | 1，只负责拉起本地 worker | 8 个，tp_rank 8..15 |
| node2 | dummy health check | 0 | 0 | 1，只负责拉起本地 worker | 8 个，tp_rank 16..23 |
| ... | ... | ... | ... | ... | ... |
| node7 | dummy health check | 0 | 0 | 1，只负责拉起本地 worker | 8 个，tp_rank 56..63 |

因此总进程数量可以近似理解为：

```text
node0:
  1 main/http/tokenizer process
  1 detokenizer process
  1 DataParallelController process
  8 scheduler worker processes

node1..node7:
  每个 node:
    1 main/dummy health process
    1 DataParallelController process
    8 scheduler worker processes
```

整个 engine 合计：

```text
DataParallelController:
  8 个进程
  只有 node0 的 controller 做请求 dispatch

Scheduler worker:
  64 个进程

对外业务 HTTP URL:
  1 个，通常是 node0
```

注意这里说的 `DataParallelController` 数量和“请求级 DP 数量”不是一回事：

```text
DataParallelController 进程数:
  8 个，按节点算

请求级 DP rank:
  8 个，按 dp_size 算，routed_dp_rank = 0..7
```

这两个数在 8 节点 x 8 卡例子里刚好都是 8，但含义完全不同。

### 3.5 为什么一个节点内 TP8 需要 8 个 scheduler worker

这里要区分两个概念：

```text
请求入口:
  一个 attention DP group 里通常只有一个 rank 从 socket 拉请求

参与计算:
  attention TP8 的 8 个 ranks 都要参与 model forward
```

所以在一个节点里：

```text
attention DP0 = ranks [0..7]
```

虽然业务请求看起来是“一路”进入 DP0，但这一路请求不是只由一个进程完成 forward。实际是：

```text
rank 0:
  attn_tp_rank = 0
  从 DataParallelController 对应的 socket 拉请求
  作为这个 attention TP group 的请求入口 rank

rank 1..7:
  attn_tp_rank = 1..7
  不直接从 tokenizer/controller socket 拉请求
  通过 broadcast 收到同一批 work requests
  和 rank0 一起参与 TP8 forward
```

源码位置：

```text
python/sglang/srt/managers/scheduler_components/ipc_channels.py
  SchedulerIpcChannels.create()

python/sglang/srt/managers/scheduler_components/request_receiver.py
  _pull_raw_reqs()
  _broadcast_reqs_across_ranks()
```

`SchedulerIpcChannels.create()` 里只有 `is_rank_zero=True` 的 scheduler worker 会创建：

```text
recv_from_tokenizer
recv_from_rpc
```

其它 rank：

```text
recv_from_tokenizer = None
recv_from_rpc = None
```

在 `request_receiver._pull_raw_reqs()` 里，也只有：

```text
attn_tp_rank == 0
attn_cp_rank == 0
```

的 rank 会执行：

```python
recv_req = self.recv_from_tokenizer.recv_pyobj(...)
```

然后在 `_broadcast_reqs_across_ranks()` 里，如果：

```text
enable_dp_attention = true
attn_tp_size = 8
```

会把 work requests 广播到同一个 attention TP group：

```python
work_reqs = broadcast_pyobj(
    work_reqs,
    self.attn_tp_group.rank,
    self.attn_tp_cpu_group,
    src=self.attn_tp_group.ranks[0],
)
```

因此：

```text
控制流入口:
  每个 DP group 一个入口 rank

实际计算:
  每个 DP group 里 8 个 scheduler worker / TP ranks 一起算
```

这就是为什么：

```text
一个节点内 attention TP8
  需要 8 个 scheduler worker 进程

但请求入口看起来只有一路
  因为只有 attn_tp_rank=0 直接从 socket 收请求
```

可以把一个 node 上的 DP0 画成：

```text
DataParallelController
  |
  v
rank0 scheduler worker
  |
  | broadcast work requests inside ATTN_TP group
  v
rank0, rank1, rank2, ..., rank7
  |
  | TP8 model forward
  v
output
```

所以更准确的说法是：

```text
每个 GPU rank 一个 scheduler worker 进程。
同一个 attention DP group 内，只有 leader rank 收请求；
但所有 TP ranks 都要参与计算。
```

这里再补一个关键理解：

```text
8 个 scheduler worker 不是 8 路独立调度。
它们属于同一个 attention TP group，会对同一批 work requests 做一致的调度和 batch 构造。
```

也就是说，rank0 从 socket 拉到请求后，广播给 rank1..rank7。随后这 8 个 scheduler worker 都会看到同一批请求，并维护同一批请求的调度状态：

```text
同一批 requests
同一批 prefill / decode 决策
同一批 batch 逻辑
同一批 req ids / token ids / positions 语义
```

但这不等于 8 个进程每一步 side effect 都完全一样。

不同点包括：

```text
1. 只有 attn_tp_rank=0 直接从 socket 拉请求
2. 只有部分 leader rank 负责把输出发回 tokenizer / detokenizer
3. 每个 rank 的模型权重是 TP shard
4. 每个 rank 的 attention heads / logits / hidden 中间结果可能是局部分片
5. TP collective 后才形成完整语义上的输出
```

所以更精确地说：

```text
调度语义:
  一致

batch 语义:
  一致

模型逻辑输入:
  同一批 token / position / request metadata

本地张量和本地输出:
  不一定完全一样，因为 TP rank 持有不同权重分片和不同 head/logit 分片

最终采样和返回:
  通过 TP 通信 / gather / leader output 路线得到一致的用户可见结果
```

因此你说的“数据实际上就一路”可以理解为：

```text
请求入口一路
batch 语义一路
但计算是 TP8 的 8 路分片协同
```

### 3.6 它怎么保证 8 个 scheduler worker 不跑偏

不是简单地说：

```text
广播后输入一样，所以后面自然一定一样
```

更准确是 SGLang 用 SPMD 方式约束它们：

```text
1. leader rank 收请求
2. work requests 广播到同一个 attention TP group
3. 8 个 scheduler worker 维护相同的请求队列 / running batch 逻辑
4. batch 构造是相同状态 + 相同输入上的确定性决策
5. model forward 中的 TP collective 要求各 rank 按相同顺序进入相同通信
6. 输出只由 leader / 对应输出路径发回上游
```

所以一致性主要来自：

```text
相同输入流
相同初始状态
确定性的 scheduler 状态机
TP collective 的同步约束
leader-only I/O 分工
```

如果某个 rank 因为 bug 或非确定性走了不同 batch：

```text
1. collective 的 shape / 次序可能对不上
2. 轻则报错
3. 重则 hang
```

所以 TP 场景下，scheduler 不能让 rank0 decode，而 rank1 prefill，也不能让 rank0 batch 有 10 个 token，rank1 batch 有 9 个 token。它们必须在同一个 batch 语义上前进。

源码里请求一致性的第一层保障在：

```text
python/sglang/srt/managers/scheduler_components/request_receiver.py
  _pull_raw_reqs()
  _broadcast_reqs_across_ranks()
```

`_pull_raw_reqs()` 只有 leader rank 拉请求：

```text
attn_tp_rank == 0
attn_cp_rank == 0
```

`_broadcast_reqs_across_ranks()` 再把 `work_reqs` 广播到 attention TP group。

第二层保障是 scheduler 本身是 SPMD 状态机。每个 rank 都有自己的 `Scheduler` 对象，但它们看到同一批请求，并且按同样的规则构造 batch。

第三层是模型 forward 的 collective。比如 attention TP / MoE / logits 相关 collective 都要求各 rank 同步进入。如果某个 rank 的 batch 不一致，通常很快会在 collective 阶段暴露。

最后还有采样 token 的细节。

源码位置：

```text
python/sglang/srt/layers/sampler.py
  _sync_token_ids_across_tp()
```

里面有注释说明：

```text
默认情况下，为了性能，SGLang 不一定同步 final token ids across TP ranks。
它依赖最后的 all-reduce、lm_head matmul、sampling kernels 的确定性。
```

如果出现极少数非确定性导致 TP ranks token id 不一致，可能会导致后续 ranks 状态不同步。SGLang 提供了环境变量：

```bash
SYNC_TOKEN_IDS_ACROSS_TP=1
```

开启后会对 `batch_next_token_ids` 做一次 TP all-reduce 同步。代码里 grammar 场景也会触发同步，因为 grammar 约束下更容易需要严格一致。

所以可以这样记：

```text
正常路径:
  依赖确定性和 collective 约束，避免额外同步，性能更好

更保守路径:
  设置 SYNC_TOKEN_IDS_ACROSS_TP=1
  用一次额外同步换更强的一致性保障
```

### 3.7 每个 rank 的 moe_ep_rank

同一个函数里还会计算：

```python
moe_ep_rank = (
    tp_rank
    % (server_args.tp_size // server_args.moe_dp_size)
    // (
        server_args.tp_size
        // server_args.moe_dp_size
        // server_args.ep_size
    )
)
```

本文配置：

```text
tp_size = 64
moe_dp_size = 1
ep_size = 64
```

所以：

```text
server_args.tp_size // server_args.moe_dp_size // server_args.ep_size
= 64 // 1 // 64
= 1

moe_ep_rank = tp_rank
```

也就是：

```text
rank 0:
  moe_ep_rank = 0

rank 1:
  moe_ep_rank = 1

...

rank 63:
  moe_ep_rank = 63
```

## 4. 每个 worker 初始化分布式环境

每个 scheduler worker 进程启动后，会在进程内部创建 `Scheduler`，`Scheduler` 再创建 `TpModelWorker`，`TpModelWorker` 再创建 `ModelRunner`。这些不是额外进程。

简化成：

```text
1 个 scheduler worker 进程
  contains Scheduler
  contains TpModelWorker
  contains ModelRunner
```

然后这个进程会初始化分布式环境。

源码位置：

```text
python/sglang/srt/model_executor/model_runner.py
```

关键逻辑：

```python
init_distributed_environment(
    world_size=self.tp_size * self.pp_size,
    rank=self.tp_size * self.pp_rank + self.tp_rank,
    ...
)

initialize_model_parallel(
    tensor_model_parallel_size=self.tp_size,
    attention_data_parallel_size=self.dp_size,
    pipeline_model_parallel_size=self.pp_size,
    expert_model_parallel_size=self.moe_ep_size,
    attention_context_model_parallel_size=self.attn_cp_size,
    moe_data_model_parallel_size=self.moe_dp_size,
)

initialize_dp_attention(
    server_args=self.server_args,
    model_config=self.model_config,
)
```

本文配置代入：

```text
world_size = tp_size * pp_size = 64 * 1 = 64
rank = tp_rank

tensor_model_parallel_size = 64
attention_data_parallel_size = 8
expert_model_parallel_size = 64
moe_data_model_parallel_size = 1
```

所以每个 worker 都知道：

```text
自己在 global TP/root group 里的 tp_rank
自己在 attention DP group 里的 attn_dp_rank / attn_tp_rank
自己在 MoE EP group 里的 moe_ep_rank
```

## 5. 建组结果总结

对于本文配置，建组后可以记成下面这张表：

| 组 | size | ranks | 作用 |
| --- | --- | --- | --- |
| global TP/root group | 64 | `[0..63]` | SGLang 的全局 model-parallel rank space；DeepEP group 也用它；不是 attention TP64 |
| attention DP | 8 | 8 个 DP groups | 请求级 DP 路由维度 |
| attention TP | 8 | `[0..7]`, `[8..15]`, ..., `[56..63]` | 每个请求流内部 attention TP8 |
| MoE EP | 64 | `[0..63]` | DeepEP64，专家跨 64 rank 切分 |
| MoE TP | 1 | 每个 rank 单独 | expert 内部不再 TP 切 |
| MoE DP | 1 | 每个 rank 单独 | 本配置没有 MoE data parallel |

换成一句话：

```text
同一批 64 张卡：
  attention 看成 8 组，每组 8 卡；
  MoE 看成 1 组，整组 64 卡。
```

## 6. HTTP 请求进入后发生什么

### 6.1 对外 URL

虽然内部有 64 个 worker rank，但对外业务 URL 仍然是一个：

```text
http://$SERVER_IP:$PORT
```

不是 64 个 URL，也不是 8 个 URL。

HTTP server 和 TokenizerManager 在主进程里，收到 OpenAI 请求后，会 tokenize，然后把请求送到 scheduler 侧。

由于：

```text
dp_size = 8
```

中间有：

```text
DataParallelController
```

### 6.2 DataParallelController 如何选择 DP rank

源码位置：

```text
python/sglang/srt/managers/data_parallel_controller.py
  maybe_external_dp_rank_routing()
  round_robin_scheduler()
```

如果请求里显式写了：

```json
{
  "routed_dp_rank": 3
}
```

或者 HTTP header：

```text
X-Data-Parallel-Rank: 3
```

那么 controller 会直接把请求发给：

```text
DP rank 3
```

源码逻辑：

```python
if req.routed_dp_rank is not None:
    self.workers[req.routed_dp_rank].send_pyobj(req)
    return True
```

如果不指定，则按 load balance 策略选择 DP rank，例如 round robin / total requests / total tokens 等。

### 6.3 `routed_dp_rank` 范围

本文配置：

```text
dp_size = 8
```

所以：

```text
routed_dp_rank = 0..7
```

不要写：

```text
routed_dp_rank = 32
```

因为 32 是 `tp_rank` / `moe_ep_rank` 可能的值，不是请求级 DP rank。

如果你指定：

```text
routed_dp_rank = 3
```

请求会进入：

```text
attention DP3
  ranks [24..31]
```

而不是进入单独的 rank 3。

## 7. 一个请求在 attention 阶段怎么跑

假设一个请求被路由到：

```text
routed_dp_rank = 3
```

那么它进入：

```text
attention DP3 = ranks [24..31]
```

在 Qwen3 MoE attention 里，源码位置：

```text
python/sglang/srt/models/qwen3_moe.py
  Qwen3MoeAttention.__init__()
```

attention 使用的是：

```python
attn_tp_rank = get_parallel().attn_tp_rank
attn_tp_size = get_parallel().attn_tp_size
```

不是直接用全局 `tp_size=64`。

所以本文配置下：

```text
Q/K/V projection:
  tp_size = attn_tp_size = 8

o_proj:
  tp_size = attn_tp_size = 8
```

这意味着：

```text
DP3 的 ranks [24..31] 共同完成该请求的 attention TP8
其它 DP groups 不参与这个请求的 attention
```

如果模型有：

```text
num_attention_heads = H
```

那么 attention head 是按：

```text
attn_tp_size = 8
```

切，而不是按 64 切。

这也是 DPA 的重要价值：

```text
全局 64 卡参与 engine
但 attention 不需要切成 TP64
```

## 8. 一个 decoder layer 内部流程

以 Qwen3 MoE 为例，decoder layer 源码位置：

```text
python/sglang/srt/models/qwen3_moe.py
  Qwen3MoeDecoderLayer.forward()
```

主流程是：

```python
hidden_states, residual = self.layer_communicator.prepare_attn(...)

hidden_states = self.self_attn(...)

hidden_states, residual = self.layer_communicator.prepare_mlp(...)

hidden_states = self.mlp(...)

hidden_states, residual = self.layer_communicator.postprocess_layer(...)
```

可以画成：

```text
layer input
  |
  v
LayerCommunicator.prepare_attn
  |
  v
self_attn
  |
  v
LayerCommunicator.prepare_mlp
  |
  v
Qwen3MoeSparseMoeBlock.forward_deepep
  |
  v
LayerCommunicator.postprocess_layer
  |
  v
layer output
```

这里 `LayerCommunicator` 的职责是处理不同 scatter mode 之间的转换。

对于本文配置：

```text
moe_a2a_backend = deepep
```

`LayerScatterModes._compute_mlp_mode()` 会把 sparse MoE 的 MLP mode 设成：

```text
SCATTERED
```

因为源码里有：

```python
if not get_moe_a2a_backend().is_none():
    return ScatterMode.SCATTERED
```

这意味着：

```text
MoE token dispatch/combine 交给 DeepEP dispatcher 处理
LayerCommunicator 不走 none 后端那种 FULL gather 路线
```

## 9. MoE 阶段：DeepEP64 怎么工作

### 9.1 Qwen3MoeSparseMoeBlock 分支

源码位置：

```text
python/sglang/srt/models/qwen3_moe.py
  Qwen3MoeSparseMoeBlock.forward()
```

关键分支：

```python
if not get_moe_a2a_backend().is_deepep():
    return self.forward_normal(...)
else:
    return self.forward_deepep(hidden_states, forward_batch)
```

本文配置：

```text
moe_a2a_backend = deepep
```

所以走：

```text
forward_deepep()
```

### 9.2 router / topk

`forward_deepep()` 里先做：

```python
router_logits, _ = self.gate(hidden_states)
topk_output = self.topk(...)
```

可以理解成：

```text
每个 token 先在当前 attention DP group 内得到 hidden_states
router 对 token 打分
topk 选出该 token 要去的 experts
```

比如：

```text
token A -> expert 3, expert 61
token B -> expert 8, expert 40
```

因为本配置：

```text
moe_ep_size = 64
moe_tp_size = 1
```

专家是跨 64 个 EP ranks 分布的。

如果 routed experts 数量是 `num_experts`，每个 rank 本地拥有的 routed experts 数量大致是：

```text
num_local_routed_experts = num_experts / 64
```

具体还要看模型是否有 shared experts / fused shared experts。

### 9.3 FusedMoE 选择 DeepEP dispatcher

MoE expert compute 在 FusedMoE 层里完成。

源码位置：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  create_moe_dispatcher()
```

当：

```text
a2a_backend.is_deepep()
```

会创建：

```text
MaybeTboDeepEPDispatcher(...)
```

它内部使用 DeepEPDispatcher。

关键点：

```text
group = get_tp_group().device_group
```

因为本文配置下：

```text
global TP/root group = [0..63]
MoE EP group = [0..63]
```

DeepEP dispatch/combine 就是在这 64 个 ranks 上发生。

### 9.4 DeepEP dispatch / compute / combine

源码位置：

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  DeepEPDispatcher
```

主接口：

```python
dispatch()
combine()
```

内部拆成：

```text
dispatch_a
dispatch_b
combine_a
combine_b
```

简化逻辑：

```text
hidden_states + topk_output
  |
  v
DeepEP dispatch
  |
  v
每个 expert owner rank 收到自己需要处理的 token-expert 任务
  |
  v
local expert GEMM / fused MoE compute
  |
  v
DeepEP combine
  |
  v
输出回到原来的 token / attention DP layout
```

对于前面的例子：

```text
token A -> expert 3, expert 61
```

如果：

```text
expert 3 在 rank 3
expert 61 在 rank 61
```

那么即使 token A 原来属于：

```text
attention DP3 = ranks [24..31]
```

DeepEP 也会把 token-expert 任务发到：

```text
rank 3
rank 61
```

计算完成后，再 combine 回 token A 原来的布局。

这就是：

```text
attention 是 DPA8/TP8
MoE 是 DeepEP64
```

同时成立的原因。

### 9.5 `deepep_mode=auto`

DeepEPDispatcher 里会按当前 batch 类型解析 mode：

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  DeepEPDispatcher._get_impl()
```

逻辑是：

```python
is_extend_in_batch = get_is_extend_in_batch()
resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
```

前面第 4 篇已经讲过：

```text
auto:
  prefill / extend -> normal
  decode -> low_latency
```

所以你的配置：

```bash
--sglang-deepep-mode auto
```

意味着：

```text
prefill:
  DeepEP normal 路径

decode:
  DeepEP low_latency 路径
```

## 10. `moe_dense_tp_size=1` 在这里做什么

这个参数容易和 MoE EP 混在一起。

你的配置：

```bash
--sglang-moe-dense-tp-size 1
```

对应：

```text
server_args.moe_dense_tp_size = 1
```

源码注释位置：

```text
python/sglang/srt/server_args.py
  --moe-dense-tp-size
```

说明它是：

```text
TP size for MoE dense MLP layers
```

它不是：

```text
MoE expert EP size
```

也不是：

```text
attention TP size
```

它主要影响 MoE 模型里 dense MLP 层的 TP 处理方式。

在 DPA 初始化里还有一套 local attention DP 公式：

```text
python/sglang/srt/layers/dp_attention.py
  compute_dp_attention_local_info()
```

关键逻辑：

```python
local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size
local_dp_size = max(1, dp_size // (tp_size // local_tp_size))
local_attn_tp_size = local_tp_size // local_dp_size
```

代入：

```text
tp_size = 64
dp_size = 8
moe_dense_tp_size = 1
```

得到：

```text
local_tp_size = 1
local_dp_size = max(1, 8 // (64 // 1))
              = max(1, 8 // 64)
              = 1
local_attn_tp_size = 1 // 1 = 1
```

可以直观理解成：

```text
dense MLP 相关路径尽量不再按大 TP group 切
而是更偏本地 / DP 化的执行语义
```

为什么业务上会这么配：

```text
1. 全局 tp_size=64 很大
2. 某些 dense MLP 权重维度如果继续 TP64，单 rank 分片可能太小
3. kernel / GEMM 可能不合适
4. moe_dense_tp_size=1 可以避免 dense MLP 被过度 TP 切分
```

要注意：

```text
它不改变主 MoE routed experts 的 DeepEP64
```

主 MoE expert 还是：

```text
moe_ep_size = 64
```

## 11. `enable_dp_lm_head` 在这里做什么

你的配置：

```bash
--sglang-enable-dp-lm-head
```

对应：

```text
server_args.enable_dp_lm_head = true
```

源码约束：

```text
python/sglang/srt/server_args.py
  _handle_data_parallelism()
```

里面有：

```python
if self.enable_dp_lm_head:
    assert self.enable_dp_attention
```

所以：

```text
enable_dp_lm_head 必须配 enable_dp_attention
```

Qwen3 MoE 的 LM head 初始化在：

```text
python/sglang/srt/models/qwen3_moe.py
  Qwen3MoeForCausalLM.__init__()
```

关键代码：

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    prefix=...,
    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
)
```

所以打开它后：

```text
LM head 使用 attention TP group
```

本文配置下：

```text
attention TP group size = 8
```

因此 LM head 更贴近每个 attention DP group 内的 TP8，而不是直接使用全局 TP64。

直观目的：

```text
在 DP attention 场景下，减少 logits / vocab parallel 阶段跨 DP group 的 all-gather 压力
```

对这个业务配置来说，这个参数是合理的，因为：

```text
dp attention 已开启
每个请求只属于 8 个 DP groups 之一
LM head 也应该尽量跟着 attention TP group 走
```

### 11.1 如果不设置 `enable_dp_lm_head` 会怎样

这里有一个容易误解的问题：

```text
如果没有设置 enable_dp_lm_head，
LM head 是不切，还是默认按 TP64 切？
```

答案是：

```text
不是不切。
默认更接近按全局 TP group 切，也就是 TP64 语义。
```

因为 `Qwen3MoeForCausalLM` 初始化 LM head 时传的是：

```python
use_attn_tp_group=get_global_server_args().enable_dp_lm_head
```

所以：

```text
enable_dp_lm_head = false:
  use_attn_tp_group = false
  LM head 使用默认全局 TP group
  本配置下就是 TP64 语义

enable_dp_lm_head = true:
  use_attn_tp_group = true
  LM head 使用 attention TP group
  本配置下就是每个 DP group 内 TP8
```

也就是说，你前面的理解是对的：

```text
开启 enable_dp_lm_head 后，
LM head 从全局 TP64 切分语义，
变成每个 attention DP group 内部 TP8 切分语义。
```

映射到本文 8 节点 x 8 卡：

```text
DP0 / node0:
  ranks [0..7]
  LM head TP8

DP1 / node1:
  ranks [8..15]
  LM head TP8

...

DP7 / node7:
  ranks [56..63]
  LM head TP8
```

这样 LM head / logits 相关通信就主要限制在一个 attention DP group 内。在这个部署映射里，也就是一个节点内 8 卡。

### 11.2 LM head TP 切分后的通信是 all-gather

另一个关键点：

```text
LM head 采用 TP / vocab parallel 切分时，
通信主要是 all-gather，不是 all-reduce。
```

原因是 LM head 通常按 vocab 维切分：

```text
rank0:
  logits[:, vocab shard 0]

rank1:
  logits[:, vocab shard 1]

...

rank7:
  logits[:, vocab shard 7]
```

每个 rank 得到的是不同 vocab 范围的 logits 分片：

```text
local_logits = [num_tokens, vocab_size / tp_size]
```

如果后续 sampling / logprob 需要完整 vocab logits，就要把这些 vocab shards 拼起来：

```text
all-gather -> full_logits = [num_tokens, vocab_size]
```

源码位置：

```text
python/sglang/srt/layers/logits_processor.py
```

关键逻辑：

```python
if self.do_tensor_parallel_all_gather:
    if self.use_attn_tp_group:
        logits = self._gather_attn_tp_logits(logits)
    else:
        logits = tensor_model_parallel_all_gather(logits)
```

其中 `_gather_attn_tp_logits()` 用的是：

```python
attn_tp_all_gather_into_tensor(...)
```

所以本文配置开启 `enable_dp_lm_head` 后，可以记成：

```text
LM head:
  在每个 DP group 内做 vocab parallel TP8

logits 通信:
  DP group 内 attention TP all-gather

通信范围:
  ranks [0..7] 或 [8..15] ... 或 [56..63]
```

不要把它和 input embedding 的通信混在一起。

input embedding 也是 vocab parallel，但它的语义不同：

```text
input embedding:
  token id 只命中某个 vocab shard
  各 rank 产出 embedding contribution
  常用 all-reduce 合并 hidden embedding

output LM head:
  各 rank 产出不同 vocab logits shard
  要用 all-gather 拼完整 logits
```

## 12. 一次请求的端到端图

假设请求被路由到：

```text
routed_dp_rank = 3
```

完整路径可以画成：

```text
HTTP request
  |
  v
FastAPI / TokenizerManager
  |
  v
DataParallelController
  |
  | routed_dp_rank = 3
  v
DP3 scheduler input socket
  |
  v
attention DP3 workers: ranks [24..31]
  |
  | scheduling / ForwardBatch
  v
Qwen3MoeForCausalLM.forward
  |
  v
Decoder layer
  |
  +--> attention:
  |      ranks [24..31]
  |      attn_tp_size = 8
  |
  +--> MoE router/topk:
  |      local hidden tokens produce expert ids
  |
  +--> DeepEP dispatch:
  |      across ranks [0..63]
  |
  +--> expert compute:
  |      each expert owner rank computes received token-expert tasks
  |
  +--> DeepEP combine:
  |      output returns to DP3 token layout
  |
  v
next decoder layer
  |
  v
LM head with attention TP group
  |
  v
sampling / token output
  |
  v
DetokenizerManager
  |
  v
HTTP response / stream chunk
```

最关键的跳变是：

```text
attention 阶段:
  ranks [24..31]

MoE 阶段:
  dispatch 到 ranks [0..63]

MoE combine 后:
  回到 ranks [24..31] 对应的 DP3 token layout
```

## 13. 为什么不是 64 个 DP 流

很多人看到：

```text
64 张 GPU
ep_size = 64
```

会直觉认为：

```text
64 个 rank 都能各自接请求
```

但本文配置里，请求级 DP 由：

```text
dp_size = 8
enable_dp_attention = true
```

决定。

因此请求级 worker 是：

```text
DP0..DP7
```

而不是：

```text
rank0..rank63
```

每个 DP worker 内部有 8 个 TP ranks：

```text
DP0 = ranks [0..7]
DP1 = ranks [8..15]
...
DP7 = ranks [56..63]
```

所以：

```text
HTTP 请求路由到 DP rank
不是路由到 GPU rank
```

这也是你做 RL 时最需要记住的点。

## 14. 这个配置的性能直觉

这个配置是在几个目标之间折中。

### 14.1 为什么不用 DeepEP-only

DeepEP-only 可能是：

```text
tp_size = 64
dp_size = 1
ep_size = 64
enable_dp_attention = false
```

那就是：

```text
attention TP64
MoE DeepEP64
请求级 DP = 1
```

问题：

```text
1. 只有一个请求级 DP 流
2. RL rollout 的大量独立请求不能自然分到 8 个 DP groups
3. attention head / dense 部分 TP64 也可能过细
```

### 14.2 为什么不是 DPA64

DPA64 是：

```text
tp_size = 64
dp_size = 64
enable_dp_attention = true
ep_size = 64
```

它会变成：

```text
attn_dp_size = 64
attn_tp_size = 1
```

这会给你 64 个请求级 DP 流，但每个请求流只有单卡 attention。

这不一定适合大模型，因为：

```text
1. attention/dense 权重可能单卡放不下或效率不好
2. batch/token 形态可能更碎
3. 每个 DP 流的本地计算太小，调度和通信开销可能变高
```

### 14.3 为什么 DPA8 + DeepEP64 合理

你的配置：

```text
tp_size = 64
dp_size = 8
enable_dp_attention = true
ep_size = 64
```

折中成：

```text
8 个请求级 DP 流
每个 DP 流 attention TP8
MoE 仍然 DeepEP64
```

好处：

```text
1. 比 DeepEP-only 多了 8 倍请求级并行入口
2. 比 DPA64 更保守，每个请求流仍有 TP8 承载 attention/dense
3. MoE expert pool 最大化使用 64 卡 DeepEP
4. routed_dp_rank 对 RL 侧足够清晰，范围是 0..7
```

## 15. 源码阅读顺序

建议你按下面顺序读源码。

### 15.1 参数处理

```text
python/sglang/srt/server_args.py
  _handle_data_parallelism()
    dp_size=1 时关闭 enable_dp_attention
    enable_dp_lm_head 要求 enable_dp_attention
    enable_dp_attention 要求 tp_size % dp_size == 0

  _handle_a2a_moe()
    moe_a2a_backend == deepep 时 ep_size = tp_size

  --moe-dense-tp-size
    dense MLP TP size 参数
```

### 15.2 启动进程

```text
python/sglang/srt/entrypoints/http_server.py
  launch_server()

python/sglang/srt/entrypoints/engine.py
  Engine._launch_subprocesses()
  Engine._launch_scheduler_processes()

python/sglang/srt/managers/data_parallel_controller.py
  DataParallelController.__init__()
  launch_dp_attention_schedulers()
  launch_tensor_parallel_group()
```

### 15.3 rank 计算和建组

```text
python/sglang/srt/layers/dp_attention.py
  compute_dp_attention_world_info()
  compute_dp_attention_local_info()
  initialize_dp_attention()

python/sglang/srt/distributed/parallel_state.py
  initialize_model_parallel()
```

### 15.4 请求路由

```text
python/sglang/srt/entrypoints/openai/serving_base.py
  X-Data-Parallel-Rank header

python/sglang/srt/managers/tokenizer_manager.py
  routed_dp_rank 范围校验

python/sglang/srt/managers/data_parallel_controller.py
  maybe_external_dp_rank_routing()
  round_robin_scheduler()
```

### 15.5 模型 forward

```text
python/sglang/srt/models/qwen3_moe.py
  Qwen3MoeForCausalLM
  Qwen3MoeDecoderLayer.forward()
  Qwen3MoeAttention
  Qwen3MoeSparseMoeBlock.forward_deepep()
```

### 15.6 DeepEP

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  create_moe_dispatcher()
  forward()
  run_moe_core()

python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  DeepEPDispatcher
  dispatch_a / dispatch_b
  combine_a / combine_b
```

## 16. 最终记忆

把这篇文档压缩成一句话：

```text
这个配置是 64 卡一个 engine：
attention 按 DPA8 x TP8 跑请求流；
MoE 按 DeepEP64 跑 expert dispatch；
LM head 跟 attention TP group；
RL 侧可路由 DP rank 只有 0..7。
```

再压缩成一张图：

```text
64 GPU engine

attention:
  DP0 [0..7]
  DP1 [8..15]
  ...
  DP7 [56..63]

MoE:
  DeepEP group [0..63]

RL routed_dp_rank:
  0..7
```

## 17. 单条请求时的实际参与路径

最后补一个很贴近实际运行的场景。

假设当前只有 1 条请求进来，并且没有显式指定：

```text
routed_dp_rank
```

那么它会先进入 node0 的 HTTP server / TokenizerManager，然后交给 node0 的 DataParallelController。

DataParallelController 会按当前 load balance 策略选择一路 DP：

```text
round_robin
total_requests
total_tokens
或者其它配置的 load balance method
```

这不是严格随机。为了方便说明，假设它选中了：

```text
routed_dp_rank = 3
```

那么 attention / dense 主干阶段只会走：

```text
attention DP3 = ranks [24..31]
```

也就是：

```text
attention 阶段:
  只用 8 张卡
  这 8 张卡做 attention TP8
```

到 MoE 层时，流程变成：

```text
hidden states
  -> router / topk
  -> DeepEP dispatch across ranks [0..63]
  -> expert owner ranks compute
  -> DeepEP combine back to DP3 token layout
```

所以 MoE 阶段会进入：

```text
DeepEP64 通信域
```

更精确地说：

```text
不是 64 张卡一定都有实际 expert GEMM
而是 token-expert 任务会被发到 64 个 EP ranks 中对应 expert owner rank
```

如果这条请求 token 很少、topk 很少：

```text
实际收到 token-expert 任务的 expert ranks 可能只是 64 个 ranks 的一部分
```

但通信域仍然是：

```text
MoE EP group = [0..63]
```

所以这条单请求的路径可以总结为：

```text
请求入口:
  1 条请求

DP 路由:
  选中 1 个 DP rank，例如 DP3

attention:
  只在 DP3 的 8 张卡上做 TP8

MoE:
  进入 DeepEP64
  token-expert 任务分发到 64 卡中对应 expert 所在 ranks

combine:
  MoE 输出回到 DP3 token layout

下一层:
  继续在 DP3 的 attention TP8 上跑
```

一句话：

```text
单请求时，attention 只占用一路 DP 的 8 卡；
MoE 阶段会进入 DeepEP64，把 token-expert 分发到 64 卡中的 expert owner ranks。
```

# 04. MoE EP 的 DeepEP 后端

本文讨论：

```text
moe_a2a_backend = deepep
```

它是 MoE EP 的 A2A 通信后端，不是新的并行维度。

和 03 文档里的普通 EP `none` 后端相比，DeepEP 最大的变化是 MoE 阶段的通信路线：

```text
none 后端:
  token 复制在每个 EP rank 上
  本地 expert 计算 partial output
  最后 EP all-reduce

deepep 后端:
  token 按 topk dispatch 到 expert 所在 rank
  expert rank 只计算收到的 token
  combine 把结果聚合回来
```

一句话：

> DeepEP 是把普通 EP 的“输出 all-reduce”换成“token dispatch + expert compute + output combine”的 A2A 路线。

## 0. DeepEP 不是随便替换 `none`

当前 SGLang 支持的 A2A backend 包括：

```text
none
deepep
mooncake
nixl
mori
ascend_fuseep
flashinfer
megamoe
```

本文只讨论 `deepep`。

为了后面查阅，先把两类 MoE backend 记在这里。


### 0.1 `moe-a2a-backend`：MoE 跨 rank 通信后端

这个参数决定 MoE 阶段 token / output 怎么跨 EP ranks 通信。

默认值：

```text
--moe-a2a-backend none
```

当前可选值：

| backend | 主要职责 | 简要理解 |
| --- | --- | --- |
| `none` | 不启用专门 A2A token dispatch | 普通 EP 路线，token 复制，本地 expert 计算，最后 output all-reduce |
| `deepep` | DeepEP token dispatch / combine | NVIDIA/DeepSeek 系常见大规模 EP A2A 后端 |
| `mooncake` | Mooncake EP 通信 | 偏 elastic inference / RDMA 场景 |
| `nixl` | NIXL-EP 通信 | 偏弹性、RDMA/NVLink、动态扩缩场景 |
| `mori` | MORI-EP 通信 | AMD ROCm 场景 |
| `ascend_fuseep` | Ascend NPU fused EP | Ascend NPU 场景 |
| `flashinfer` | FlashInfer A2A | 通常和 FlashInfer/CuteDSL/Cutlass MoE runner 组合 |
| `megamoe` | MegaMoE 特定优化路径 | 特定优化，不作为普通学习默认选择 |

注意：

```text
none 是默认值
none 不是“不开 EP”
none 表示“不开专门 A2A token dispatch”
```


### 0.2 `moe-runner-backend`：本地 expert 计算后端

这个参数决定 token 已经到本地 expert 后，expert GEMM / fused MoE compute 怎么算。

默认值：

```text
--moe-runner-backend auto
```

当前可选值：

| backend | 主要职责 | 简要理解 |
| --- | --- | --- |
| `auto` | 自动选择 MoE compute backend | 默认值，按模型、量化、硬件选择 |
| `deep_gemm` | DeepGEMM grouped GEMM | 大规模 MoE 常见高性能 expert GEMM 后端 |
| `triton` | Triton fused MoE | 通用 Triton 实现 |
| `triton_kernel` | Triton kernel MoE | Triton kernel 路径 |
| `flashinfer_trtllm` | FlashInfer + TensorRT-LLM MoE | 常用于 Blackwell / FP4 等优化场景 |
| `experimental_sgl_trtllm` | 实验性 SGL/TRTLLM 路径 | 实验优化 |
| `flashinfer_trtllm_routed` | FlashInfer routed MoE | 使用 SGLang topk routing 的 FlashInfer TRTLLM 路径 |
| `flashinfer_cutlass` | FlashInfer + CUTLASS MoE | FP4/FP8 等低精度 MoE 常见 |
| `flashinfer_mxfp4` | FlashInfer MXFP4 MoE | MXFP4 相关 |
| `flashinfer_cutedsl` | FlashInfer CuteDSL MoE | NVFP4 / CuteDSL 相关 |
| `cutlass` | CUTLASS MoE | NVIDIA CUTLASS GEMM 路径 |
| `aiter` | AITER MoE | AMD/ROCm 相关优化路径 |
| `marlin` | Marlin MoE | 量化相关后端 |

可以记成：

```text
moe-a2a-backend:
  管通信
  token 怎么到 expert rank
  output 怎么 combine 回来

moe-runner-backend:
  管计算
  到了本地 expert 后 GEMM 怎么算
```

比如：

```bash
--moe-a2a-backend deepep \
--moe-runner-backend deep_gemm
```

含义是：

```text
通信:
  DeepEP dispatch / combine

计算:
  DeepGEMM expert GEMM
```

DeepEP 和 `none` 不是简单字符串替换。它会改变 MoE 层内部通信方式，也会触发 server args 的自动改参：

```python
if self.moe_a2a_backend == "deepep":
    self.ep_size = self.tp_size
```

所以 DeepEP 当前语义基本是：

```text
ep_size = tp_size
moe_tp_size = 1
```

如果你写：

```bash
--ep-size 8 \
--moe-a2a-backend deepep
```

但漏了：

```bash
--tp-size 8
```

那不是 EP8。因为默认 `tp_size=1`，DeepEP 会把：

```text
ep_size = tp_size = 1
```

所以 DeepEP 启动也建议显式写全：

```bash
--tp-size N \
--ep-size N \
--moe-a2a-backend deepep
```

## 1. 启动命令

### 1.1 单节点 8 卡 DeepEP

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

推荐把 `--ep-size 8` 显式写上，虽然 DeepEP 代码会执行 `ep_size = tp_size`。这样文档、日志、脚本意图都更清楚。

如果模型和硬件适合 DeepGEMM，也常见写法是：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

### 1.1.1 DeepEP 和 DeepGEMM 是什么关系

这里有两个容易混的参数：

```text
--moe-a2a-backend deepep
--moe-runner-backend deep_gemm
```

它们负责的阶段不同。

`--moe-a2a-backend deepep` 管的是 MoE token 通信：

```text
router/topk 之后
把 token-expert 任务 dispatch 到 expert 所在 rank
expert 计算之后
把 expert 输出 combine 回原 token 布局
```

`--moe-runner-backend deep_gemm` 管的是 expert 本地计算：

```text
dispatch 之后
每个 rank 收到自己要处理的 tokens
对本地 experts 做 grouped GEMM / fused MoE compute
产出本地 expert output
```

所以完整链路是：

```text
hidden_states
  -> router / topk
  -> DeepEP dispatch        由 --moe-a2a-backend deepep 负责
  -> expert GEMM compute    由 --moe-runner-backend deep_gemm 负责
  -> DeepEP combine         由 --moe-a2a-backend deepep 负责
  -> MoE output
```

换句话说：

```text
DeepEP:
  解决 token 怎么跨 rank 搬到 expert 所在位置

DeepGEMM:
  解决 token 到了本地 expert 之后，矩阵乘怎么算得快
```

它们可以组合，但不是同一个东西。只写：

```bash
--moe-a2a-backend deepep
```

表示 MoE 通信走 DeepEP，MoE 计算 backend 仍然按 `--moe-runner-backend` 的默认值 `auto` 选择。

显式写：

```bash
--moe-a2a-backend deepep \
--moe-runner-backend deep_gemm
```

表示：

```text
通信:
  DeepEP dispatch / combine

计算:
  DeepGEMM expert GEMM
```

为什么很多 DeepEP 示例会同时写 DeepGEMM：

```text
1. DeepEP 把 tokens 高效搬到本地 expert
2. DeepGEMM 对本地 expert grouped GEMM 做高性能计算
3. 二者组合适合大规模 MoE，尤其是 DeepSeek/Qwen 这类大 MoE serving 场景
```

但这不是说 DeepEP 必须永远搭配 DeepGEMM。是否指定 `deep_gemm` 取决于：

```text
模型精度 / quantization
GPU 架构
DeepGEMM 是否安装和可用
当前 SGLang auto 是否已经能选到合适 runner
```

学习阶段可以先记住：

```text
moe-a2a-backend:
  通信后端

moe-runner-backend:
  本地 expert 计算后端
```

### 1.2 两机 2 x 8 卡 DeepEP16

node 0：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 16 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
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
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --dp-size 1 \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --dist-init-addr $NODE0_IP:50000 \
  --nnodes 2 \
  --node-rank 1
```

对 RL rollout 来说，业务 URL 仍然只有一个：

```text
base_url = http://$NODE0_IP:30000
```

DeepEP 改的是 server 内部 MoE 通信，不会让每张卡变成一个独立 HTTP endpoint。

## 2. DeepEP 的 mode

DeepEP mode 有三个：

```text
normal
low_latency
auto
```

源码里：

```python
class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY
```

可以理解成：

```text
normal:
  更偏 prefill / extend，吞吐优先

low_latency:
  更偏 decode，低延迟和 CUDA Graph 友好

auto:
  prefill 时用 normal
  decode 时用 low_latency
```

一般学习和线上默认先用：

```bash
--deepep-mode auto
```

如果设置：

```bash
--deepep-mode normal
```

代码里会关闭 cuda graph：

```python
if self.deepep_mode == "normal":
    self.cuda_graph_config.decode.backend = Backend.DISABLED
    self.cuda_graph_config.prefill.backend = Backend.DISABLED
```

所以 `normal` 更适合排查 prefill 或吞吐路径，不一定适合作为 decode 低延迟默认配置。

## 3. DeepEP 和 `none` 后端的核心差异

### 3.1 `none` 后端

普通 EP `none` 的 MoE 逻辑是：

```text
每个 EP rank 都有同一批 tokens
每个 rank 只算本地 experts
没命中本地 expert 的 token contribution = 0
最后 EP all-reduce
```

图：

```text
token A -> expert 1 + expert 6

rank 0 owns expert 0,1:
  compute expert 1 contribution

rank 1 owns expert 2,3:
  contribution = 0

rank 2 owns expert 4,5:
  contribution = 0

rank 3 owns expert 6,7:
  compute expert 6 contribution

EP all-reduce:
  expert 1 contribution + expert 6 contribution
```

### 3.2 DeepEP 后端

DeepEP 的 MoE 逻辑是：

```text
每个 rank 先有本地 tokens
router/topk 算出 token 应该去哪些 experts
DeepEP dispatch 把 token-expert 任务发到 expert 所在 rank
expert rank 只计算收到的 token
DeepEP combine 把结果聚合回原来的 token/rank 布局
```

同样的例子：

```text
token A -> expert 1 + expert 6
```

如果：

```text
rank 0 owns expert 0,1
rank 3 owns expert 6,7
```

DeepEP 会做：

```text
dispatch:
  token A for expert 1 -> rank 0
  token A for expert 6 -> rank 3

expert compute:
  rank 0 computes expert 1 contribution
  rank 3 computes expert 6 contribution

combine:
  把 expert 1 contribution 和 expert 6 contribution 聚合回 token A 的输出
```

这里不是每个 rank 都对 token A 产出一个 `[hidden]` partial output 再 all-reduce，而是 token-expert 任务被发送到实际拥有 expert 的 rank。

## 4. 单层执行图

假设：

```text
tp_size = 4
ep_size = 4
moe_tp_size = 1
moe_a2a_backend = deepep
deepep_mode = auto
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

单层图：

```text
              ┌──────────────────────────────┐
              │        input hidden_states    │
              │        [num_tokens, hidden]   │
              └───────────────┬──────────────┘
                              │
                  attention 仍然按 TP 切
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
              │ 每个 rank 有同一批 hidden     │
              └───────────────┬──────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ router gate + topk            │
              │ token A -> expert 1 + 6       │
              └───────────────┬──────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ DeepEP dispatch              │
              │ token-expert tasks A2A       │
              └───────┬────────────────┬─────┘
                      │                │
                      ▼                ▼
        ┌─────────────────┐    ┌─────────────────┐
        │ rank 0          │    │ rank 3          │
        │ owns expert 0,1 │    │ owns expert 6,7 │
        │ recv A -> e1    │    │ recv A -> e6    │
        │ compute e1      │    │ compute e6      │
        └────────┬────────┘    └────────┬────────┘
                 │                      │
                 └──── DeepEP combine ──┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │ final MoE output per token   │
              │ e1 contribution + e6 contrib │
              └──────────────────────────────┘
```

这张图和 03 的 `none` 图对比：

```text
none:
  每个 rank 原地算本地 expert contribution
  最后 all-reduce output

deepep:
  token-expert 任务先 dispatch 到 expert rank
  expert rank 算收到的 tokens
  combine 回来
```

## 5. 源码路径

### 5.1 server args

文件：

```text
python/sglang/srt/server_args.py
```

关键逻辑：

```python
if self.moe_a2a_backend == "deepep":
    if self.deepep_mode == "normal":
        self.cuda_graph_config.decode.backend = Backend.DISABLED
        self.cuda_graph_config.prefill.backend = Backend.DISABLED
    self.ep_size = self.tp_size
```

结论：

```text
DeepEP 会强制 ep_size = tp_size
```

### 5.2 MoE block

文件：

```text
python/sglang/srt/models/qwen3_moe.py
```

DeepEP 路径：

```python
if get_moe_a2a_backend().is_deepep():
    return self.forward_deepep(hidden_states, forward_batch)
```

`forward_deepep()` 做：

```text
gate
topk
experts(hidden_states, topk_output)
```

注意这里不再像普通 EP `forward_normal()` 那样在 block 外面显式做：

```python
moe_expert_parallel_all_reduce(...)
```

DeepEP 的 dispatch/combine 被封装在 `FusedMoE` 的 dispatcher 里。

### 5.3 dispatcher 选择

文件：

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

关键逻辑：

```python
elif a2a_backend.is_deepep():
    return MaybeTboDeepEPDispatcher(...)
```

也就是说 `moe_a2a_backend=deepep` 会从 `StandardDispatcher` 切到 DeepEP dispatcher。

### 5.4 DeepEP dispatcher

文件：

```text
python/sglang/srt/layers/moe/token_dispatcher/deepep.py
```

核心阶段：

```text
dispatch_a
dispatch_b
run_moe_core
combine_a
combine_b
```

普通 forward 入口：

```python
def dispatch(self, hidden_states, topk_output):
    self.dispatch_a(hidden_states, topk_output)
    ret = self.dispatch_b()
    return ret

def combine(self, combine_input):
    self.combine_a(combine_input)
    ret = self.combine_b()
    return ret
```

mode 选择：

```python
resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
if resolved_deepep_mode == DeepEPMode.NORMAL:
    return self._normal_dispatcher
elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
    return self._low_latency_dispatcher
```

## 6. DeepEP 的依赖和限制

### 6.1 需要安装 DeepEP

`deepep.py` 里会尝试：

```python
from deep_ep import Buffer, Config
```

如果没有安装：

```text
DeepEP is not installed. Please install DeepEP package...
```

所以 `--moe-a2a-backend deepep` 不是纯 Python 配置开关，环境里必须有 DeepEP 依赖。

### 6.2 low_latency dispatch token 上限

环境变量：

```text
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

默认值在当前代码里是：

```text
128
```

它只和 DeepEP 的 `low_latency` 路径有关。代码里 low-latency dispatch 会把它传给 DeepEP：

```python
buffer.low_latency_dispatch(
    hidden_states,
    topk_ids,
    self.num_max_dispatch_tokens_per_rank,
    self.num_experts,
    ...
)
```

可以理解成：

```text
DeepEP low_latency 模式下，每个 rank 预留的最大 dispatch token 容量
```

这个值主要用于分配 / 规划 DeepEP low-latency 通信 buffer。它不是：

```text
不是 HTTP 请求数
不是 max_running_requests
不是模型总 batch size
不是 num_experts
```

更接近：

```text
MoE decode 阶段，经过 topk routing 后，每个 rank 可能要 dispatch / receive 的 token 容量上限
```

因为 MoE routing 会有负载偏斜，某些 expert/rank 可能收到比平均值更多的 tokens，所以这个值需要能覆盖实际 decode 时的峰值。

DeepEP 里还有硬性断言：

```python
assert self.num_max_dispatch_tokens_per_rank <= 1024
```

原因注释里写得很直接：

```text
DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
num-tokens-sent-from-one-rank-to-another-rank must be less than it
```

也就是说：

```text
这个值不能无限调大
当前代码要求 <= 1024
```

是否需要用户手动设置：

```text
普通学习 / 小 batch:
  不需要，先用默认 128

decode batch 较大 / 并发较高 / topk 较大 / expert 负载偏斜明显:
  可能需要调大

出现 DeepEP low_latency dispatch 容量相关错误:
  需要调大
```

设置方式：

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto
```

调参取舍：

```text
设太小:
  大 batch / 热 expert 场景可能不够用，dispatch 失败或触发容量问题

设太大:
  DeepEP low_latency buffer 占用更多显存 / 通信资源
  不一定带来性能收益
```

实践建议：

```text
先用默认 128
只有当 decode 并发上来，或者日志/错误显示 DeepEP dispatch 容量不够时，再逐步调到 256 / 512
不要一开始就拉到 1024
```

### 6.3 `ep_size = tp_size`

DeepEP 当前不适合下面这种 hybrid 配置：

```text
tp_size = 16
ep_size = 8
moe_tp_size = 2
```

因为代码会强制：

```text
ep_size = tp_size
```

如果要学 hybrid EP + MoE TP，先用：

```bash
--moe-a2a-backend none
```

### 6.4 attention head 约束仍然存在

DeepEP 只改变 MoE 阶段通信。没有启用 DPA 时：

```text
attn_tp_size = tp_size
```

所以 Q heads 仍然需要能被 `tp_size` 整除。DeepEP 不解决 attention head 不够切的问题。

如果想要：

```text
ep_size = 32
attention 不按 32 切
```

要学的是后面的：

```text
DPA + EP + DeepEP
```

## 7. 对 RL rollout 的影响

DeepEP 不改变对外服务形态：

```text
一个 SGLang server instance
一个业务 base_url
```

例如两机 DeepEP16：

```text
node 0:
  http://$NODE0_IP:30000  业务入口

node 1:
  参与 distributed world
  不是业务推理入口
```

RL worker 仍然只打：

```text
http://$NODE0_IP:30000/v1/chat/completions
```

DeepEP 改善的是 server 内部 MoE dispatch/combine 通信，不会把 EP ranks 变成多个 rollout endpoint。

## 8. 什么时候选 DeepEP

建议选择 DeepEP 的场景：

```text
1. MoE 模型很大，EP 是主要扩展方式
2. ep_size = tp_size
3. 希望 MoE 阶段按 token-expert 做 A2A dispatch
4. 硬件和环境支持 DeepEP
5. 普通 none 后端的 output all-reduce 成为瓶颈
```

不建议一开始就选 DeepEP 的场景：

```text
1. 只是学习 EP 基本逻辑
2. hybrid EP + MoE TP，例如 tp16 ep8
3. DeepEP 依赖没装好
4. attention head 本身不够 tp_size 切
5. 还没搞清楚 none 后端的 token 复制 + all-reduce 路线
```

学习顺序上：

```text
先懂 none:
  为什么 token 复制
  为什么 output all-reduce

再懂 DeepEP:
  为什么 token dispatch
  dispatch/combine 如何替代 output all-reduce
```

## 9. 和 03 文档的对照总结

| 项目 | `none` 后端 | `deepep` 后端 |
| --- | --- | --- |
| token 是否 A2A | 否 | 是 |
| 每个 EP rank 是否保留同一批 tokens | 是 | MoE dispatch 后不是 |
| expert 计算 | 本地 expert 命中才算，否则 0 | 收到 token-expert 任务才算 |
| MoE 后通信 | EP all-reduce | DeepEP combine |
| 是否强制 `ep_size=tp_size` | 否，但 EP-only 通常这样设 | 是 |
| 是否适合 hybrid EP+MoE TP | 是 | 不适合 |
| 依赖复杂度 | 低 | 高 |

最终记忆：

```text
none = expert 权重切分 + token 复制 + output all-reduce
deepep = expert 权重切分 + token dispatch + output combine
```

## 10. 源码阅读顺序

建议按这个顺序看：

```text
docs/advanced_features/expert_parallelism.md
  看 SGLang 对 EP backend 的官方分类

python/sglang/srt/server_args.py
  看 moe_a2a_backend == deepep 时如何强制 ep_size = tp_size
  看 deepep_mode normal 如何影响 cuda graph

python/sglang/srt/layers/moe/utils.py
  看 DeepEPMode normal / low_latency / auto

python/sglang/srt/models/qwen3_moe.py
  看 forward_deepep()
  看它和 forward_normal() 的区别

python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  看 create_moe_dispatcher()
  看 deepep 如何选择 dispatcher

python/sglang/srt/layers/moe/token_dispatcher/deepep.py
  看 DeepEPDispatcher
  看 dispatch_a / dispatch_b / combine_a / combine_b
```

## 11. 引出下一步：DeepEP 不是 DP

这里最后补一个很关键的边界。

在本文讨论的 DeepEP-only 配置里：

```text
tp_size = ep_size
dp_size = 1
enable_dp_attention = false
moe_a2a_backend = deepep
```

它和 03 文档的 `none` 后端一样，本质上还是一个模型副本内部的 EP。外部请求进入 server 后，attention 阶段不是多个 DP rank 各自处理不同请求流。

DeepEP 让“不同 rank 看到不同 token”的位置发生在 MoE 内部：

```text
attention 之后:
  每个 rank 仍属于同一个请求 batch 的并行计算

router / topk 之后:
  DeepEP dispatch 把 token-expert 任务发到 expert 所在 rank

expert compute 阶段:
  每个 EP rank 实际计算的 token-expert 子集可以不同

combine 之后:
  输出回到原 token 布局，继续后续层
```

所以要区分两句话：

```text
DeepEP:
  MoE expert compute 阶段，不同 EP rank 收到的 token-expert 任务可以不同。

DPA + EP:
  attention 阶段开始，不同 DP rank 就可以承接不同请求 / token 流。
```

如果你的目标是 RL rollout 里让不同 GPU rank 承接不同请求，同时 MoE expert 又做 EP 切分，下一步应该学习：

```text
--enable-dp-attention
--dp-size N
--tp-size N
--ep-size N
```

也就是第 5 篇要讲的 DPA + EP。

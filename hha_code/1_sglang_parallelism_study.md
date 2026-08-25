# SGLang 并行与组合并行学习文档

本文面向 SGLang 的 LLM/SRT serving runtime，学习目标是从启动命令一路理解到内部并行组、调度、模型 forward、MoE dispatch/runner。示例优先以 MoE 模型为背景，例如 Qwen3.5 / Qwen MoE / DeepSeek 这类模型。

说明：

- 这里的“并行”主要指 `python -m sglang.launch_server` 和 `python -m sglang_router.launch_server` 这条推理服务路径。
- diffusion runtime 也有 TP/SP/FSDP/CFG 等并行，但不是 Qwen3.5 MoE LLM 这条主线，放在附录。
- 本文基于当前仓库文档和代码整理，重点参考 `docs/advanced_features/*.md`、`python/sglang/srt/server_args.py`、`python/sglang/srt/distributed/parallel_state.py`、`python/sglang/srt/model_executor/model_runner.py`。

## 1. 先建立并行地图

SGLang LLM/SRT 里主线并行可以分成三层：

| 层次 | 并行方式 | CLI 参数 | 核心作用 |
| --- | --- | --- | --- |
| 模型切分 | Tensor Parallelism | `--tp-size` / `--tp` | 把线性层、attention head、vocab/lm_head 等切到多卡 |
| 模型切分 | Pipeline Parallelism | `--pp-size` / `--pp` | 按层切模型，不同 stage 之间 P2P 传激活 |
| MoE 专用 | Expert Parallelism | `--ep-size` / `--ep` | 把 MoE expert 分布到多卡 |
| MoE 专用 | MoE TP | 无单独 CLI，由公式算出 | 对 MoE expert 内部 FFN 再做张量并行 |
| MoE 专用 | MoE DP | `--moe-dp-size` | MoE 侧数据并行，配合 EP/CP 使用 |
| 请求副本 | Data Parallelism | `--dp-size` / `--dp` | 多个完整模型副本处理不同请求 |
| attention 专用 | DP Attention / DPA | `--enable-dp-attention` + `--dp-size` | attention 侧按 DP 拆 KV/cache，MoE/MLP 侧继续跨卡协作 |
| attention 专用 | Attention Context Parallel | `--attn-cp-size` 或 `--enable-prefill-cp --cp-strategy ...` | 长上下文 prefill 时按 context/token 维度切 attention |
| 部署拆分 | PD Disaggregation | `--disaggregation-mode prefill/decode` | prefill 和 decode 分开部署 |
| 部署拆分 | EPD Disaggregation | `--encoder-only` / `--language-only` + PD | VLM 的 encoder/prefill/decode 三段拆开 |
| 路由扩展 | SGLang Model Gateway | `python -m sglang_router...` | 推荐的生产级 DP/PD 路由入口 |
| EP 优化 | TBO/SBO | `--enable-two-batch-overlap` / `--enable-single-batch-overlap` | 计算和通信重叠，不是新的并行维度 |

最重要的心智模型：

```text
一个 SGLang worker 进程组:

world_size = tp_size * pp_size

在每个 tp group 内，SGLang 再拆：

attention:
  attn_tp_size = tp_size / (dp_size * attn_cp_size)
  attn_dp_size = dp_size                    # 仅 --enable-dp-attention 时真正生效
  attn_cp_size = --attn-cp-size

MoE:
  moe_tp_size = tp_size / (ep_size * moe_dp_size)
  moe_ep_size = ep_size
  moe_dp_size = --moe-dp-size
```

这些公式来自 `initialize_model_parallel()`。所以 `tp_size` 在组合并行下常常不是“纯 TP”，而是一个大并行组的大小，里面再拆出 attention TP、attention DP、attention CP、MoE TP、MoE EP、MoE DP。

## 2. 启动参数入口

主要入口：

```bash
python -m sglang.launch_server --help
```

代码入口：

- `python/sglang/srt/server_args.py`
  - `ServerArgs` 定义默认值。
  - `add_cli_args()` 定义 CLI。
  - `post_init()` 和一系列 `_handle_*()` 做参数自动改写和校验。
- `python/sglang/srt/model_executor/model_runner.py`
  - 初始化 torch distributed。
  - 调用 `initialize_model_parallel()`。
- `python/sglang/srt/distributed/parallel_state.py`
  - 真正创建 TP/PP/attention/MoE 进程组。

常见写法：

仓库文档和测试里经常能看到 `--tp/--dp/--ep` 这类短写；为了复制命令时更稳，本文示例统一使用长参数。

| 常见短写 | 长参数 | 内部字段 |
| --- | --- | --- |
| `--tp` | `--tensor-parallel-size` / `--tp-size` | `tp_size` |
| `--pp` | `--pipeline-parallel-size` / `--pp-size` | `pp_size` |
| `--dp` | `--data-parallel-size` / `--dp-size` | `dp_size` |
| `--ep` | `--expert-parallel-size` / `--ep-size` | `ep_size` |
| 无 | `--attention-context-parallel-size` / `--attn-cp-size` | `attn_cp_size` |
| 无 | `--moe-data-parallel-size` / `--moe-dp-size` | `moe_dp_size` |

## 3. Tensor Parallelism

### 3.1 启动方式

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B \
  --trust-remote-code \
  --tp-size 8
```

### 3.2 它切什么

TP 主要切：

- attention 的 QKV/O projection。
- MLP 的 up/gate/down projection。
- vocab embedding 和 lm_head。
- 对 RowParallelLinear 结果做 all-reduce。
- 对部分 ColumnParallelLinear 或 logits 做 all-gather。

源码入口：

- `python/sglang/srt/layers/linear.py`
  - `ColumnParallelLinear`
  - `QKVParallelLinear`
  - `MergedColumnParallelLinear`
  - `RowParallelLinear`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
  - `VocabParallelEmbedding`
  - `ParallelLMHead`
- `python/sglang/srt/distributed/communication_op.py`
  - `tensor_model_parallel_all_reduce`
  - `tensor_model_parallel_all_gather`
- `python/sglang/srt/models/*`
  - 不同模型会读取 `get_parallel().tp_size` 或 `get_parallel().attn_tp_size`。

### 3.3 学习重点

先用 dense 模型理解 TP，再进入 MoE。你需要能回答：

- ColumnParallelLinear 是切输出维还是输入维？
- RowParallelLinear 为什么 forward 后要 all-reduce？
- QKVParallelLinear 如何处理 `num_heads / tp_size`？
- logits 为什么有时需要 all-gather？

## 4. Pipeline Parallelism

### 4.1 启动方式

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 \
  --trust-remote-code \
  --tp-size 4 \
  --pp-size 2
```

多节点时：

```bash
# node 0
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8 \
  --pp-size 2 \
  --dist-init-addr $MASTER_IP:50000 \
  --nnodes 2 \
  --node-rank 0

# node 1
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8 \
  --pp-size 2 \
  --dist-init-addr $MASTER_IP:50000 \
  --nnodes 2 \
  --node-rank 1
```

注意：`world_size = tp_size * pp_size`，多节点只是把这个 world 分布到多台机器。

### 4.2 内部实现

源码入口：

- `python/sglang/srt/distributed/parallel_state.py`
  - 创建 PP group。
- `python/sglang/srt/model_executor/model_runner.py`
  - 初始化时 `world_size=self.tp_size * self.pp_size`。
- `python/sglang/srt/managers/scheduler.py`
  - PP 下的 batch 启动、P2P 元数据处理。
- `docs/advanced_features/pipeline_parallelism.md`
  - 解释 micro-batching event loop、async P2P、多 stream、dynamic chunking。

### 4.3 约束

代码中明确校验：

- `pp_size > 1` 时不兼容 overlap schedule。
- `pp_size > 1` 时不兼容 speculative decoding。
- `moe_dp_size > 1` 时要求 `pp_size == 1`。
- Elastic EP 下要求 `pp_size == 1`。

### 4.4 何时学 PP

建议放到 TP、EP、DP attention 之后再学。PP 会引入调度器、micro-batch、P2P、dynamic chunking，心智负担明显更高。

## 5. Data Parallelism

SGLang 里 DP 有两种常见形态：native DP 和 SGLang Model Gateway DP。

### 5.1 Native DP

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4
```

含义：单个 SGLang 进程体系内启动多个 replica，由 `DataParallelController` 做简单负载均衡。

源码入口：

- `python/sglang/srt/managers/data_parallel_controller.py`
- `python/sglang/srt/entrypoints/engine.py`

局限：仓库文档明确不推荐生产使用 native DP；生产建议用 SGLang Model Gateway。

### 5.2 SGLang Model Gateway DP

推荐入口：

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

也可以先启动 worker，再单独启动 router：

```bash
python -m sglang.launch_server --model-path $MODEL --port 8000
python -m sglang.launch_server --model-path $MODEL --port 8001

python -m sglang_router.launch_router \
  --worker-urls http://node1:8000 http://node2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

文档入口：

- `docs/advanced_features/dp_dpa_smg_guide.md`
- `docs/advanced_features/sgl_model_gateway.md`

### 5.3 DP 和 TP 组合

例如 8 卡机器上 2 个副本，每个副本 TP=4：

```bash
python -m sglang_router.launch_server \
  --model-path $MODEL \
  --dp-size 2 \
  --tp-size 4
```

理解方式：

```text
总 GPU = dp_size * tp_size = 2 * 4 = 8
每个 DP replica 内部是一组 TP=4 的模型切分。
不同 replica 处理不同请求。
```

## 6. DP Attention / DPA

### 6.1 它解决什么

DPA 是 attention 侧的数据并行。它让不同 attention DP rank 处理不同 batch，并维护自己的 KV cache，减少 KV cache 重复。对 MLA 模型收益最大，但文档也说明标准 attention 的 Qwen 模型也支持。

### 6.2 启动方式

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

Qwen/Qwen MoE 学习时可从更保守配置开始：

```bash
python -m sglang.launch_server \
  --model-path $QWEN_MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto
```

### 6.3 代码约束

代码中 `_handle_data_parallelism()` 做了这些事：

- 如果 `dp_size == 1`，自动关闭 `enable_dp_attention` 和 `enable_dp_lm_head`。
- 如果启用 DPA，要求 `tp_size % dp_size == 0`。
- 如果启用 DPA，会把 `chunked_prefill_size` 除以 `dp_size`。
- `--enable-dp-lm-head` 要求同时启用 `--enable-dp-attention`。

另一个全局校验：

- 多节点 `dp_size > 1` 且没有启用 DPA 时不支持。

### 6.4 进程组公式

启用 DPA 后：

```text
attn_dp_size = dp_size
attn_tp_size = tp_size / (dp_size * attn_cp_size)
```

例子：

```text
tp=8, dp=8, attn_cp=1
attn_tp_size = 8 / (8 * 1) = 1

含义：
attention 不再做 8 卡 TP，而是 8 个 attention DP rank 各自处理自己的请求和 KV。
MoE 侧如果 ep=8，则 expert 分布在 8 卡上。
```

源码入口：

- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/layers/logits_processor.py`

## 7. Expert Parallelism

### 7.1 基础概念

EP 是 MoE 专用：把 expert 权重分布到多卡。token 经过 router/topk 后，会被 dispatch 到拥有对应 expert 的 rank，做 grouped GEMM，再 combine 回原 token 位置。

典型 MoE forward：

```text
hidden_states
  -> router / topk
  -> dispatcher.dispatch
  -> moe runner grouped GEMM
  -> dispatcher.combine
  -> final_hidden_states
```

源码入口：

- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/layers/moe/router.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `python/sglang/srt/layers/moe/token_dispatcher/*`
- `python/sglang/srt/layers/moe/moe_runner/*`
- `python/sglang/srt/layers/moe/utils.py`

### 7.2 启动方式

不指定 A2A backend 的 EP/TP 组合：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 2 \
  --moe-a2a-backend none
```

大规模 EP 常见 DeepEP：

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --deepep-mode auto
```

### 7.3 MoE A2A backend

当前代码里的 `MOE_A2A_BACKEND_CHOICES`：

| backend | 作用 | 备注 |
| --- | --- | --- |
| `none` | 默认，不使用专门 all-to-all backend | 支持 hybrid EP/TP |
| `deepep` | DeepEP token dispatch/combine | 代码会把 `ep_size` 调整为 `tp_size` |
| `mooncake` | Mooncake EP 通信 | 代码会把 `ep_size` 调整为 `tp_size` |
| `nixl` | NIXL-EP | 代码会把 `ep_size` 调整为 `tp_size` |
| `mori` | AMD MORI-EP | 代码会把 `ep_size` 调整为 `tp_size`，auto mode 会设为 normal |
| `ascend_fuseep` | Ascend NPU fused EP | 代码会把 `ep_size` 调整为 `tp_size` |
| `flashinfer` | FlashInfer A2A | 要求 `enable_dp_attention` 且 `dp_size == tp_size` |
| `megamoe` | Mega MoE | 代码会把 `ep_size` 调整为 `tp_size` |

重要约束：

- 文档说明 DeepEP/Mooncake/NIXL/Ascend/MORI 通常只支持 `ep_size = tp_size`。
- 代码中这些 backend 也会自动把 `ep_size` 设置成 `tp_size`。
- 如果你要学 hybrid EP/TP，例如 `tp=8, ep=2`，先用 `--moe-a2a-backend none`。

### 7.4 MoE runner backend

当前代码里的 `MOE_RUNNER_BACKEND_CHOICES`：

| backend | 作用 |
| --- | --- |
| `auto` | 根据模型、硬件、量化自动选择 |
| `deep_gemm` | DeepGEMM grouped GEMM，常见于 DeepEP + FP8 MoE |
| `triton` | Triton grouped GEMM |
| `triton_kernel` | Triton kernel 路径，代码限制 `ep_size == 1` |
| `flashinfer_trtllm` | FlashInfer + TensorRT-LLM MoE |
| `experimental_sgl_trtllm` | 实验路径 |
| `flashinfer_trtllm_routed` | 使用 SGLang topk 的 routed TRTLLM |
| `flashinfer_cutlass` | FlashInfer + CUTLASS |
| `flashinfer_mxfp4` | MXFP4 MoE |
| `flashinfer_cutedsl` | CuteDSL MoE，modelopt_fp4 场景 |
| `cutlass` | CUTLASS MoE |
| `aiter` | AMD AITER |
| `marlin` | Marlin MoE |

学习阶段建议：

1. 先用 `auto`。
2. 再固定 `triton`，读清楚通用 MoE runner。
3. 再看 `deep_gemm`，理解大 MoE 高性能路径。
4. 最后看 FlashInfer/CUTLASS/FP4 路径。

## 8. MoE TP、EP、MoE DP 的关系

核心公式：

```text
moe_tp_size = tp_size / (ep_size * moe_dp_size)
```

例子 1：纯 TP，不做 EP

```text
tp=8, ep=1, moe_dp=1
moe_tp_size = 8
```

含义：expert 不跨 EP 分布，MoE FFN 内部做 8 卡 TP。

例子 2：全 EP

```text
tp=8, ep=8, moe_dp=1
moe_tp_size = 1
```

含义：每个 expert 不再内部切 TP，而是 expert 按 8 卡分布。

例子 3：hybrid EP + MoE TP

```text
tp=8, ep=2, moe_dp=1
moe_tp_size = 4
```

含义：expert 分成 2 个 EP rank，同时每个 expert 计算还有 4 卡 MoE TP。注意此时通常只能用 `--moe-a2a-backend none`。

例子 4：EP + MoE DP

```text
tp=8, ep=4, moe_dp=2
moe_tp_size = 1
```

代码约束：

- `tp_size % moe_dp_size == 0`
- `ep_size * moe_dp_size <= tp_size`
- 如果 `ep_size > 1`，要求 `ep_size * moe_dp_size == tp_size`
- `moe_dp_size > 1` 时 `pp_size == 1`

## 9. Attention Context Parallel / Prefill CP

### 9.1 参数

基础大小参数：

```bash
--attn-cp-size N
```

prefill CP 开关：

```bash
--enable-prefill-cp \
--cp-strategy zigzag
```

`--cp-strategy` 可选：

- `zigzag`
- `interleave`

旧参数仍在代码里，但已经标为 deprecated：

- `--enable-dsa-prefill-context-parallel`
- `--enable-nsa-prefill-context-parallel`
- `--enable-prefill-context-parallel`
- `--dsa-prefill-cp-mode`

### 9.2 公式

```text
attn_tp_size = tp_size / (dp_size * attn_cp_size)
```

例子：

```text
tp=8, dp=2, attn_cp=2
attn_tp_size = 8 / (2 * 2) = 2
```

含义：在一个 TP 大组里，attention 被拆成 DP、CP、TP 三个维度。

### 9.3 约束

代码里要求：

- `tp_size % attn_cp_size == 0`
- `tp_size % (dp_size * attn_cp_size) == 0`
- `attn_cp_size > 1` 不支持 AITER allreduce fusion。
- `attn_cp_size != moe_dp_size` 时，只允许 `moe_dp_size == 1`。

`enable_prefill_cp` 会初始化 CP strategy，长上下文 prefill 场景再深入即可，不建议第一阶段就学。

## 10. EP 优化：TBO、SBO、EPLB

### 10.1 Two-Batch Overlap

启动：

```bash
--enable-two-batch-overlap
```

含义：把请求拆成 micro-batch，让 attention、dispatch、MLP、combine 的不同阶段交错执行，隐藏通信延迟。

约束：

- 代码要求启用 TBO 时 `moe_a2a_backend` 不能是 `none`。

源码入口：

- `python/sglang/srt/batch_overlap/*`
- `python/sglang/srt/layers/attention/tbo_backend.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
- `python/sglang/srt/layers/moe/utils.py`

### 10.2 Single-Batch Overlap

启动：

```bash
--enable-single-batch-overlap
```

含义：在一个 batch 内重叠通信和计算，例如 shared expert 计算与 dispatch/combine 重叠。

源码入口：

- `python/sglang/srt/batch_overlap/single_batch_overlap.py`
- `python/sglang/srt/layers/moe/token_dispatcher/base.py`
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
- `python/sglang/srt/layers/moe/moe_runner/runner.py`

### 10.3 EPLB

启动：

```bash
--enable-eplb
```

作用：统计 expert 负载，重新安排或冗余 expert，减少 expert 热点。

相关参数：

- `--ep-num-redundant-experts`
- `--ep-dispatch-algorithm`
- `--init-expert-location`
- `--eplb-algorithm`
- `--eplb-rebalance-num-iterations`
- `--enable-expert-distribution-metrics`

约束：

- 启用 EPLB 时要求 `ep_size > 1`。

## 11. PD Disaggregation

PD 不是模型内并行，而是部署级拆分：prefill 和 decode 用不同 server。

### 11.1 单机示例

```bash
# prefill
python -m sglang.launch_server \
  --model-path $MODEL \
  --disaggregation-mode prefill \
  --port 30000

# decode
python -m sglang.launch_server \
  --model-path $MODEL \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1

# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

### 11.2 MoE + DPA + PD 示例

```bash
# prefill worker
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode prefill \
  --trust-remote-code \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8

# decode worker
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode decode \
  --trust-remote-code \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

文档入口：

- `docs/advanced_features/pd_disaggregation.md`
- `docs/advanced_features/sgl_model_gateway.md`

## 12. EPD Disaggregation

EPD 是 VLM 场景：Encoder、Prefill、Decode 三段拆开。

常见参数：

- `--encoder-only`
- `--language-only`
- `--encoder-urls`
- `--encoder-transfer-backend`
- `--enable-mm-global-cache`

这个和 Qwen3.5 文本 MoE 主线关系不大。如果你后续学 Qwen3-VL，再回来看。

文档入口：

- `docs/advanced_features/epd_disaggregation.md`

## 13. 常见组合并行方式

### 13.1 单机 TP

```bash
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8
```

学习目标：理解线性层切分和 all-reduce/all-gather。

### 13.2 TP + PP

```bash
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 4 \
  --pp-size 2
```

学习目标：理解 `world_size = tp * pp`，以及层切分、P2P、micro-batch。

### 13.3 Router DP + TP

```bash
python -m sglang_router.launch_server \
  --model-path $MODEL \
  --dp-size 2 \
  --tp-size 4
```

学习目标：每个 replica 是 TP=4，router 负责请求分发。

### 13.4 TP + EP，A2A none

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --ep-size 2 \
  --moe-a2a-backend none
```

学习目标：hybrid EP/TP，理解 `moe_tp_size = 8 / 2 = 4`。

### 13.5 TP + EP + DeepEP

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-runner-backend auto
```

学习目标：理解 all-to-all dispatch/combine 和 MoE runner。

### 13.6 DPA + EP + DeepEP

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto
```

学习目标：attention 用 DP，MoE 用 EP，是大 MoE serving 的核心组合。

### 13.7 DPA + EP + FlashInfer A2A

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend flashinfer \
  --moe-runner-backend flashinfer_cutedsl
```

代码约束：

- `--moe-a2a-backend flashinfer` 要求 `--enable-dp-attention`。
- 要求 `dp_size == tp_size`。
- runner 只能是 `flashinfer_cutlass` 或 `flashinfer_cutedsl`。

### 13.8 TP + DP attention + CP

```bash
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8 \
  --dp-size 2 \
  --enable-dp-attention \
  --attn-cp-size 2 \
  --enable-prefill-cp \
  --cp-strategy zigzag
```

学习目标：理解 attention 的三维拆分：

```text
attn_tp_size = 8 / (2 * 2) = 2
attn_dp_size = 2
attn_cp_size = 2
```

### 13.9 PD + DPA + EP

```bash
# prefill 和 decode 分别启动，二者内部都可以配置：
--tp-size 16 \
--dp-size 8 \
--enable-dp-attention \
--moe-a2a-backend deepep
```

学习目标：理解部署层的 prefill/decode 解耦，不要把 PD 当成模型内部并行。

## 14. Qwen3.5 / MoE 学习建议

当前仓库 `docs/basic_usage/qwen3_5.md` 提到 Qwen3.5 具有 hybrid attention、MoE with shared experts、multimodal 能力。示例启动：

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B \
  --tp-size 8 \
  --trust-remote-code
```

实际学习时不要一开始就上最大配置。建议按这个顺序：

### 第 1 轮：只学启动和 TP

```bash
python -m sglang.launch_server \
  --model-path $QWEN_OR_SMALL_MOE_MODEL \
  --trust-remote-code \
  --tp-size 1

python -m sglang.launch_server \
  --model-path $QWEN_OR_SMALL_MOE_MODEL \
  --trust-remote-code \
  --tp-size 2
```

看日志里 server args、rank、world size、model load。

### 第 2 轮：学 MoE forward，但先不学 DeepEP

```bash
python -m sglang.launch_server \
  --model-path $QWEN_MOE_MODEL \
  --trust-remote-code \
  --tp-size 4 \
  --ep-size 1 \
  --moe-a2a-backend none \
  --moe-runner-backend triton
```

重点读：

- `topk.py`
- `fused_moe_triton/layer.py`
- `moe_runner/triton.py`
- `token_dispatcher/standard.py`

### 第 3 轮：学 hybrid EP/TP

```bash
python -m sglang.launch_server \
  --model-path $QWEN_MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 2 \
  --moe-a2a-backend none
```

目标：手算 `moe_tp_size = 8 / 2 = 4`，然后在代码和日志里验证。

### 第 4 轮：学 DeepEP 全 EP

```bash
python -m sglang.launch_server \
  --model-path $QWEN_MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-runner-backend auto
```

目标：理解 dispatch/combine 为什么需要 all-to-all。

### 第 5 轮：学 DPA + EP

```bash
python -m sglang.launch_server \
  --model-path $QWEN_MOE_MODEL \
  --trust-remote-code \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --deepep-mode auto
```

目标：理解 attention KV cache 为什么不再按 TP 重复，以及 MoE token 为什么仍然要跨 EP dispatch。

### 第 6 轮：学部署组合

再学：

- router DP
- PD disaggregation
- PP
- CP
- TBO/SBO
- EPLB

## 15. 源码阅读路线

建议按这个顺序读：

### 15.1 参数到进程组

1. `python/sglang/srt/server_args.py`
2. `python/sglang/srt/model_executor/model_runner.py`
3. `python/sglang/srt/distributed/parallel_state.py`
4. `python/sglang/srt/runtime_context.py`

要看懂的问题：

- CLI 参数在哪里被解析？
- 哪些参数会被自动改写？
- `tp_size`、`pp_size`、`dp_size`、`ep_size` 怎么传到 model runner？
- `initialize_model_parallel()` 里创建了哪些 group？

### 15.2 TP 层实现

1. `python/sglang/srt/layers/linear.py`
2. `python/sglang/srt/layers/vocab_parallel_embedding.py`
3. `python/sglang/srt/layers/logits_processor.py`
4. `python/sglang/srt/distributed/communication_op.py`

要看懂的问题：

- 哪些层切输出维？
- 哪些层切输入维？
- 哪些地方 all-reduce？
- 哪些地方 all-gather？

### 15.3 Attention / DPA / CP

1. `python/sglang/srt/layers/dp_attention.py`
2. `python/sglang/srt/layers/communicator.py`
3. `python/sglang/srt/layers/communicator_dsa_cp.py`
4. `python/sglang/srt/model_executor/forward_batch_info.py`
5. `python/sglang/srt/layers/attention/*`

要看懂的问题：

- `attn_tp_size` 和 `tp_size` 何时不同？
- DPA 下 logits/lm_head 怎么 gather？
- CP 只影响 prefill 还是 decode 也影响？

### 15.4 MoE

1. `python/sglang/srt/layers/moe/topk.py`
2. `python/sglang/srt/layers/moe/router.py`
3. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
4. `python/sglang/srt/layers/moe/token_dispatcher/base.py`
5. `python/sglang/srt/layers/moe/token_dispatcher/standard.py`
6. `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
7. `python/sglang/srt/layers/moe/moe_runner/base.py`
8. `python/sglang/srt/layers/moe/moe_runner/runner.py`
9. `python/sglang/srt/layers/moe/moe_runner/triton.py`
10. `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`

要看懂的问题：

- topk 输出的数据结构是什么？
- dispatch 前后 hidden_states 的 shape 怎么变？
- runner input/output 格式是什么？
- combine 如何恢复原 token 顺序？
- shared expert fusion 在哪里影响路径？

### 15.5 调度、DP、router

1. `python/sglang/srt/managers/scheduler.py`
2. `python/sglang/srt/managers/data_parallel_controller.py`
3. `python/sglang/srt/entrypoints/engine.py`
4. `sgl-model-gateway/`

要看懂的问题：

- 请求如何进入 scheduler？
- native DP 如何做负载均衡？
- router DP 和 native DP 的边界在哪里？

## 16. 参数组合速查

| 目标 | 推荐参数 |
| --- | --- |
| 单模型多卡 | `--tp-size N` |
| 模型太大，需要跨层 | `--tp-size M --pp-size K` |
| 多副本吞吐 | `python -m sglang_router.launch_server --dp-size N` |
| MoE expert 分布 | `--tp-size N --ep-size N --moe-a2a-backend deepep` |
| hybrid EP/TP | `--tp-size N --ep-size M --moe-a2a-backend none` |
| MLA/大 MoE decode 吞吐 | `--tp-size N --dp-size N --enable-dp-attention --ep-size N --moe-a2a-backend deepep` |
| 长上下文 prefill CP | `--attn-cp-size N --enable-prefill-cp --cp-strategy zigzag/interleave` |
| prefill/decode 分离 | `--disaggregation-mode prefill/decode` + router `--pd-disaggregation` |
| EP 通信计算重叠 | `--enable-two-batch-overlap` 或 `--enable-single-batch-overlap` |
| expert 负载均衡 | `--enable-eplb` |

## 17. 容易踩坑的规则

1. `tp_size` 在组合并行里经常是“大组大小”，不是纯 attention TP。
2. `--enable-dp-attention` 必须配 `--dp-size > 1`，否则代码会自动关闭。
3. DPA 要求 `tp_size % dp_size == 0`。
4. 多节点 native DP 不支持，除非启用 DPA。
5. DeepEP/Mooncake/NIXL/MORI/Ascend/FlashInfer/MegaMoE 这类 A2A backend 会把 `ep_size` 调整成 `tp_size`。
6. hybrid EP/TP 通常用 `--moe-a2a-backend none` 学。
7. `--moe-a2a-backend flashinfer` 要求 DPA 且 `dp_size == tp_size`。
8. `--enable-two-batch-overlap` 要求 `moe_a2a_backend != none`。
9. `pp_size > 1` 不兼容 overlap schedule 和 speculative decoding。
10. `moe_dp_size > 1` 不支持 PP。
11. `moe_dense_tp_size` 目前只支持 `None`、`1` 或 `tp_size`。
12. `--enable-dp-lm-head` 必须配 `--enable-dp-attention`。

## 18. 学习检查清单

每学一种配置，都建议记录：

```text
配置:
  tp_size =
  pp_size =
  dp_size =
  enable_dp_attention =
  attn_cp_size =
  ep_size =
  moe_dp_size =
  moe_a2a_backend =
  moe_runner_backend =

手算:
  attn_tp_size = tp_size / (dp_size * attn_cp_size)
  moe_tp_size = tp_size / (ep_size * moe_dp_size)

观察:
  日志里最终 ep_size 是否被自动改写？
  是否创建 TP/PP/attention/MoE group？
  MoE runner 最终选了哪个 backend？
  prefill/decode 的 batch 行为是否不同？
```

## 19. 建议学习顺序

### 第 1 周：启动链路和 TP

- 读 `server_args.py`、`model_runner.py`、`parallel_state.py`。
- 跑 `tp=1/2/4`。
- 看 `linear.py` 和 `vocab_parallel_embedding.py`。

### 第 2 周：MoE 基础

- 先用 `moe-a2a-backend none`。
- 读 `topk.py`、`fused_moe_triton/layer.py`、`standard.py`、`triton.py`。
- 画出 token -> expert -> combine 的数据流。

### 第 3 周：EP 和 DeepEP

- 跑 `ep=tp`。
- 读 `deepep.py` 和 `moe_runner/deep_gemm.py`。
- 理解 `deepep-mode normal/low_latency/auto`。

### 第 4 周：DPA + EP

- 跑 `tp=dp=ep`。
- 读 `dp_attention.py`、`communicator.py`、`logits_processor.py`。
- 理解 attention DP 与 MoE EP 如何在同一个 TP 大组内共存。

### 第 5 周：DP router、PD、PP

- 跑 router DP。
- 跑简单 PD。
- 再看 PP 和 dynamic chunking。

### 第 6 周：优化和 kernel

- TBO/SBO。
- EPLB。
- FlashInfer/CUTLASS/DeepGEMM/Triton kernel。
- profiling 和 benchmark。

## 20. 附录：diffusion runtime 的并行

仓库里 diffusion 相关 runtime 还有另一套并行：

- TP: `--tp-size`
- SP: `--sp-degree`
- Ulysses/Ring: `--ulysses-degree` / `--ring-degree`
- CFG parallel: `--enable-cfg-parallel`
- FSDP/offload/performance mode

文档入口：

- `docs/diffusion/api/cli.md`
- `docs/diffusion/performance/deployment_cookbook.md`
- `docs/diffusion/performance/cache/cache_dit.md`

这部分服务的是 diffusion/multimodal generation pipeline，不建议和 SRT LLM 的 TP/EP/DPA 混在一起学。

## 21. 关键参考文件

文档：

- `docs/advanced_features/server_arguments.md`
- `docs/advanced_features/expert_parallelism.md`
- `docs/advanced_features/dp_dpa_smg_guide.md`
- `docs/advanced_features/pipeline_parallelism.md`
- `docs/advanced_features/pd_disaggregation.md`
- `docs/advanced_features/epd_disaggregation.md`
- `docs/advanced_features/sgl_model_gateway.md`
- `docs/references/multi_node_deployment/multi_node.md`
- `docs/basic_usage/qwen3_5.md`

代码：

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/distributed/parallel_state.py`
- `python/sglang/srt/distributed/communication_op.py`
- `python/sglang/srt/layers/linear.py`
- `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `python/sglang/srt/layers/logits_processor.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/moe/`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/data_parallel_controller.py`
- `sgl-model-gateway/`

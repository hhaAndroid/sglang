# SGLang Prefill Context Parallel 源码学习笔记

> 研究对象：当前工作区 SGLang 源码，commit `6f005e4da1`（2026-08-14）  
> 本文中的“CP”默认指 **Prefill Context Parallel**，不是 Decode Context Parallel（DCP）。  
> 目标：先回答“为什么用、什么模型适合”，再沿当前源码讲清拓扑、数据布局、通信、后端与限制。

## 0. 先给结论

SGLang 的 Prefill CP 是一种面向长 prompt 的 **token/context 维并行**：同一个请求的 prefill token 被分给一个 CP group 内的多张卡，每张卡只计算自己负责的 query/hidden states，最后再恢复为全局 token 顺序。

它最适合：

- prompt 很长，TTFT 主要被 prefill attention/前向计算支配；
- 单卡 prefill activation 压力较大；
- CP rank 之间有高带宽互联（同机 NVLink/NVSwitch、对应平台高速互联）；
- 使用了 SGLang 已经接入 CP 的模型结构和 attention backend；
- 最理想是 PD 分离中的 prefill worker，因为可以避免 CP 拓扑拖累 decode。

它通常不适合：

- 短 prompt：all-gather、padding 和布局重排可能比节省的计算更贵；
- decode 占比高的 unified serving：Prefill CP 不会在 decode 阶段继续按 token 分摊工作，attention 权重还可能沿 CP 维复制；
- 普通低带宽跨机 CP：当前 DSA/MLA 自动配置明确限制 CP 的 TP group 在单机内，源码注释称跨机有精度问题；
- 任意“没有模型侧/attention backend 接入”的模型：仅设置 CLI 参数不代表一定可用。

最重要的实现事实是：

> SGLang 先切分的是 token/hidden，并不是先把一份已经完整的 K/V 切开。进入每个 Transformer 层后，各卡只根据自己的 local hidden 计算 local Q/K/V；此时新 K/V 还不完整，所以需要 all-gather。all-gather 完成后，每个 CP rank 才得到并保存完整 token 范围的 K/V，再用自己的 local Q 查询它。

用 `CP=2` 简化表示：

```text
GPU 0：一半 hidden -> 一半 Q/K/V --┐
                                     ├─ K/V all-gather -> 两卡都有完整 token 范围的 K/V
GPU 1：一半 hidden -> 一半 Q/K/V --┘

GPU 0：自己的一半 Q 查询完整 K/V
GPU 1：自己的一半 Q 查询完整 K/V
```

所以“每卡最终保存完整 K/V”是 all-gather 的结果，all-gather 正是把各卡刚算出的局部 K/V 补全的过程。这里的“完整”指覆盖完整 token 序列；如果还有 attention TP，每卡仍可能只持有自己的 attention head shard。

这与 ring attention 不同：ring attention 通常让局部 K/V 在卡间轮流传递、边接收边计算；SGLang 当前主路径则先在每个 CP rank 上收集出完整 token 范围的 K/V。

因此，普通 Prefill CP 的主要收益是：

1. 每卡 query 数量约降为 `1 / cp_size`；
2. 每卡 transformer body 的 token activation 约降为 `1 / cp_size`；
3. dense causal attention 的计算被均衡拆分；
4. 但每个 CP rank 通常都会实际收集完整 K/V，并把它们存进本卡的 KV cache，因此 KV 容量不会自动缩成 `1 / cp_size`。

GLM/DSA 的 `--enable-dsa-cache-layer-split` 是另一层优化：它再把 KV/indexer cache 的“层”分给 CP ranks，通过 owner broadcast/prefetch 临时读取远端层，才会显著降低每 rank 的 GPU KV cache 占用。

---

## 1. Prefill CP 到底解决什么问题

### 1.1 Prefill 与 decode 的计算形态不同

对长度为 `L` 的 causal prompt：

- prefill 一次处理大量 token；
- dense causal attention 近似是 `O(L^2)`；
- projection、MLP/MoE 等近似是 `O(L)`；
- activation 随当前 prefill chunk 的 token 数增长。

decode 每步通常每个请求只增加一个 token，它的瓶颈更偏向 KV cache 读取、batch 并发和小 GEMM。Prefill CP 只针对前一种形态。

源码中 `ForwardMode.is_context_parallel_extend()` 只把 `EXTEND`、`MIXED`（以及可选的 draft extend）看作 CP extend，普通 `DECODE` 不属于 Prefill CP：[forward_batch_info.py](../python/sglang/srt/model_executor/forward_batch_info.py#L98)。

### 1.2 收益来自哪里

设同一 PP stage 的并行 world size 为 `tp_size`，CP degree 为 `C`。

对一条长序列，CP 后每个 rank 大致只保留、计算 `L/C` 个 query token。dense attention 的理想单 rank 计算量从约 `L^2/2` 降到约 `L^2/(2C)`。真实速度不会线性提升，因为还存在：

- 每层 K/V all-gather；
- padding 和 token 重排；
- attention 与 MoE/FFN 不同并行布局之间的 gather/reduce-scatter；
- 最后一层 hidden state gather；
- kernel launch、metadata 构造和同步开销。

所以 CP 的核心判断不是“模型支持长上下文”，而是：

```text
节省的长序列计算与 activation 成本
    > K/V 通信 + 布局转换 + padding + decode 侧副作用
```

SGLang 当前只用很低的正确性门槛决定能否切分：zigzag 至少 `2*C` 个 token，interleave 至少 `C` 个 token。它不会自动替用户判断性能 break-even；上线前必须按实际 prompt length、chunk size、batch 和硬件测 TTFT/吞吐。

### 1.3 与 chunked prefill 的关系

两者解决的问题不同，但可以同时使用：

- chunked prefill：限制一次 forward 处理的全局 token 数，控制调度公平性和峰值 activation；
- Prefill CP：把当前 forward/chunk 的 token 再分给多个 CP ranks。

chunk 太小会让每个 rank 的 local token 太少，通信占比升高；chunk 很大时 CP 更容易发挥计算并行与 activation 分摊的价值。DeepSeek V4 的启动校验会按 `attn_cp_size` 推导 local chunk 上界：[deepseek_v4_hook.py](../python/sglang/srt/arg_groups/deepseek_v4_hook.py#L40)。

### 1.4 DSA 原生语义预备知识：以 Transformers 的 GLM-5.2 为例

后文会频繁出现 Indexer、top-k indices、Indexer K cache 和 DSA backend。先用 `/mnt/shared-storage-user/huanghaian/code/transformers` 中的 `GlmMoeDsa` reference 实现建立模型原生语义。该源码当前 commit 为 `a61d5f9e4f`；它用于说明“模型要算什么”，不代表 SGLang 实际采用相同的 eager kernel 和 cache 布局。

#### 一个 DSA attention layer 有两套相关但不同的 Q/K

可以先把一层画成两条支路：

```text
hidden states
    │
    ├─ Indexer 支路：轻量 Q_index / K_index
    │      └─ 为每个 query 选出 top-k token positions
    │
    └─ 主 MLA/Attention 支路：Q_main / K_main / V_main
           └─ 只在 Indexer 选中的 positions 上执行真正 attention
```

Indexer 不是主 attention 的 Q/K projection 换个名字。GLM-5.2 给它单独定义了：

- `wq_b`：从 `q_resid` 产生多头 `Q_index`；
- `wk` + `k_norm`：从 hidden 产生单份 `K_index`；
- `weights_proj`：为 Indexer 的多个 query heads 产生聚合权重；
- `index_topk`：每个 query 最终保留多少个 token position，配置默认值是 2048；
- 实现见 [GlmMoeDsaIndexer](../../../transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L164)，默认配置见 [configuration_glm_moe_dsa.py](../../../transformers/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py#L126)。

#### Indexer 如何得到 top-k positions

Transformers reference `forward()` 的逻辑可以简化为：

```python
q_index = wq_b(q_resid)                 # [B, S, H_index, D_index]
k_index = norm(wk(hidden_states))        # [B, S, D_index]
q_index, k_index = apply_rope(...)

k_index = indexer_k_cache.update(k_index)

score_per_head = relu(q_index @ k_index.T / sqrt(D_index))
head_weights = weights_proj(hidden_states)
index_scores = weighted_sum(score_per_head, head_weights)  # [B, S, T]
index_scores += causal_mask
topk_indices = topk(index_scores, K)                       # [B, S, K]
```

对应源码集中在 [Indexer.forward](../../../transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L201)。几个关键点：

1. Indexer 会为每个 query 对候选历史 `K_index` 打分，并显式应用 causal mask，不能选到未来 token；
2. Indexer 只输出 token positions，即 `topk_indices`，它不负责产生 attention output；
3. 它有自己的 **K cache**，但没有 Indexer V cache；
4. 主 attention 另有真正的 K/V cache，二者不要混为一谈。

Transformers 的 cache 类把这件事写得很直白：普通 main K/V cache 之外，`DynamicIndexedLayer` 再维护形状为 `[batch, seq_len, index_head_dim]` 的单头 Indexer K cache：[cache_utils.py](../../../transformers/src/transformers/cache_utils.py#L319)。

#### 为什么 Indexer 没有 V

普通 attention 要计算：

```text
softmax(QK^T) × V -> attention output
```

Indexer 只做检索：

```text
Q_index K_index^T -> causal mask -> top-k token positions
```

它不产生加权后的 hidden output，所以不需要 `V_index`。但“没有 V”不等于没有 cache：历史 `K_index` 必须缓存，否则 decode 的新 query 无法检索完整历史。

#### top-k 如何进入真正的 DSA attention

GLM-5.2 的主 attention 仍是 MLA 结构：先生成主 `query_states` 和压缩 KV，再在 reference 路径里展开 `key_states/value_states` 并更新主 K/V cache。随后：

- `full` Indexer layer 自己计算 `topk_indices`；
- `shared` layer 复用上一个 full Indexer layer 的 `topk_indices`，减少每层都跑 Indexer 的开销；
- 该模式由 `config.indexer_types[layer_idx]` 控制；
- 跨层传递逻辑见 [GlmMoeDsaAttention](../../../transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L311) 和 [模型层循环](../../../transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L745)。

Reference 实现提供两种表达同一语义的方式：

```text
eager / SDPA：
  把 topk_indices scatter 成 mask
  选中位置 mask=0，其余位置 mask=-inf
  再调用普通 attention

支持 sparse indices 的优化 backend：
  直接把 topk_indices 作为 indices 参数传给 attention kernel
  kernel 只读取被选中的 K/V
```

分支位于 [GlmMoeDsaAttention.forward](../../../transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L389)。Transformers 的 eager/SDPA 分支更适合验证语义，不代表它真的省掉了 dense score matrix；推理引擎要获得 DSA 性能，必须让优化 kernel 按 indices 稀疏读取 K/V。

#### 从原生语义映射到 SGLang Prefill CP

理解后文时，可以把 SGLang 的优化路径对应成：

```text
Transformers reference                   SGLang Prefill CP / DSA
──────────────────────────────────       ─────────────────────────────────
每层产生 Q_index / K_index               每 rank 只产生 local Q_index/K_index
Indexer K cache 覆盖完整历史              all-gather local K_index 得到完整 key 视图
每个 Q 计算 causal top-k                 只为本 rank 的 local Q 计算 top-k
主 attention 持有 K_main/V_main          local MLA K/V all-gather 后形成完整 cache
按 topk_indices 做 sparse attention       DSA backend 按 indices 为 local Q 读 K/V
```

因此 Interleave 能处理离散 Q 的根本原因是：每个 local Q 都保留原始 position，并得到自己独立的 top-k indices；DSA kernel 按显式 indices 访问主 K/V，不要求相邻 Q 连续。与此同时，top-k 之前的 Indexer 仍要搜索 causal 历史，这正是后文讨论连续布局与 Interleave 负载差异时需要单独关注的阶段。

---

## 2. 哪些模型更适合

### 2.1 从模型结构看

| 模型结构 | 适合度 | 原因与注意点 |
|---|---:|---|
| Dense MHA | 高（长上下文时） | attention 计算重，CP 可直接分 query；但 K/V all-gather 数据量也大 |
| GQA/MQA | 通常更好 | K/V head 较少，full KV 通信量通常小于 MHA，query 计算仍可分摊 |
| MLA | 适合但路径更专用 | latent KV 更紧凑，有利于降低 KV 通信；当前 DeepSeek V3 MLA CP 有自动配置和 backend 限制 |
| DSA/稀疏注意力 | 适合超长上下文 | 每 rank 负责部分 query，仍需完整 indexer key/KV 视图；interleave 路径专门处理 top-k/indexer metadata |
| MoE | 可用，但并行布局更复杂 | attention token shard 需要与 EP/MoE DP 协同，可能增加 gather/scatter；必须看模型与 MoE backend 的已验证组合 |
| Hybrid linear attention/SSM | 不能按 dense attention 直觉推断 | recurrent/linear state 的切分语义不同，必须有模型侧接入；不要只凭 `attn_cp_size` 强开 |
| 多模态 | 谨慎 | 文本 body 可以接入不代表 vision/cross-attention 输入布局已支持；例如 MiMo V2 CP-v2 明确只允许 text inference |

### 2.2 当前源码会自动启用 CP-v2 的模型类

`CP_V2_DEFAULT_MODEL_CLASSES` 当前包含：[cp/utils.py](../python/sglang/srt/layers/cp/utils.py#L43)

- `DeepseekV32ForCausalLM`
- `GlmMoeDsaForCausalLM`
- `GptOssForCausalLM`
- `MiMoV2FlashForCausalLM`
- `MiMoV2ForCausalLM`
- `Qwen3MoeForCausalLM`
- `DeepseekV3ForCausalLM`

这张表的准确含义是“命中时，ServerArgs 可以默认打开 CP-v2 环境开关”，不是所有可运行模型的完整白名单。DeepSeek V4 也有显式 CP-v2 CI，但通常由测试/部署显式设置 `SGLANG_ENABLE_CP_V2=1`。

MiMo V2 还有额外校验：

- 只支持 `zigzag`；
- 多模态 checkpoint 必须加 `--language-only` 或 `--language-model-only`；
- 见 [server_args.py](../python/sglang/srt/server_args.py#L6425)。

### 2.3 当前仓库端到端 CI 给出的“强证据”

比接口枚举更可信的是注册的 E2E/accuracy tests：

| 模型/结构 | 策略与典型配置 | 硬件/后端证据 | 测试源码 |
|---|---|---|---|
| Qwen3-30B-A3B-FP8，GQA MoE | zigzag，CP2/CP4，可配 EP/MoE-DP | 4×H100，FA3 显式覆盖 | [test_gqa_prefill_cp.py](../test/registered/cp/test_gqa_prefill_cp.py#L16) |
| GPT-OSS 120B MXFP4 | zigzag，CP4=TP4 | 4×B200，TRTLLM MHA + breakable prefill graph | [test_gpt_oss_4gpu_mxfp4_cp.py](../test/registered/cp/test_gpt_oss_4gpu_mxfp4_cp.py#L9) |
| DeepSeek V3 MLA | 旧名 in-seq/当前名 zigzag，TP8/DP2/CP4 | 8×H200，FA3 | [test_deepseek_v3_cp_single_node.py](../test/registered/cp/test_deepseek_v3_cp_single_node.py#L18) |
| DeepSeek V3.2 DSA | legacy zigzag 与 interleave 都有 | 8×H200 | [test_dsa_prefill_cp_legacy.py](../test/registered/cp/test_dsa_prefill_cp_legacy.py#L17) |
| GLM-5.2 DSA | CP-v2 interleave，CP8 | 8×H200，带 EAGLE | [test_dsa_prefill_cp.py](../test/registered/cp/test_dsa_prefill_cp.py#L17) |
| DeepSeek V4 Flash FP4 | CP-v2 interleave，DeepEP/MegaMoE/非 A2A，多种 spec | 4×B200 | [test_deepseek_v4_flash_fp4_b200_cp.py](../test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py#L27) |
| DeepSeek V4 Pro FP4 | interleave | 8×MI35x，HIP `dsv4`/unified-kv 路径 | [test_deepseek_v4_pro_fp4_cp.py](../test/registered/amd/test_deepseek_v4_pro_fp4_cp.py#L29) |
| Qwen3 MoE PP×CP | PP2 × CP2 | 4×H100 | [test_pp_parallel_compat.py](../test/registered/pp/test_pp_parallel_compat.py#L77) |
| GLM-5.2 DSA LayerSplit | PD prefill CP4 + decode TP4 | 8×B200，TRTLLM DSA prefill | [test_dsa_glm52_cache_layer_split.py](../test/registered/models_e2e/test_dsa_glm52_cache_layer_split.py#L26) |

因此，如果要选择学习样例：

1. 通用 dense/GQA 路径：先看 Qwen3 MoE + zigzag + FA3；
2. Blackwell MHA 和 prefill CUDA graph：看 GPT-OSS + zigzag + TRTLLM MHA；
3. 稀疏长上下文：看 GLM-5.2 + interleave + DSA；
4. 最新复杂路径：再看 DeepSeek V4 + interleave。

---

## 3. CP、TP、DP、EP、PP、DCP 的边界

### 3.1 attention 的三维分解

初始化并行组时，源码计算：

```python
attn_tp_size = tp_size // attn_cp_size // attn_dp_size
```

见 [parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2451)。因此：

```text
tp_size = attn_dp_size × attn_cp_size × attn_tp_size
```

可以把同一 PP stage 内的 rank 看成三维网格：

```text
attention DP：不同请求副本
attention CP：同一请求的不同 token/context shard
attention TP：同一 token 的不同 attention head/weight shard
```

CP 增大时，在固定 `tp_size` 下 `attn_tp_size` 会减小。attention 权重沿 attn-TP 分片、沿 CP 维复制。常见 DSA/MLA 自动配置令：

```text
attn_cp_size = tp_size / dp_size
attn_tp_size = 1
```

这意味着 prefill attention 用所有这些 rank 分 token，但每个 CP rank 拿到完整 attention 权重。

### 3.2 一个 TP8、CP2 的 rank group 例子

当 `tp_size=8, attn_dp_size=1, attn_cp_size=2`：

```text
attn_tp_size = 8 / 2 = 4

ATTN_TP groups:
  [0,1,2,3]
  [4,5,6,7]

ATTN_CP groups:
  [0,4]
  [1,5]
  [2,6]
  [3,7]
```

构组算法在 [parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2459)，对应单测在 [test_parallel_state.py](../test/registered/unit/distributed/test_parallel_state.py#L52)。

### 3.3 Prefill CP 与 DCP 不是同一功能

| 项目 | Prefill CP | Decode CP / DCP |
|---|---|---|
| 参数 | `--attn-cp-size` + `--enable-prefill-cp` | `--dcp-size`、`--dcp-comm-backend` |
| 分片对象 | 当前 prefill 的 query/hidden token | 已存在的长 KV context / decode attention 工作 |
| 目标 | 降 TTFT、分摊 prefill activation/compute | 降长 context decode 的 KV 读取/attention 开销 |
| KV cache 布局 | all-gather 后，每个 CP rank 通常保存完整 token 范围 | 每个 DCP rank 只保存约 `1/DCP` 的 token 位置 |
| Prefix Cache 容量 | 不会因为 Prefill CP 自动扩大 | 逻辑容量可随 DCP 扩大，减少容量不足导致的 prefix 淘汰 |
| 主要通信 | KV/hidden all-gather、布局转换 | AG+RS、A2A 或 FlashInfer A2A reduction |

这里先补一个理解 DCP 的背景。对 GQA/MQA，单卡 KV head 数近似为：

```text
max(1, num_kv_heads / attention_tp_size)
```

当 KV heads 比 TP ranks 少时，KV 已无法继续沿 head 维切分，每个 rank 至少要保留一个 KV head，于是同一个 KV head 会在多个 TP ranks 上复制。MLA 虽不是普通的多 KV head 形式，但共享的 latent KV 沿 TP ranks 复制时有类似问题。源码计算见 [model_config.py](../python/sglang/srt/configs/model_config.py#L1133)。

例如 MQA/MLA 只有一份 KV 表示，而模型因为权重容量或 GEMM 仍需使用 TP8：

```text
普通 TP8：
  8 个 rank 都保存这一份 KV 的全部 token
  -> KV 沿 TP 复制 8 份

TP8 + DCP8：
  rank0 保存 token 0, 8, 16, ...
  rank1 保存 token 1, 9, 17, ...
  ...
  rank7 保存 token 7, 15, 23, ...
  -> 每卡约保存 1/8 token，整个 group 合起来是一份完整 KV
```

因此更准确的思路不是“KV heads 少，所以完全不开 TP”：大模型可能仍然需要 TP 切权重、Q heads 和计算。DCP 做的是把 TP ranks 之间原本浪费的 KV 副本，改造成按 token/context 分片的有效容量。SGLang 的 DCP cache write 会按 `position % dcp_size == dcp_rank` 选择本 rank 的 token，再映射到本地物理位置：[triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L1256)。

DCP group 的逻辑 cache 空间可以约为单 rank 物理空间的 `dcp_size` 倍：[pool_host/base.py](../python/sglang/srt/mem_cache/pool_host/base.py#L334)。它能保留更多 prefix、减少因容量不足发生的 eviction，因此在工作负载确实有重复 prefix、且原先命中率受容量限制时，可能提高 Prefix/Radix Cache 命中率；它不会在没有共享 prefix 的流量中凭空提高命中率。

而本文研究的 Prefill CP 不做这种长期 KV token sharding：它切分 prefill query/hidden 计算，随后把本层 local K/V all-gather 成完整 token 范围并存入各 CP rank。二者不要因都叫 context parallel 而混在一起。

这里只建立概念边界，不代表所有 GQA/MLA 模型和 attention backend 都已支持 DCP。当前 SGLang 的 DCP、HiCache 和 speculative decoding 仍有各自的模型/backend 兼容限制，学习时应单独核对。

---

## 4. 两种 token 布局策略

当前统一 CLI：

```bash
--enable-prefill-cp
--cp-strategy zigzag        # 旧名 in-seq-split
# 或
--cp-strategy interleave    # 旧名 round-robin-split
```

strategy 抽象定义在 [cp/base.py](../python/sglang/srt/layers/cp/base.py#L39)，初始化在同文件 [init_cp_strategy](../python/sglang/srt/layers/cp/base.py#L248)。

### 4.1 Zigzag：解决 causal attention 负载不均

如果简单把 causal 序列连续切成 `C` 段：

- 前段 query 只能看较短前缀，计算少；
- 后段 query 能看很长前缀，计算多；
- 最后一个 rank 会明显更慢。

zigzag 把每条序列切成 `2*C` 个连续 block，每个 rank 拿一个早期 block 和一个对称的晚期 block。

当 `C=4`：

```text
自然顺序：b0 b1 b2 b3 b4 b5 b6 b7

rank0: b0 + b7
rank1: b1 + b6
rank2: b2 + b5
rank3: b3 + b4
```

因为早期轻 block 与晚期重 block 配对，每 rank 的 causal attention 工作更接近。实现和文件头示意在 [cp/zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L15)。

metadata 的关键字段：

- `split_list`：每条序列拆成 `2*C` 段后的长度；
- `zigzag_index`：当前 rank 应取哪些段；
- `cp_reverse_index`：all-gather 后恢复自然 token 顺序；
- `per_rank_actual_token` / `per_rank_logical_token`：物理 padding 长度与真实长度；
- `kv_len_prev/next`、`cu_seqlens_q_prev/next`：早、晚两个 query block 各自正确的 causal KV 可见范围。

构造过程见 [ZigzagCPStrategy.build_metadata](../python/sglang/srt/layers/cp/zigzag.py#L121)。

attention 执行时：

- FA3/FA4 路径对 prev 和 next 各调用一次 attention kernel；
- TRTLLM MHA 可以把两部分 geometry 合并成一次调用，并使用 zigzag page table；
- 见 [ZigzagCPStrategy.run_attention](../python/sglang/srt/layers/cp/zigzag.py#L343)。

Zigzag 最低要求是每条 extend sequence 至少 `2*C` token：[can_apply](../python/sglang/srt/layers/cp/zigzag.py#L109)。

### 4.2 Interleave：按 token index 轮转

当 `C=4`：

```text
rank0: t0, t4, t8,  t12, ...
rank1: t1, t5, t9,  t13, ...
rank2: t2, t6, t10, t14, ...
rank3: t3, t7, t11, t15, ...
```

实现见 [cp/interleave.py](../python/sglang/srt/layers/cp/interleave.py#L15)。它的特点是：

- token 数天然均匀；
- 非整除时各 rank 长度最多差 1，再 padding 到统一物理长度；
- all-gather 后用 `global_index -> rank-major buffer index` 映射恢复自然顺序；
- DSA 对 multi-request 的 local q lens / batch index 有专门 Triton kernel；
- 当前 CP-v2 的 interleave strategy 只声明支持 DSA backend。

> **当前实现约束：Interleave CP-v2 不能直接配 FA3/FA4 或 TRTLLM MHA。** 这不是说离散 Q 在数学上无法做 dense attention，而是 SGLang 当前没有为标准 FlashAttention causal 接口实现一套高效的 Interleave CP dispatch。Interleave strategy 当前只注册 `DSA`：[get_supported_attention_backend](../python/sglang/srt/layers/cp/interleave.py#L214)。

恢复顺序的核心公式：

```python
gather_index = (global_index % cp_size) * physical_rank_len \
             + (global_index // cp_size)
```

见 [_gather_interleaved_tensor](../python/sglang/srt/layers/cp/interleave.py#L172)。

### 4.3 为什么 DSA 更偏向 interleave

DSA 的每个 query 通常经历 Indexer/top-k 和稀疏 attention。top-k 之后的 DSA attention 每个 Q 最多只读取约 `index_topk` 个 K/V，不再具有持续增长的 dense causal 三角形工作量；但 top-k 之前的 Indexer 仍要在 causal 历史中搜索候选 K。Interleave 既均分 query token 数，也把早、中、晚位置的 Q 分散到各 rank，主要用于避免 Indexer 的位置分布与 rank 强绑定。DSA Indexer 会 all-gather 完整 key 视图，然后每个 rank 只为 local query 做 top-k/attention。

当前 CP-v2 对 DSA 默认只开启 interleave；DSA zigzag 仍主要走 CP-v1 legacy 路径。这个选择逻辑在 [server_args.py](../python/sglang/srt/server_args.py#L6433)。

> 学习当前 CP-v2 主线时，应把 `DSA -> Interleave` 当成明确的 backend/strategy 配对，而不是只按个人偏好在两种布局之间任选。

### 4.4 Zigzag 与 Interleave 的本质区别和实现区别

先看共同点。两种策略都不会在执行 attention 前把 Q all-gather 回完整序列；它们都是：

```text
hidden / positions / Q：按 CP rank 分片
K/V：all-gather 后恢复为完整 token 顺序
attention output：继续保持 local Q 的排列
最后一层 hidden：all-gather 并恢复全局 token 顺序
```

因此两者的区别不是“能否看到完整 K/V”，而是：

1. 每个 rank 负责哪些 Q；
2. 如何告诉 attention kernel 每个 Q 的 causal 可见范围；
3. 如何适配 dense attention 与 DSA 不同的计算形态；
4. all-gather 后如何恢复 token 顺序。

用长度 8、`CP=2` 的序列对比：

| 策略 | rank 0 的 Q | rank 1 的 Q |
|---|---|---|
| Zigzag | `Q0,Q1,Q6,Q7` | `Q2,Q3,Q4,Q5` |
| Interleave | `Q0,Q2,Q4,Q6` | `Q1,Q3,Q5,Q7` |

两种情况下，每个 rank 都能得到自然顺序的完整 `K0...K7` 和 `V0...V7`。

#### Interleave 的 Q 被打散后如何计算 attention

给定完整 K/V 后，每一个 query row 可以独立计算：

```text
O_i = Attention(Q_i, K_0...K_i, V_0...V_i)
```

所以 Q 在显存里不连续不会改变数学语义。Interleave 会把原始 position 与 hidden 一起切分：[shard_position_ids](../python/sglang/srt/layers/cp/interleave.py#L93)。例如 rank 0 拿到：

```text
local Q         = Q0, Q2, Q4, Q6
local positions = 0,  2,  4,  6

Q0 只允许看 K/V[0]
Q2 只允许看 K/V[0:3]
Q4 只允许看 K/V[0:5]
Q6 只允许看 K/V[0:7]
```

原始 position 用于 RoPE；每个 Q 对应的 causal length、batch 归属和 top-k indices 用于限制其可见 KV。local output 仍为 `O0,O2,O4,O6`，最后 gather 后按全局 index 交错恢复成 `O0,O1,...,O7`。

这对 DSA 很自然：DSA 本来就为每个 Q 生成一组显式的 top-k KV indices。Indexer 先 all-gather 完整 key 视图，然后只为 local Q 计算 top-k；稀疏 attention 再按这些 indices 读取合法的历史 K/V。它不要求相邻 Q 一起出现。

源码实现上：

- `_interleave_shard()` 直接取 `rank, rank+C, rank+2C, ...` 的 token；
- `shard_position_ids()` 以相同方式保留这些 token 的原始位置；
- `_gather_interleaved_tensor()` 使用 `global_index % C` 和 `global_index // C` 从 rank-major buffer 恢复自然顺序；
- strategy 的 `run_attention()` 是 no-op，因为真正的 indexer/top-k/attention 由 DSA backend 自己执行：[interleave.py](../python/sglang/srt/layers/cp/interleave.py#L220)。

#### 为什么普通 dense attention 不直接使用 Interleave

Interleave rank 0 的四个 Q 对应不同 causal KV 长度：

```text
Q rows       = [Q0, Q2, Q4, Q6]
visible lens = [ 1,  3,  5,  7]
```

数学上可以通过显式 mask、把每个 Q 当成长度 1 的独立 sequence，或者定制 kernel 来计算。但标准 FlashAttention causal 接口更擅长处理连续 Q block，并通过 `q_len/kv_len` 的对齐关系隐式生成 causal mask。把每个离散 Q 都拆成独立小任务，会增加 metadata、调度和 kernel 开销。

Zigzag 保留了连续 block。上面的 rank 0 可以执行：

```text
Q[0,1] 对 K/V[0:2]：一次连续 causal attention
Q[6,7] 对 K/V[0:8]：一次连续 causal attention
```

在第二次调用中，causal kernel 会让 `Q6` 看到 `K0...K6`、`Q7` 看到 `K0...K7`。FA3/FA4 路径分别对 early/late block 调用 attention；TRTLLM MHA 则可以把两组 geometry 合并：[ZigzagCPStrategy.run_attention](../python/sglang/srt/layers/cp/zigzag.py#L343)。

#### 为什么两种策略采用不同的负载均衡方法

Dense causal attention 中，位置越晚的 Q 能看到的 K/V 越多，工作量近似随 causal 三角形增长。若只按连续段平均 token 数，后段 rank 会更慢。Zigzag 用“早期轻 block + 晚期重 block”配对，使各 rank 的总可见 KV 数更接近。

DSA 要拆成两个阶段看：Indexer 在 top-k 之前仍有位置相关的候选历史；真正的 sparse attention 在 top-k 之后，每个 Q 的工作量大致受 `index_topk` 限制，通常已经比较均衡。Interleave 主要用于均分 local Q 并打散 Indexer 的 Q position，而不是为了平衡后段 DSA attention 的完整 causal 三角形。

#### 当前 CP-v2 的选择规则

> **最重要的实现结论：Zigzag 能接 FA3/FA4 和 TRTLLM MHA；Interleave 当前不能接这些 dense attention backend，只接 DSA。Interleave 数学上并非不能计算 dense attention，缺的是当前 SGLang 中适配离散 Q causal 边界的高效 FA/TRTLLM CP 实现。**

| 模型/attention 路径 | 推荐/要求的策略 | 原因 |
|---|---|---|
| DSA CP-v2 | Interleave | 显式 top-k indices 支持离散 Q，按 Q 数量均衡 |
| FA3/FA4 dense MHA/GQA/MLA | Zigzag | 连续 Q block 可直接复用 causal FlashAttention |
| TRTLLM MHA | Zigzag | 使用 prev/next 或 combined zigzag geometry |
| 其他模型/backend | 不能默认任选 | 必须先确认模型和 backend 已接入 Prefill CP |

学习当前主线时，可以记成：

> CP-v2 中，Interleave 基本是为 DSA 这种带显式稀疏索引的 attention 设计的；Zigzag 是已接入的 dense causal attention 主路径。也就是 `DSA -> Interleave`，`FA3/FA4/TRTLLM MHA -> Zigzag`。Legacy CP-v1 的 DSA Zigzag 是历史例外，不能据此推导 CP-v2 的支持关系。

### 4.5 DSA 为什么不用朴素连续布局

这里的“连续布局”是指把一条序列只切成 `C` 个连续块，每个 rank 拿一个块；它不同于 Zigzag 的“每 rank 一个早期块 + 一个晚期块”。例如长度 16、`CP=4`：

```text
朴素连续切分：
  rank0: Q0,  Q1,  Q2,  Q3
  rank1: Q4,  Q5,  Q6,  Q7
  rank2: Q8,  Q9,  Q10, Q11
  rank3: Q12, Q13, Q14, Q15

Interleave：
  rank0: Q0, Q4, Q8,  Q12
  rank1: Q1, Q5, Q9,  Q13
  rank2: Q2, Q6, Q10, Q14
  rank3: Q3, Q7, Q11, Q15
```

#### 连续布局在数学上没有问题

只要每个 rank 都有完整 Indexer K、主 K/V、正确的原始 position 和 causal metadata，连续布局同样可以得到正确结果。它甚至还有实现简单、local Q 内存连续、all-gather 后容易直接拼接的优点。

所以 SGLang 没选择它不是因为模型语义不允许，而是因为它让 `rank id` 与 `Q position` 强相关：低 rank 总拿早期 Q，高 rank 总拿晚期 Q。

#### 主要潜在不均衡在 Indexer，不在稳定阶段的 sparse attention

对位置 `i` 的 Q：

```text
Indexer 的合法候选历史量：约 i + 1
DSA sparse attention 的有效 K/V 数：约 min(i + 1, index_topk)
```

仍以长度 16、`CP=4` 为例，把每个 Q 的 causal 候选数量相加：

```text
朴素连续：
  rank0:  1 +  2 +  3 +  4 = 10
  rank1:  5 +  6 +  7 +  8 = 26
  rank2:  9 + 10 + 11 + 12 = 42
  rank3: 13 + 14 + 15 + 16 = 58

Interleave：
  rank0: 1 + 5 +  9 + 13 = 28
  rank1: 2 + 6 + 10 + 14 = 32
  rank2: 3 + 7 + 11 + 15 = 36
  rank3: 4 + 8 + 12 + 16 = 40
```

这组数字表达的是逻辑上的有效候选范围，不是对某个具体 kernel 耗时的精确预测。Indexer kernel 可能计算 padding 后的矩形 logits，或者利用 `ks/ke` 跳过部分无效范围；实际收益取决于实现。但 Interleave 至少不会把所有最晚、最长 causal history 的 Q 集中在同一个 rank。

top-k 之后则不同。序列位置超过 `index_topk` 后，每个 Q 通常只让真正的 DSA attention 读取约 `index_topk` 个 K/V，因此 sparse attention 阶段不会像 dense attention 那样随位置持续变重。它仍可能因短 prefix、`-1` padding、page 对齐和不同请求长度存在小幅差异，但主要位置相关的不均衡来自 Indexer。

#### Multi-batch 下 Interleave 更稳健

朴素按全局扁平 token buffer 连续切分，还可能让某些 rank 主要拿到一个长请求、另一些 rank 拿到多个短请求。即使总 token 数相近，请求数、prefix 长度、top-k metadata 和后续 MoE 路由也可能不同。

当前 Interleave 为多请求专门计算每个 rank 的 local q length 和有效 batch index，长度不能整除时各 rank token 数最多差 1：[InterleaveCPStrategy.shard_per_request](../python/sglang/srt/layers/cp/interleave.py#L120)。它让 local Q 数量以及来自不同序列的位置分布更稳定。理论上也可以设计“逐请求连续切块”的第三种策略，但需要另一套 metadata/backend 接入，当前源码没有提供。

#### 为什么这里不用 Zigzag 代替 Interleave

Zigzag 对 dense causal work 的平衡更精确，并保留连续 Q block，因此适合 FA3/FA4。但 DSA 已经用 per-Q `topk_indices` 表达稀疏可见范围，不需要为了标准 causal FA 接口维护 early/late 两组 block geometry。

历史 DSA Zigzag 路径仍存在，但主要属于 CP-v1，并带有 batch size 1 等 legacy 约束。Interleave 路径更自然地接入 multi-batch、per-request q lens、Fused MoE 和 FP8 KV cache 等当前组合。对 CP-v2，源码也明确只把 Interleave 注册给 DSA。

可以把三种布局的取舍总结为：

| 布局 | 优点 | 主要问题/用途 |
|---|---|---|
| 朴素连续 | 最简单、local 内存连续、恢复容易 | position 与 rank 绑定，Indexer/multi-batch 负载不够稳健；当前没有独立 strategy |
| Zigzag | 精确配平 dense causal 三角形，保留连续 Q block | metadata/early-late geometry 更复杂；当前主要配 FA/TRTLLM MHA |
| Interleave | local Q 数量均匀，早中晚 positions 分散，per-Q DSA metadata 自然 | 需要交错 shard 和 gather 后重排；当前专门配 DSA |

最终结论：

> DSA 不用朴素连续布局并非出于正确性，而是工程上的稳健负载选择。Interleave 主要打散 top-k 之前的 Indexer position、均分 multi-batch local Q；top-k 之后的 DSA sparse attention 本身通常已经较均衡。

---

## 5. CP-v1 与 CP-v2

### 5.1 CP-v1：模型内部显式切分

CP-v1 的典型流程散落在模型实现中：

1. 模型 `forward` 前判断 `can_cp_split` / `can_dsa_cp_split`；
2. 构建 `forward_batch.attn_cp_metadata`；
3. 在模型内部切 hidden states 和 positions；
4. attention backend 内 all-gather KV；
5. 最后一层后 all-gather hidden states。

DeepSeek V2/V3 系列的代表入口：

- metadata 初始化：[deepseek_v2.py](../python/sglang/srt/models/deepseek_v2.py#L3095)
- 输入切分：[deepseek_v2.py](../python/sglang/srt/models/deepseek_v2.py#L2805)
- 最终 hidden gather：[deepseek_v2.py](../python/sglang/srt/models/deepseek_v2.py#L2916)
- 旧公共函数集中在 [layers/utils/cp_utils.py](../python/sglang/srt/layers/utils/cp_utils.py#L88)

优点是模型可以高度定制；缺点是重复逻辑多，模型、后端和数据布局耦合紧。

### 5.2 CP-v2：runner 边界 + strategy 抽象

CP-v2 把公共流程提升到 runner：

```text
ForwardBatch（全局 token metadata）
    │
    ├─ prepare_cp_forward(): 构建 strategy metadata + padding
    │
    ├─ 先做 input embedding
    │
    ├─ cp_shard_model_inputs(): hidden/position 按 CP rank 切分
    │
    ├─ model.model(): transformer body 只跑 local token
    │      ├─ attention backend 按 strategy 收集完整 K/V 并写入本卡 cache
    │      └─ attention/MoE/FFN 保持 CP-local token 布局
    │
    ├─ cp_gather_after_forward(): 恢复全局 token 顺序
    │
    └─ logits_processor(): 使用全局 hidden 和原始 batch metadata
```

Eager runner 的完整边界见 [eager_runner.py](../python/sglang/srt/model_executor/runner/eager_runner.py#L251)，CP-v2 body 见 [_execute_extend_cp_v2](../python/sglang/srt/model_executor/runner/eager_runner.py#L349)。

strategy 统一负责：

- `can_apply`
- `build_metadata`
- `shard_hidden_states` / `shard_position_ids`
- `gather_hidden_states` / `gather_kv_cache`
- `run_attention`
- `materialize_full_kv` / `materialize_full_mla_kv`（源码函数名；作用是收集完整 K/V 并写入本卡 cache）

接口见 [ContextParallelStrategy](../python/sglang/srt/layers/cp/base.py#L93)。

### 5.3 CP-v2 如何启用

环境变量定义：

```bash
SGLANG_ENABLE_CP_V2=1
```

默认值本身是 false：[environ.py](../python/sglang/srt/environ.py#L623)。但 ServerArgs 会对上一节列出的模型类自动设为 true；如果用户显式设置环境变量，则尊重显式值。

学习和排障时建议显式设置，避免误判自己走的是 v1 还是 v2：

```bash
SGLANG_ENABLE_CP_V2=1 python -m sglang.launch_server ...
```

---

## 6. 一次 CP-v2 prefill 的完整数据流

### 6.1 启动阶段

#### 第一步：参数归一化

当前推荐入口：

```bash
--attn-cp-size C
--enable-prefill-cp
--cp-strategy zigzag|interleave
```

旧入口仍由 argparse 作为 deprecated alias 接受：

```text
--enable-prefill-context-parallel
--enable-dsa-prefill-context-parallel
--enable-nsa-prefill-context-parallel
--prefill-cp-mode in-seq-split
--dsa-prefill-cp-mode in-seq-split|round-robin-split
--nsa-prefill-cp-mode ...
```

映射关系：

```text
in-seq-split       -> zigzag
round-robin-split  -> interleave
```

兼容 parser 见 [server_args.py](../python/sglang/srt/server_args.py#L8718)，内部双向归一化见 [_handle_legacy_cp_arguments](../python/sglang/srt/server_args.py#L6383)。新部署应只写 canonical flags。

#### 第二步：模型特定 override

DeepSeek DSA/MLA 会自动调整若干参数：[overrides.py](../python/sglang/srt/arg_groups/overrides.py#L556)

- 开启 DP-attention framework；
- `moe_dense_tp_size=1`；
- 常把 `attn_cp_size` 重算为 `tp_size // dp_size`；
- zigzag MLA/DSA 常要求 DeepEP、`ep_size=tp_size`；
- prefill CUDA graph 通常禁用；
- DSA CP 限制 `tp_size <= 8`，即 CP 所在 TP group 单机。

因此，在这些模型上用户传入的 `--attn-cp-size` 不一定是最终生效值，必须看启动日志中的 resolved topology。

DeepSeek V4 进一步只允许 interleave，并限制 MoE A2A backend 为 `none/deepep/megamoe`：[deepseek_v4_hook.py](../python/sglang/srt/arg_groups/deepseek_v4_hook.py#L158)。

#### 第三步：创建并行组

`initialize_model_parallel` 创建 `_ATTN_CP`、`_ATTN_TP`、`_MOE_DP` 等 group。如果 `attn_cp_size > moe_dp_size`，源码把 `_MOE_DP` 直接指向 `_ATTN_CP`，让现有 MoE-DP gather/scatter 覆盖 CP partners：[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2528)。

#### 第四步：绑定 strategy

`init_cp_strategy` 根据 `cp_strategy` 构造进程级 singleton：

```text
zigzag    -> ZigzagCPStrategy
interleave -> InterleaveCPStrategy
```

worker subprocess 若没有重新运行 ServerArgs 初始化，会由 `get_cp_strategy()` 从全局 runtime args 懒初始化：[cp/base.py](../python/sglang/srt/layers/cp/base.py#L277)。

### 6.2 forward 前：构建 metadata 与 padding

`is_cp_v2_active(forward_batch)` 同时检查：

- CP-v2 环境开关；
- 当前 mode 是 context-parallel extend；
- strategy 已初始化；
- 有 `input_ids`；
- strategy 的长度/模式门槛满足。

然后 `prepare_cp_forward()`：

1. 从 `seq_lens_cpu` 和 `extend_seq_lens_cpu` 计算 layout；
2. 构建 `attn_cp_metadata`；
3. 调用 `pad_logical_token_to_physical()`，使各 rank collective shape 一致；
4. 修正 local DP buffer length；
5. 截断 `out_cache_loc` 到真实 token 数。

见 [cp/utils.py](../python/sglang/srt/layers/cp/utils.py#L139)。

padding 对 zigzag 按 `2*C` 对齐，对 interleave 按 `C` 对齐：[cp/padding.py](../python/sglang/srt/layers/cp/padding.py#L24)。metadata 同时保存：

- logical token 数：真实结果需要保留；
- physical token 数：通信/kernel 的统一 shape；
- gather 时会去掉 padding。

### 6.3 模型入口：全局 embedding 后切 local token

Eager CP-v2 先对全局 `input_ids` 做 embedding，再切 `input_embeds` 和 `positions`。也就是说 embedding lookup 本身在每个 CP rank 上有少量重复，但大部分 transformer body 从第一层开始只处理 local tokens。

切分期间只临时替换模型输入；退出 context manager 后，`ForwardBatch` 的全局 metadata 保持不变，保证 logits/logprob 处理看到原始 batch 语义。

### 6.4 每层 attention：local Q，full KV

对 dense MHA/GQA/MLA zigzag 路径：

1. local hidden 生成 local Q/K/V；
2. 把 local K/V 合并或拼接；
3. 在 `attn_cp_group` 内 all-gather；
4. 按 strategy 恢复全局 token 顺序；
5. 写入该层的 full KV cache；
6. local Q 的 prev/next block 分别以正确 causal length 查询 full cache；
7. attention output 仍保持 local zigzag 顺序。

FA3/FA4 的接入点：

- 收集完整 K/V 并写入本卡 cache：[flashattention_backend.py](../python/sglang/srt/layers/attention/flashattention_backend.py#L1178)
- CP attention dispatch：[flashattention_backend.py](../python/sglang/srt/layers/attention/flashattention_backend.py#L1389)

Zigzag gather 的实现会：

- local tensor pad 到所有 rank 最大长度；
- `all_gather_into_tensor`；
- 去掉每 rank padding；
- 根据 `cp_reverse_index` 恢复自然顺序；
- 见 [cp/zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L441)。

这里没有跨 CP rank 做 attention output 求和，因为不同 rank 负责的是不同 query token，不是同一 query 的 partial softmax。需要的是 full K/V 可见性，而不是把同一 Q 的分块输出 reduce。

### 6.5 DSA interleave：indexer 与稀疏 attention

DSA 多一层 indexer/top-k：

1. token/hidden/position 按 interleave 切分；
2. indexer key 在 CP group all-gather 成 full key；
3. local query 对 full key 做 top-k；
4. DSA attention 用 local query + full/gathered KV；
5. metadata 只保留当前 rank 实际拥有的 request/q lengths。

CP-v2 indexer gather 入口在 [dsa_indexer.py](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L475)。FP8 TRTLLM DSA KV 会先 pack 成 raw bytes 再 gather，避免 FP8 collective dtype 问题：[cp/interleave.py](../python/sglang/srt/layers/cp/interleave.py#L235)。

### 6.6 attention 与 MoE/FFN 之间

hidden state 在 transformer body 内尽量保持 CP-local token 布局。对于 DSA/MLA 特殊模型，`DSACPLayerCommunicator` 改写 layer communicator：

- attention/local token -> MoE/FFN full token 时 all-gather；
- MoE/FFN full token -> attention local token 时 reduce-scatter；
- 相关实现见 [communicator_dsa_cp.py](../python/sglang/srt/layers/communicator_dsa_cp.py#L71)。

对于通用 MoE 路径，如果 `attn_cp_size > moe_dp_size`，前面提到的 `_MOE_DP = _ATTN_CP` 会复用 MoE DP 的 token sharing 机制。

这也是为什么 CP 与 EP、MoE DP 的组合不能只看数学上能否整除，还要看 communicator 与 MoE backend 是否已接入。

### 6.7 最后一层：恢复全局 token 顺序

最后一个 PP rank 调用 `cp_gather_after_forward()`：

- all-gather 各 rank hidden states；
- 去 padding；
- zigzag/interleave 恢复自然 token 顺序；
- 再进入 lm_head/logits processor。

这样 sampling、logprob、请求边界仍使用普通全局顺序，不需要上层 API 感知 CP。

---

## 7. 后端支持：要分三层理解

### 7.1 CP strategy 与 attention backend 的直接接入

当前 CP-v2 strategy 源码声明：

| Strategy | CP-v2 直接支持的 calling convention | 实际入口 |
|---|---|---|
| zigzag | `FLASH_ATTENTION` | FA3/FA4 的 `FlashAttentionBackend` |
| zigzag | `TRTLLM_MHA` | `TRTLLMHAAttnBackend`，非 MLA |
| interleave | `DSA` | DSA backend 自己执行 attention，strategy 的 `run_attention` 是 no-op |

声明位置：

- zigzag：[cp/zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L337)
- interleave：[cp/interleave.py](../python/sglang/srt/layers/cp/interleave.py#L214)

`CPAttentionBackendKind.from_string()` 还把 `flashinfer` 映射成 `FLASH_ATTENTION`：[cp/base.py](../python/sglang/srt/layers/cp/base.py#L66)。但当前 `flashinfer_backend.py` 没有 CP-v2 strategy dispatch，而 `from_string()` 也没有在实际创建路径中使用。故不能仅凭这个枚举宣称 FlashInfer dense backend 已端到端支持 CP-v2；当前可靠选择应以 FA3/FA4、TRTLLM MHA 和注册 CI 为准。

### 7.2 DSA umbrella backend 内部实现

`--attention-backend dsa` 是上层 backend；其内部还有：

```text
--dsa-prefill-backend
  flashmla_sparse
  flashmla_sparse_q8
  flashmla_kv
  flashmla_auto
  flashinfer_sparse_mla
  fa3
  tilelang
  aiter
  trtllm
```

choices 见 [server_args.py](../python/sglang/srt/server_args.py#L347)。这些 implementation 还有各自的架构、KV dtype 和 kernel 限制，不能把“DSA backend 支持 CP”机械推导成“所有子实现、所有硬件、所有 dtype 组合都已验证”。当前强证据包括：

- GLM-5.2 H200 CP-v2 interleave：auto resolved DSA 路径；
- GLM-5.2 B200 LayerSplit：显式 `--dsa-prefill-backend trtllm`；
- DeepSeek V4 B200：V4 专用 DSA backend；
- DeepSeek V4 Pro MI35x：HIP `dsv4` + unified-kv path。

选择 DSA 子 backend 时，应先使用模型/硬件自动默认，再以对应 test recipe 为基线。

### 7.3 硬件与 collective backend

#### NVIDIA CUDA

当前覆盖最完整：

- H100/H200：FA3、MLA/DSA；
- B200：TRTLLM MHA、DSA/V4、FP4/MXFP4；
- CP collectives 通过 `attn_cp_group` 的 group coordinator，GPU 上通常落到 PyNCCL/NCCL；
- 对符合条件的分配可使用 symmetric memory context。

#### AMD ROCm

当前有 DeepSeek V4 Pro MI35x 的 registered accuracy test。HIP backend 会把全局 metadata 按 CP rank 重索引，local query 对 full gathered KV：[deepseek_v4_backend_hip_radix.py](../python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py#L1334)。

这证明的是 V4 专用 HIP 路径，不应泛化为所有通用 zigzag/FA 路径都在 ROCm 可用。

#### Ascend NPU

Ascend backend 中有 CP-v1 风格的专用实现：

- 合并 K/V 后做一次 all-gather；
- non-MLA 使用 FIA 做 CP attention；非 FIA 分支明确未实现；
- DSA 有 balance attention 路径；
- 见 [ascend_backend.py](../python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L250) 和 [do_cp_attn_fia](../python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L973)。

仓库教程给出 Qwen3-235B-A22B、PD prefill、batch size 1 的 NPU PCP recipe，但文档基于 v0.5.16，仍使用 deprecated 参数。学习当前主线时要把它视为平台专用 legacy 路径，而不是 CP-v2 通用 strategy 的证明。

#### MUSA

MUSA FlashAttention backend 有 CP KV all-gather 和 zigzag attention hook：[musa/flashattention_backend.py](../python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py#L251)。当前没有在 `test/registered/cp` 中看到 MUSA CP E2E 注册，因此更适合描述为“源码存在平台路径，需在目标环境验证”。

---

## 8. KV cache、Radix Cache、PD 与 LayerSplit

### 8.1 普通 CP 为什么不直接省 `1/C` KV cache

以 zigzag 为例，切分的是本层输入 hidden；本层的新 K/V 在开始时还不存在。每个 rank 根据 local hidden 产生 local K/V，随后 `materialize_full_kv()` 把其他 CP rank 新算出的 K/V 也收集过来。这个源码函数名里的 `materialize`，通俗地说就是“真的把完整数据收集到本卡并存下来”。具体步骤是：

1. 拼接 local K/V；
2. 在 CP group gather；
3. 恢复自然顺序；
4. 把 full key/value 写入 token-to-KV pool。

见 [cp/zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L396)。MLA latent KV 同样会 gather 后写 full buffer。如果存在已经命中的历史 prefix cache，那部分 K/V 原本就已在 cache 中；这里主要交换和补全的是本轮新增 token 的 K/V。

这样做的好处是：

- local query 可直接复用成熟的 paged KV attention kernel；
- prefix/radix cache 仍维持普通全局 token slot 语义；
- decode 阶段无需重新拼接分布式 KV。

代价是 CP rank 上的 KV cache 普通情况下仍是完整视图。

### 8.2 prefix cache/radix cache

Zigzag metadata 用：

```text
prefix_offset = seq_len - extend_len
```

把已有 prefix 长度加进 prev/next 的 KV 可见长度，因此支持“已有 radix prefix + 本轮只 extend 一部分 token”的语义：[cp/zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L138)。

### 8.3 PD 分离很适合 Prefill CP，但当前不能直接组合 DCP Decode

Prefill worker 可以专门用 CP 优化长 prompt；decode worker 使用更适合 token-by-token decode 的 TP/DP 拓扑。这样避免 unified server 中：

- attention TP degree 被 CP degree 吃掉；
- CP rank 上 attention 权重复制导致 decode 重复 GEMM；
- 短 prefill/decode 承担额外 gather。

但这里有一个非常重要的当前实现限制：

> 如果 Decode worker 开启 `--dcp-size > 1`，Prefill worker 必须满足 `attn_cp_size == 1`。也就是说，目前不支持 `P: Prefill CP > 1 -> D: DCP > 1`。

Decode worker 在读取 Prefill bootstrap 信息时会直接校验并拒绝这个组合：[disaggregation/common/conn.py](../python/sglang/srt/disaggregation/common/conn.py#L551)。因此“PD 允许 P/D 使用不同拓扑”不能推导成任意 Prefill CP 与 DCP 都已经支持。

当前可选择的实际拓扑是：

| Prefill worker | Decode worker | 当前状态 |
|---|---|---|
| Prefill CP `>1`，DCP1 | 普通 TP/DP、DCP1 | 支持；用 PCP 优化 P，D 不做 decode context sharding |
| Prefill CP1、DCP1 | DCPN | MLA/hybrid-MLA 支持；P sender 按目标 DCP rank 做 token relayout，直接写 D local pool |
| Prefill CP1、DCPN | 相同 DCPN | 支持；matching DCP ranks 直接传 local pages，但 P 端 DCP 通常增加 prefill 开销 |
| Prefill CP `>1` | DCPN | 当前不支持 |

其中 `P DCP1 -> D DCPN` 并不是把 full KV 发到每个 D rank 后再切。P sender 会根据目标 `dcp_size/dcp_rank` 选择 token rows，并直接写入 D rank 的 local physical pool。相同 DCPN 则要求 P/D DCP rank 匹配，local shard 直接传输。完整数据流见 [sglang_dcp_source_study.md](./sglang_dcp_source_study.md) 的 PD disaggregation 章节。

PD bootstrap/route metadata 已包含 `attn_cp_rank/size`，prefill polling 也会在 CP/TP group 内同步。

注意：`SGLANG_DISAGG_STAGING_BUFFER` 当前不支持 prefill CP，因为 CP 会重写 per-rank `index_slice`：[disaggregation/prefill.py](../python/sglang/srt/disaggregation/prefill.py#L175)。

### 8.4 DSA cache LayerSplit

`--enable-dsa-cache-layer-split` 是普通 Prefill CP 之上的进一步显存优化。普通 PCP 按 token 分 query/hidden，但每个 CP rank 最终通常仍保存所有层、完整 token 范围的 KV；LayerSplit 再把“哪些层的 cache 由谁长期保存”分给不同 CP ranks。

当前实现只允许 **DSA/MLA + Interleave PCP + PD Prefill worker**。它不是普通 PCP 的默认组成部分，也不能用于 unified service 或 PD Decode worker。

#### 8.4.1 它利用了 Transformer 的按层执行顺序

Transformer 必须顺序执行：

```text
layer 0 -> layer 1 -> layer 2 -> ... -> layer N-1
```

执行第 `i` 层 attention 时，只会读取第 `i` 层的历史 KV/indexer cache，不会同时读取第 `i+1`、`i+2` 层的 cache。因此从“当前这一瞬间需要读取什么”来看，每张卡只需要当前正在计算的这一层 cache，而不需要把所有层都同时放在本卡上。LayerSplit 正是利用这个时间特性：

1. 每个 CP rank 只长期保存自己负责的一段层；
2. 计算非本 rank 所有的层时，由 owner 临时广播该层 cache；
3. 非 owner rank 将它放进一个可复用的 remote scratch buffer；
4. 下一层到来时，这个 scratch buffer 可以被覆盖。

源码给每个 rank 分配的 main KV layer 数量近似为：

```text
ceil(num_layers / cp_size) + 1 remote scratch layer
```

推导见 [get_glm_dsa_layer_split_effective_num_layers](../python/sglang/srt/layers/cp/utils.py#L87)。DSA 还有独立的 Indexer K cache，LayerSplit 也为它建立 owner-only buffer 和 remote scratch buffer。

#### 8.4.2 一个 `CP=4、64 layers` 的例子

普通 Prefill CP 中，每个 rank 都长期保存全部 64 层的 cache：

```text
rank0: layers 0..63
rank1: layers 0..63
rank2: layers 0..63
rank3: layers 0..63
```

LayerSplit 将层连续、尽量均匀地分配：

```text
rank0 owns layers  0..15
rank1 owns layers 16..31
rank2 owns layers 32..47
rank3 owns layers 48..63

每个 rank 另外准备一个可反复覆盖的 remote layer scratch buffer
```

当模型执行 layer 20 时，rank1 是 owner。下面先画的是读取该层**已经存在的历史 cache**；本轮刚生成的新 K/V 仍由 PCP all-gather 汇总，8.4.5 会把两种通信分开说明：

```text
rank1 的 layer 20 cache
          │
          ├─ broadcast -> rank0 remote scratch
          ├─ broadcast -> rank2 remote scratch
          └─ broadcast -> rank3 remote scratch

随后所有 rank 都执行 layer 20 attention：
rank0 处理自己的 local Q
rank1 处理自己的 local Q
rank2 处理自己的 local Q
rank3 处理自己的 local Q
```

进入 layer 21 后，scratch buffer 可以改为保存 layer 21。owner 判断和连续层范围划分见 [cp/utils.py](../python/sglang/srt/layers/cp/utils.py#L105)，owner-only buffer 与空占位符的分配见 [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L302)。

若忽略对齐和独立 Indexer buffer，持久 cache 大致由：

```text
普通 PCP：每 rank 约 64 层
LayerSplit：每 rank 约 16 层 + 1 个临时层
```

当层数远大于 CP degree 时，持久 layer cache 的节省接近 `cp_size` 倍，但不是严格的 `1 / cp_size`：还要加 main KV 和 Indexer 的 scratch buffer、padding、page metadata 等开销。

#### 8.4.3 它不是 Pipeline Parallel，也不减少每张卡计算的层数

LayerSplit 的名字容易让人误以为“rank0 计算前 16 层、rank1 计算后 16 层”。实际不是这样。

| 机制 | 主要切分对象 | 每个 rank 是否执行所有层 | hidden 是否在层边界发往下一个 stage |
| --- | --- | --- | --- |
| Prefill CP | 当前请求的 token/query | 是 | 否 |
| DSA cache LayerSplit | 各层 KV/indexer cache 的长期保存位置 | 是 | 否 |
| Pipeline Parallel | 层权重与层计算 | 否 | 是 |

LayerSplit 后：

- 模型权重没有按 LayerSplit owner 切开；
- 每个 PCP rank 仍按顺序运行全部 Transformer layers；
- 每个 rank 仍负责自己的 local query token；
- owner 只表示“谁长期保存这一层的 cache”，不表示“谁独占这一层的计算”。

因此它降低的是 KV/indexer cache 常驻显存，不会把一张卡的 64 层计算直接降为 16 层。

#### 8.4.4 一层 DSA 的两套 cache 都需要处理

DSA 有两条相关但不同的路径：

```text
Indexer 路径：Indexer K cache
  用于计算 top-k token positions

主 attention 路径：MLA latent KV cache
  用于在选中的 positions 上计算真正 attention output
```

LayerSplit 必须同时管理两套 cache：

- `LayerSplitIndexKeyCache`：只有 owner 为这一层长期分配 Indexer K buffer；
- `LayerSplitDSATokenToKVPool`：只有 owner 为这一层长期分配 MLA latent KV buffer；
- 非 owner 分别通过 remote scratch 读取当前层数据；
- owner-only 写入确保一层不会在所有 CP ranks 上长期重复保存。

Indexer cache 的 owner broadcast 见 [LayerSplitIndexKeyCache.get_broadcastable_buffer](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L131)，MLA cache 的 owner-only write/read 见 [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L350)。

这也是 DSA 做 LayerSplit 更有价值的原因之一：除了主 MLA KV，它还有额外的 Indexer K cache；把两者都从“每 rank 全层复制”改成“每层只有一个 owner”，节省空间更明显。

#### 8.4.5 最关键的区别：all-gather 新 K/V，broadcast 补历史 cache

普通 Prefill CP 的基本过程没有因为 LayerSplit 消失。以 `CP=2`、当前层新处理一段 token 为例：

```text
rank0 local hidden -> rank0 local Q/K/V --┐
                                          ├─ PCP all-gather + 恢复 token 顺序
rank1 local hidden -> rank1 local Q/K/V --┘
                                          │
                                          ▼
                              本轮新增 token 的完整 K/V
```

因此，在一个全新请求、没有任何历史 prefix、整个 prompt 只用一个 chunk 处理的理想语义中，当前新 K/V 的 all-gather 已经足够支持 attention：

```text
prefix_len = 0
visible KV = 本轮新生成并 all-gather 的 KV
```

LayerSplit broadcast 解决的是另一个问题：非 owner 没有长期保存当前层此前已经存在的 cache。这里的“历史 prefix”不只指跨请求的 Radix Cache 命中，也包括：

1. prefix/radix cache 命中的系统提示词或公共前缀；
2. chunked prefill 前几个 chunk 已经生成的 KV；
3. 该请求之前已经写入 cache、而本轮只继续 extend 的任何 token。

例如，某请求已经有 `P` 个 cached token，本轮再 extend `E` 个 token，layer `L` 的 owner 是 rank0：

```text
本轮开始前：

rank0 persistent layer-L cache：token [0, P)       <- owner 长期保存
rank1 persistent layer-L cache：不存在             <- 非 owner 没有这层

本轮需要：

每个 local Q 可见 token [0, P+E)
```

这时逻辑上需要两条不同来源的数据流：

```text
历史部分 token [0, P)：

rank0 owner persistent cache
            │
            └─ LayerSplit broadcast -> rank1 remote scratch


本轮新增 token [P, P+E)：

rank0 local new K/V --┐
                      ├─ PCP all-gather -> full new K/V
rank1 local new K/V --┘
```

两部分合并后才是当前 attention 所需的完整上下文：

```text
full visible KV
  = historical prefix KV [0, P)
  + current extend KV    [P, P+E)
```

以主 MLA latent KV 为例，各 rank 的使用和保留方式可以理解为：

```text
rank0（owner）：
  persistent cache 中原本有 historical prefix
  将 full new KV 写入 persistent cache 的新增 slots
  forward 结束后继续长期保留 [0, P+E)

rank1（non-owner）：
  broadcast 得到 historical prefix，放入 remote scratch
  将本轮 full new KV 更新到 remote scratch 的新增 slots
  用 scratch 中的 [0, P+E) 完成当前层 attention
  进入其他层后，scratch 可以被覆盖，不长期保留 layer L
```

LayerSplit pool 的 `set_mla_kv_buffer()` 正体现了这个区别：如果 remote scratch 当前装的是该层，它会先更新 scratch 中本轮新增 slots；只有 owner 还会把新 KV 写入该层的 persistent buffer，见 [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L420)。

可以用表格记住两种通信：

| 通信 | 发送者 | 主要数据 | 为什么需要 |
| --- | --- | --- | --- |
| PCP K/V all-gather | 所有 CP ranks | 各 rank 本轮刚计算的 local K/V | 汇总本轮新增 token 的完整 K/V |
| LayerSplit owner broadcast | 当前层唯一 owner | owner 长期保存的当前层 cache buffer | 让非 owner 临时恢复自己没有持久保存的当前层 cache，关键缺口是历史 prefix |

Interleave strategy 对本轮新 Indexer K 的汇总见 [materialize_full_indexer_k_cache](../python/sglang/srt/layers/cp/interleave.py#L217)，对本轮新 MLA latent KV 的汇总见 [materialize_full_mla_kv](../python/sglang/srt/layers/cp/interleave.py#L258)。这两处是 PCP current-token collective；它们与 LayerSplit pool 的 owner broadcast 不能混为一谈。

##### 完全没有 prefix 时还需不需要 broadcast

从 attention 数学和数据依赖看：

```text
P = 0
=> 没有历史 KV 缺口
=> 本轮新 K/V all-gather 已足够
```

但当前实现采用统一的 cache-buffer/scratch/prefetch 路径，并不一定根据 `prefix_len==0` 把 owner broadcast 完全裁掉。例如 Indexer 读取会通过 owner-broadcastable buffer，随后还会调用 `prefetch_kv_buffer()` 准备主 MLA buffer；所以即使有效历史区域为空，运行时仍可能发起 buffer broadcast。

应当区分：

```text
语义上必须传什么：
  没有 prefix 时，只需汇总本轮新 K/V

当前统一实现实际上可能传什么：
  仍可能广播当前层 cache buffer，以建立统一的 remote scratch 读取路径
```

这类广播在零 prefix 时更多是实现统一和预取机制带来的开销，不是因为本轮新 K/V 在 all-gather 后又从数学上缺失了一遍。具体 wire payload 是当前层 buffer，而不是一个显式切出的“仅 prefix 长度”tensor，因此 buffer 中也可能已有本轮 Indexer K 等有效内容；“主要补历史 prefix”描述的是它不可替代的数据依赖，不表示实现只发送 `[0, P)` 这一段。

##### Chunked Prefill 为什么很快就会需要它

假设一个 2048-token prompt 分成两个 1024-token chunk：

```text
chunk 1：extend token [0, 1024)
  没有历史 prefix

chunk 2：extend token [1024, 2048)
  token [0, 1024) 已经成为 cached prefix
```

到 chunk 2 时，即使这个请求从未命中过其他请求的 Radix Cache，前一个 chunk 也已经构成历史 cache。对于非 owner：

```text
owner broadcast：取回 chunk 1 的当前层 cache
PCP all-gather：汇总 chunk 2 刚生成的新 K/V
```

所以 LayerSplit broadcast 不只出现在“prefix cache 跨请求命中”场景；长 prompt 常见的 chunked prefill 本身就会产生它需要处理的历史部分。

#### 8.4.6 DSA 为什么特别适合隐藏 LayerSplit 通信

LayerSplit 并不是免费省内存。读取非本 rank 所有的层、且该层已经存在 owner-only cache 时，需要执行：

```text
owner-broadcast(current layer persistent cache -> remote scratch)
```

DSA 的 `Indexer -> main sparse attention` 两阶段结构提供了一个自然的通信重叠窗口：

```text
1. 取得当前层 Indexer K cache（包括需要读取的历史部分）
2. 异步开始 owner-broadcast 当前层 MLA latent cache
3. GPU 同时执行 Indexer score 和 top-k
4. Indexer 完成时，期望 MLA KV 已经准备好
5. 使用 top-k positions 执行主 DSA attention
```

代码顺序非常直接：

```python
buf = self.get_buffer(layer_id)          # 取得当前层 Indexer K
self.pool.prefetch_kv_buffer(layer_id)   # 异步预取当前层 MLA latent KV
return GetKAndS.execute(...)             # 继续做 Indexer
```

见 [LayerSplitIndexKeyCache.get_k_and_scale](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L97)。MLA KV 的异步 broadcast 使用独立 CUDA stream，实现在 [prefetch_kv_buffer](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L458)。

对于不经过 Indexer 的 full-attention layer，没有上述同层 Indexer 计算可用于隐藏通信。SGLang 会尝试提前一层预取下一个 full-attention layer 的 KV，让传输与当前层 attention 重叠：[maybe_prefetch_next_full_attention_kv](../python/sglang/srt/layers/communicator_dsa_cp.py#L52)。

另外，DSA 使用 MLA latent KV，单 token 保存的是压缩后的 latent 表示，而不是普通 MHA 中展开的多 KV-head K/V。单层 cache 相对紧凑，使“逐层广播换常驻显存”更可能划算。普通 MHA/GQA 的单层 K/V 更大，照搬这一方案可能让逐层广播代价过高。

这里描述的是 cache 读取与通信隐藏；本轮新生成的 Indexer K/MLA KV 仍按 8.4.5 所述经过 PCP current-token collective。实现可以通过 stream、fused store 和 scratch update 对实际先后顺序做重叠，但两类数据依赖不变。

#### 8.4.7 不要误解成“Indexer 选完 top-k，只传 top-k KV”

当前 LayerSplit 的核心读取方式不是：

```text
先得到 top-k indices
  -> 只从 owner 拉取这些位置的少量 KV
```

而是：

```text
owner
  -> broadcast 当前层所需的 cache buffer（关键是非 owner 缺少的历史内容）
  -> 非 owner 写入 remote scratch
  -> 将本轮 all-gather 后的新 KV 合入对应 slots
  -> 每个 rank 再为自己的 local Q 执行 Indexer/attention
```

`remote_kv_buffer` 按完整 token pool 容量分配，见 [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L330)；cache miss 时 `_get_broadcastable_kv_buffer()` 从 owner 广播当前层 buffer，见 [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L504)。

所以 DSA top-k 并不会直接把 LayerSplit broadcast 变成“只传 top-k 个 KV”。DSA 在这里的特殊优势主要是：

1. MLA latent KV 本身紧凑；
2. 同时拆分主 KV 与额外 Indexer K，显存收益大；
3. Indexer 计算可以与主 MLA KV 预取重叠；
4. sparse attention 之后只对 top-k positions 执行主要 attention 计算。

#### 8.4.8 为什么强制限制在 PD Prefill worker

LayerSplit 最适合 Prefill-only worker，因为它只需要完成 Prompt 前向并把结果交给 Decode 端：

```text
PD Prefill worker
  rank0 owns 一段 layers
  rank1 owns 一段 layers
  ...
       │
       │ 各 owner 通过 PD transfer 发送自己负责的层
       ▼
PD Decode worker
  按 Decode 端自己的正常 KV cache 布局接收
  负责后续逐 token Decode
```

在 unified service 中，同一批 worker 做完 Prefill 后还要继续 Decode。LayerSplit 下每个 rank 都缺少大部分层的持久 cache，于是每生成一个 token，都需要依次访问全部层：

```text
每个 Decode step：
  broadcast layer 0 cache
  计算 layer 0
  broadcast layer 1 cache
  计算 layer 1
  ...
  broadcast last layer cache
```

这会把逐层大块 cache 通信放进每一个 latency-sensitive Decode step；当前 Decode backend 也没有按这种 cache 语义实现。因此源码不是仅仅“建议不要用”，而是明确拒绝：

- `disaggregation_mode == "decode"`：报错；
- 普通 `disaggregation_mode == "null"`：也报错；
- 只有 `disaggregation_mode == "prefill"` 才允许。

校验与错误信息见 [server_args.py](../python/sglang/srt/server_args.py#L5318)。即使普通服务只发 `max_new_tokens=0` 请求，当前启动检查也不会动态放宽，因为 cache pool 和执行语义在服务启动时已经确定。

#### 8.4.9 当前完整使用条件与限制

当前 commit 下，需要同时满足：

```text
DSA model
+ MLA backend
+ 非 draft worker
+ PD Prefill worker
+ --enable-prefill-cp
+ --cp-strategy interleave
+ --enable-dsa-cache-layer-split
+ Mooncake/Mooncake TCP transfer backend
+ pp_size == 1
```

其中模型/MLA/draft 条件见 [is_glm_dsa_cache_layer_split_enabled](../python/sglang/srt/layers/cp/utils.py#L56)；PD、Interleave、transfer backend 和 PP 限制集中在 [server_args.py](../python/sglang/srt/server_args.py#L5314)。当前 ServerArgs help 也说明 transfer backend 只支持 `mooncake/mooncake_tcp`，MORI/NIXL 尚未接入：[server_args.py](../python/sglang/srt/server_args.py#L1122)。

可以用下面这张表快速判断：

| 部署方式 | PCP | LayerSplit | 当前是否允许 |
| --- | ---: | ---: | --- |
| 普通 unified service | 开 | 关 | 是 |
| 普通 unified service | 开 | 开 | 否 |
| PD Prefill worker | 开 | 关 | 是 |
| PD Prefill worker，DSA Interleave | 开 | 开 | 是，仍需满足 backend/PP 等限制 |
| PD Decode worker | 不适用 | 开 | 否 |

最后可以这样理解它为什么“特殊”：

> LayerSplit 利用“Transformer 每一时刻只读取当前层 cache”的执行顺序，用逐层 owner broadcast 换取接近 `1 / cp_size` 的持久 layer-cache 显存。DSA 又恰好具有紧凑 MLA latent KV、额外 Indexer cache、可与传输重叠的 Indexer 阶段，以及 PD Prefill-only 的使用场景，所以当前实现只为这条路径付出工程复杂度；这不是普通模型在数学上绝对不能按层拆 cache，而是其他模型和 unified Decode 上通常缺少足够的收益与配套 backend。

---

## 9. 为什么采用 K/V all-gather

### 9.1 all-gather 补全的是“本层刚算出来的新 K/V”

每个 Transformer 层都有自己独立的 K/V。进入某一层时，CP 已经把 hidden 按 token 分到不同 rank；该层的新 K/V 还不存在，并不是先生成一份完整 K/V 再把它切开。

```text
local hidden
    │ 本层 Q/K/V projection
    ▼
local Q/K/V
    │ 只对 K/V 做 CP all-gather
    ▼
每个 CP rank 得到完整 token 范围的 K/V
    │
    ├─ 本 rank 的 local Q 查询完整 K/V
    └─ 完整 token 范围的 K/V 写入本卡 cache
```

因此，“每卡最终保存完整 K/V”是 all-gather 的结果，all-gather 是各 rank 相互补齐 K/V 的过程。这里的“完整”只表示覆盖完整 token 序列；若还有 attention TP，每卡仍可能只保存自己的 attention head shard。

到了下一层，输入仍是按 token 分片的 local hidden，因此会重新执行“local K/V projection → K/V all-gather”。不同层的 K/V 不能互相复用。

### 9.2 为什么这个交换可能是划算的

普通 Prefill CP 优先解决的是长 prompt 的计算和中间 activation，而不是 KV cache 容量。对 dense attention，可以粗略看成：

```text
attention 计算量          ~ O(L²)
本层 K/V 通信量与存储量   ~ O(L)
```

CP 让每卡只保留约 `L/C` 个 query/hidden token，把主要的 attention、MLP/MoE token 计算分到多张卡；代价是本层 K/V all-gather。序列足够长、互联带宽足够高时，节省的 `O(L²)` 计算可能明显大于新增的 `O(L)` K/V 通信。

GQA/MQA/MLA 的 K/V 宽度通常又小于普通 MHA，交换 K/V 的代价相对更低。这也是它们适合该方案的重要原因之一。

采用完整 K/V 还可以继续复用成熟的：

- paged attention kernel；
- token-to-KV pool；
- Radix/Prefix Cache；
- 已有 causal attention metadata；
- 后续 decode 的 KV cache 访问方式。

all-gather 也是 GPU 集群上高度优化的 collective。工程上，这是用一套规则清晰、容易接入现有 cache/backend 的通信，换取 query 和 transformer body 的并行。

### 9.3 为什么不直接使用 Ring Attention

Ring Attention 通常让每张卡只持有一部分 K/V，并让这些 K/V 在 rank 之间轮流传递。local Q 每收到一块 K/V 就计算一部分 attention，再用 online softmax 合并各块结果。它有机会避免在每张卡同时保存完整 token 范围的 K/V，但会带来：

- 多轮点对点通信和同步；
- 分块 softmax 的数值与 kernel 实现复杂度；
- variable-length batch、causal mask 和 prefix hit 下更复杂的调度；
- 与 paged KV、Radix Cache、PD transfer 以及现有 decode kernel 的额外适配。

更重要的是，Ring Attention 只改变 prefill attention 的计算方式，并不会自动解决 prefill 结束后的长期 KV 存储。若 decode 也要让 K/V 分布在多卡上，还需要一套支持分布式 KV cache 的 decode 方案。

所以 SGLang 当前主路径选择的是较直接的折中：prefill 计算按 query/token 分摊，但通过 all-gather 保持普通完整 token 范围的 KV cache 语义。

### 9.4 “prefill 放不下 KV，就无法 decode”应当怎样理解

这个理解对 **同一组 GPU 完成 prefill 和 decode 的 unified serving** 基本成立。设 prompt 长度为 `L`：

```text
prefill 完成后：KV token 数 = L
decode 第 t 步：KV token 数 = L + t
```

如果同一 worker 在 prefill 结束时连长度 `L` 的完整 prompt KV 都无法保存，请求确实不能正常进入后续 decode。因此，普通 Prefill CP 不是用来解决“单个请求的 KV 根本放不下”的工具；它假设 KV 基本放得下，主要优化生成这些 KV 时的计算和 activation。

但不能进一步说“KV cache 压力只在 decode 阶段”。例如输入 100K token、输出 1K token 时，prefill 结束已经生成 100K token 的 KV，decode 最终只是增长到 101K。大部分 KV 仍来自 prompt，只是它会在整个 decode 生命周期中长期驻留。

两阶段的典型显存压力不同：

| 阶段 | 主要压力 |
|---|---|
| Prefill | 完整 prompt KV，加上大批 token 的 activation、attention 临时 buffer |
| Decode | 多请求 KV 长期驻留、输出持续增长、并发与 cache 命中保留 |

Prefill CP 能降低 local query/hidden、部分 activation 和计算压力，但普通路径不降低完整 prompt KV。Decode 阶段的 KV 问题之所以更常成为服务容量问题，是因为它持续时间长、并发请求多，而且 KV 会继续增长。

### 9.5 PD 分离是重要例外

在 PD 分离中，两类 worker 可以采用不同的拓扑：

```text
Prefill worker：生成 KV -> 临时保存 -> 传给 Decode worker -> 释放
Decode worker：接收 KV -> 长期保存 -> 每步追加
```

因此，Prefill CP rank 暂时收集完整 K/V，不等于 Decode worker 必须使用相同的 CP 复制方式；P/D 的 TP、DP、PP 等拓扑可以独立设计。

但是当前 DCP transfer 是一个关键例外：D worker 设置 `dcp_size>1` 时会要求 P worker 的 `attn_cp_size==1`。所以当前不能同时采用“P 用 Prefill CP 优化长 prompt、D 用 DCP 分片长期 KV”这一看起来最自然的组合。现阶段必须在以下方案中选择：

```text
方案 A：P 用 Prefill CP，D 使用 DCP1
        -> 优先优化长 prompt/TTFT

方案 B：P 使用 Prefill CP1/DCP1，D 使用 DCPN
        -> P sender 做 DCP token relayout，优先优化 D 的 KV capacity

方案 C：P/D 使用相同 DCPN，且 P 的 Prefill CP=1
        -> local shard rank-to-rank 直传，但 P 端 DCP 通常增加 prefill 开销
```

限制和 direct-local transfer 详见 [sglang_dcp_source_study.md](./sglang_dcp_source_study.md) 的 PD disaggregation 章节。

最终可用一句话判断：

> 如果瓶颈是长 prompt 的 prefill 计算或 activation，K/V all-gather 的 Prefill CP 可能很合理；如果瓶颈是 KV cache 容量本身，普通 Prefill CP 不是目标方案，应进一步考虑 DSA LayerSplit、PD、HiCache/offload 或面向 decode 的 DCP。

---

## 10. Decode 阶段会发生什么

Prefill CP 只在 extend 活跃，但启动后的并行拓扑和权重布局不会消失。

当 `attn_cp_size=tp_size` 时，`attn_tp_size=1`，attention weights 往往在各 CP rank 复制。decode 若仍让每个 rank 用完整权重计算同一 token，会产生重复 GEMM。

实验性选项：

```bash
--enable-cp-decode-attn-tp
```

它在 decode 临时把复制的 attention linear weight 按 CP rank slice，恢复类似普通 TP 的 GEMM，再在退出 context 时还原。实现见 [cp_decode_attn_tp.py](../python/sglang/srt/layers/cp/cp_decode_attn_tp.py#L1)。

当前仅白名单：

- DeepSeek V4 及 NextN/DSpark variants；
- GLM-5.x DSA 及 NextN variants。

启动校验见 [server_args.py](../python/sglang/srt/server_args.py#L5194)。这不是 DCP；它只是复用 CP ranks 做 decode attention TP weight slicing。

对 unified serving，更稳妥的第一选择通常仍是：先对比关闭 CP，或使用 PD 分离；不要默认这个实验开关能消除所有 decode 代价。

---

## 11. 启动参数、约束与可复用 recipe

### 11.1 通用硬约束

ServerArgs 当前校验：[server_args.py](../python/sglang/srt/server_args.py#L6425)

```text
tp_size % attn_cp_size == 0
tp_size % (dp_size * attn_cp_size) == 0

如果 attn_cp_size != moe_dp_size：
    只允许 moe_dp_size == 1

attn_cp_size > 1：
    不支持 AITER allreduce fusion
```

注意最后一条说的是 `--enable-aiter-allreduce-fusion`，不是说 DSA 的 `--dsa-prefill-backend aiter` 一定不可用；二者是不同功能。

当 `moe_dp_size > 1`：

- `tp_size % moe_dp_size == 0`；
- `ep_size * moe_dp_size <= tp_size`；
- 当前禁止 PP；
- 若 `ep_size > 1`，要求 `ep_size * moe_dp_size == tp_size`。

所以 PP×CP 不是一概不支持：Qwen3 CI 已验证 `PP2 × CP2`，但其 `moe_dp_size=1`。

### 11.2 Qwen3 MoE / GQA 学习配置

最直接的 FA3 zigzag 样例：

```bash
SGLANG_ENABLE_CP_V2=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --tp-size 4 \
  --ep-size 4 \
  --attn-cp-size 4 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --moe-a2a-backend deepep \
  --attention-backend fa3 \
  --disable-piecewise-cuda-graph
```

它适合按以下顺序观察：

1. TP/CP groups；
2. zigzag metadata；
3. FA3 full KV materialization；
4. local attention；
5. MoE gather/scatter；
6. final hidden gather。

### 11.3 GLM-5.2 / DSA interleave 学习配置

```bash
SGLANG_ENABLE_CP_V2=1 \
python -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --trust-remote-code \
  --tp-size 8 \
  --attn-cp-size 8 \
  --enable-prefill-cp \
  --cp-strategy interleave
```

对 DSA family，最终 `attn_cp_size` 可能由模型 override 重算为 `tp_size // dp_size`。以启动日志的 resolved 值为准。

PD prefill 上再研究 LayerSplit时，基于仓库测试增加：

```bash
--disaggregation-mode prefill \
--disaggregation-transfer-backend mooncake \
--enable-dsa-cache-layer-split \
--dsa-prefill-backend trtllm
```

不要在 unified server 随意打开 LayerSplit。

### 11.4 GPT-OSS / TRTLLM MHA / prefill graph

```bash
SGLANG_ENABLE_CP_V2=1 \
python -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tp-size 4 \
  --attn-cp-size 4 \
  --enable-prefill-cp \
  --cp-strategy zigzag \
  --attention-backend trtllm_mha \
  --cuda-graph-backend-prefill breakable
```

CP-v2 的 breakable prefill CUDA graph 当前只在以下组合启用：[cp/bcg.py](../python/sglang/srt/layers/cp/bcg.py#L43)

```text
CP enabled
attn_cp_size == tp_size
strategy == zigzag
prefill attention backend == trtllm_mha
```

其他 CP 配置通常走 eager prefill；decode CUDA graph 可以仍然开启。

---

## 12. 关键兼容性与限制清单

### 12.1 策略/模型限制

- DeepSeek V4：只支持 interleave；
- DSA CP-v2：当前只支持 interleave；zigzag 主要是 CP-v1；
- MiMo V2 CP-v2：只支持 zigzag，且只支持 text inference；
- zigzag：每条 extend sequence 至少 `2*C` token；
- interleave：全局 extend token 至少 `C`；
- legacy DSA/MLA zigzag 路径的模型 override 仍提示 batch size 1；
- CP-v2 strategy metadata 已支持多 request，但是否可用还取决于具体模型/backend/communicator。

### 12.2 硬件/拓扑限制

- DSA/MLA CUDA/ROCm 自动配置明确 `tp_size <= 8`，即 CP group 单机；
- PP 可以让多个单机 CP stage 跨节点串联，不等于 CP collective 本身跨机；
- AITER allreduce fusion 与 CP 不兼容；
- CPU/XPU 没有看到注册的 CP E2E 证据；
- MUSA/NPU 有平台专用源码路径，但应按平台 recipe 单独验证。

### 12.3 调度与 batch

- legacy DSA zigzag scheduler 明确把 prefill batch 限为 1：[schedule_policy.py](../python/sglang/srt/managers/schedule_policy.py#L1201)；
- DSA interleave 支持 multi-batch token 分配；
- 不能仅用 `max-running-requests` 判断一次 prefill forward 的真实 batch；还要看 chunking、radix hit、continuous batching；
- CP 的最低 token 门槛是正确性条件，不是性能门槛。

### 12.4 speculative decoding

当前 CI 证明部分组合可用：

- GLM-5.2 CP-v2 + EAGLE；
- DeepSeek V4 CP-v2 + EAGLE；
- DeepSeek V4 CP-v2 + DSpark。

这不等于任意模型、任意 speculative algorithm 都兼容。draft/target 的 forward mode、aux hidden gather 和 graph pool 都有专用处理。

### 12.5 KV/存储与 PD

- 普通 CP 不等于 KV cache token sharding；
- Decode worker 开 DCP 时，当前要求 Prefill worker `attn_cp_size == 1`；`P: Prefill CP>1 -> D: DCP>1` 会在连接校验阶段被拒绝；
- DCP PD 仅支持 `P DCP1 -> D DCPN` 的 MLA/hybrid-MLA token relayout，或 P/D 使用相同 DCP size 且 DCP rank 匹配；
- DCP1→DCPN 由 P sender 选择目标 rank 拥有的 token rows，直接写入 D local pool，不会在 D 端先接收 full KV 再切分；
- DSA LayerSplit 目前只支持 Mooncake transfer family；
- disaggregation staging buffer 不支持 prefill CP；
- HiCache/file key 会加入 CP rank/size，避免不同 CP shard 写冲突；
- PD 的 prefill/decode topology 必须让 bootstrap 正确映射 TP/CP/DP/PP ranks。

---

## 13. 性能分析方法

### 13.1 一个简化模型

每层、每 rank 的 CP 时间可粗略写成：

```text
T_cp ≈ T_qkv_proj(L/C)
     + T_kv_allgather(L, kv_width, C)
     + T_attention(L^2/C)          # dense causal
     + T_mlp_or_moe_layout(L/C or L)
     + T_padding_and_reorder
```

普通 TP 则近似：

```text
T_tp ≈ T_qkv_proj(L, heads/TP)
     + T_attention(L^2, heads/TP)
     + TP collectives
```

CP 是否赢取决于模型的 head/KV width、TP 与 CP 的替换关系、序列长度、互联带宽和 MoE layout。

### 13.2 建议测的指标

固定模型和并发，至少做以下矩阵：

```text
prompt length: 1K, 4K, 8K, 16K, 32K, 64K, ...
chunk size:    4K, 8K, 16K, ...
CP degree:     1, 2, 4, 8
batch/concurrency: 1, 低并发, 目标线上并发
deployment: unified vs PD-prefill
```

记录：

- TTFT p50/p90/p99；
- prefill tokens/s；
- decode ITL/TPOT 是否退化；
- 单卡 peak allocated/reserved memory；
- NCCL/RCCL/HCCL 时间占比；
- attention、indexer-topk、MoE dispatch/combine 时间；
- padding 后 local physical token 与 logical token 的比值。

### 13.3 如何判断瓶颈

| 现象 | 可能原因 |
|---|---|
| CP2 有收益，CP4/8 反而变慢 | local compute 太少，KV all-gather/launch 开销占主导 |
| 长 prompt 加速明显，短 prompt 退化 | 正常的 break-even 行为 |
| 某个 rank 明显更慢 | zigzag metadata/不等长 batch/padding，或 MoE token 不均衡 |
| prefill 显存下降但 KV capacity 没提升 | 普通 CP 只降 activation，full KV 仍复制，这是预期 |
| unified decode 明显变慢 | `attn_tp_size` 降低、attention weight 沿 CP 复制、额外同步 |
| DSA indexer 占比升高 | full key gather/top-k 成为瓶颈，attention 已不再是主导 |
| 开 CP 后 graph 没命中 | 只有特定 TRTLLM MHA zigzag 组合支持 CP-v2 BCG |

---

## 14. 推荐的源码阅读顺序

### 第一阶段：概念与参数

1. [server_args.py：参数定义](../python/sglang/srt/server_args.py#L1056)
2. [server_args.py：legacy alias 归一化](../python/sglang/srt/server_args.py#L6383)
3. [server_args.py：CP 校验和 strategy 初始化](../python/sglang/srt/server_args.py#L6425)
4. [parallel_state.py：ATTN_CP/ATTN_TP group](../python/sglang/srt/distributed/parallel_state.py#L2451)

读完应能回答：`tp_size、attn_dp_size、attn_cp_size、attn_tp_size` 如何相乘，哪些 ranks 在一个 CP group。

### 第二阶段：通用 CP-v2 数据流

1. [cp/base.py：strategy interface](../python/sglang/srt/layers/cp/base.py#L93)
2. [cp/utils.py：prepare/shard/gather](../python/sglang/srt/layers/cp/utils.py#L132)
3. [cp/padding.py：logical vs physical token](../python/sglang/srt/layers/cp/padding.py#L24)
4. [eager_runner.py：runner 边界](../python/sglang/srt/model_executor/runner/eager_runner.py#L251)

读完应能画出“全局 batch -> local transformer body -> 全局 logits”的流程。

### 第三阶段：先精读 zigzag

1. [cp/zigzag.py：文件头布局示意](../python/sglang/srt/layers/cp/zigzag.py#L15)
2. [build_metadata](../python/sglang/srt/layers/cp/zigzag.py#L121)
3. [shard/gather](../python/sglang/srt/layers/cp/zigzag.py#L301)
4. [run_attention](../python/sglang/srt/layers/cp/zigzag.py#L343)
5. [收集完整 K/V 并写入 cache 的实现](../python/sglang/srt/layers/cp/zigzag.py#L396)
6. [FA backend CP dispatch](../python/sglang/srt/layers/attention/flashattention_backend.py#L1389)

建议用 `C=2, L=8` 手算：

```text
blocks: b0 b1 b2 b3
rank0: b0 b3
rank1: b1 b2
all-gather rank-major: b0 b3 b1 b2
reverse: b0 b1 b2 b3
```

### 第四阶段：再读 DSA interleave

1. [cp/interleave.py](../python/sglang/srt/layers/cp/interleave.py#L60)
2. [dsa/utils.py：round-robin split](../python/sglang/srt/layers/attention/dsa/utils.py#L150)
3. [dsa_indexer.py：full index key gather](../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L475)
4. [dsa_backend.py：metadata 与 local request lens](../python/sglang/srt/layers/attention/dsa_backend.py#L800)
5. [communicator_dsa_cp.py](../python/sglang/srt/layers/communicator_dsa_cp.py#L71)

### 第五阶段：看模型集成与复杂功能

1. Qwen3 MoE： [qwen3_moe.py](../python/sglang/srt/models/qwen3_moe.py#L995)
2. DeepSeek V2/V3： [deepseek_v2.py](../python/sglang/srt/models/deepseek_v2.py#L2805)
3. DeepSeek V4： [deepseek_v4.py](../python/sglang/srt/models/deepseek_v4.py#L2455)
4. prefill breakable graph： [cp/bcg.py](../python/sglang/srt/layers/cp/bcg.py#L43)
5. DSA LayerSplit： [dsa_cache_layer_split.py](../python/sglang/srt/mem_cache/dsa_cache_layer_split.py#L249)
6. decode weight slicing： [cp_decode_attn_tp.py](../python/sglang/srt/layers/cp/cp_decode_attn_tp.py#L50)

---

## 15. 调试清单

### 15.1 启动时先确认

```text
最终 tp_size / pp_size / dp_size / ep_size
最终 attn_tp_size / attn_cp_size / attn_dp_size
cp_strategy
SGLANG_ENABLE_CP_V2 的实际值
prefill/decode attention backend
DSA 的 prefill/decode 子 backend
moe_a2a_backend / moe_dp_size / moe_dense_tp_size
prefill CUDA graph 是否真的启用
```

不要只相信命令行，因为模型 override 会重写 resolved config。

### 15.2 forward 时可断点/日志的位置

```text
is_cp_v2_active
prepare_cp_forward
strategy.build_metadata
strategy.shard_hidden_states
attention backend materialize_full_*kv
strategy.run_attention
DSACPLayerCommunicator gather/reduce-scatter
cp_gather_after_forward
```

重点打印：

```text
forward_mode
extend_seq_lens_cpu
seq_lens_cpu
prefix lengths
per_rank_logical_token
per_rank_actual_token
local hidden/q/k/v shape
gather 后 full kv shape
恢复后 hidden shape/token order
```

### 15.3 正确性不一致时

按以下顺序缩小问题：

1. 固定 deterministic input，CP1 与 CP2 比 logits；
2. 关闭 speculative decoding；
3. 关闭 radix cache，排除 prefix slot 映射；
4. 使用单 request、整除长度；
5. 再加入非整除长度，验证 padding；
6. 再加入 prefix cache；
7. 再加入 multi-request；
8. 最后恢复 MoE A2A、spec、graph 和 PD transfer。

仓库里的 strategy CPU unit tests 很适合先理解和验证 token order：[test_cp_strategy_unit.py](../test/registered/cp/test_cp_strategy_unit.py#L1)。

---

## 16. 常见问题

### Q1：Prefill CP 会让模型支持更长 context 吗？

它能降低每卡当前 prefill activation 和 query 计算压力，因此可能让更大的 prefill chunk/更长 prompt 可运行。但普通路径中，每个 CP rank 通常仍会收集并保存完整 K/V，所以最大 KV capacity 不一定按 `C` 倍增长。要降低 DSA KV 常驻内存，需看 LayerSplit；其他模型则要结合 KV dtype、page size、HiCache 或 DCP/PD 设计。

### Q2：CP degree 越大越好吗？

不是。CP 增大使 local query/activation 下降，但 KV all-gather、padding、同步占比上升，而且 `attn_tp_size` 会下降。最优 degree 取决于序列长度和互联。

### Q3：为什么 zigzag 要切 `2*C` 段，而不是 `C` 段？

为了把一个早期低 causal work block 与一个晚期高 work block 配对，使每 rank 的 attention 工作接近，而不只是 token 数相等。

### Q4：为什么 interleave 不也调用两次 attention？

当前 interleave 主要服务 DSA，DSA backend 自己基于 local query metadata 执行 indexer/top-k/稀疏 attention；strategy 的 `run_attention()` 明确是 no-op。

### Q5：`--enable-dp-attention` 和 CP 冲突吗？

不能一概而论。数学拓扑允许 `DP × CP × attention-TP`，但具体 strategy/model communicator 有限制。当前 DSA interleave 自动配置要求有效 `dp_size==1`；zigzag DSA/MLA 有 DP2/CP4 的 CI。必须按模型 recipe。

### Q6：PP 和 CP 能一起用吗？

能，但有条件。Qwen3 MoE 已有 PP2×CP2 CI。`moe_dp_size>1` 的通用校验当前禁止 PP；DSA 的 CP collective 仍要求单机 TP group，PP stage 可以跨机。

### Q7：为什么打开 CP 后短请求/统一部署 decode 变慢？

短请求节省的计算不足以覆盖 KV/hidden gather；decode 又不能按 prompt token 并行，同时固定 world size 下 attention TP degree 被 CP degree 降低。官方 GLM 文档也明确提示 CP 会增加 short prefill/unified decode latency。

### Q8：应该从 CP-v1 还是 CP-v2 学？

先学 CP-v2，它把核心概念集中在 strategy 和 runner 边界。理解后再看 CP-v1，主要用于读 DSA zigzag、旧平台 backend 和迁移兼容逻辑。

---

## 17. 一页复习图

```text
CLI
  --attn-cp-size C
  --enable-prefill-cp
  --cp-strategy zigzag|interleave
          │
          ▼
ServerArgs normalize + model overrides
          │
          ├─ attn_tp = tp / (attn_dp × attn_cp)
          ├─ create ATTN_CP group
          └─ init CP strategy
          │
          ▼
ForwardMode.EXTEND / MIXED
          │
          ▼
prepare_cp_forward
  build layout metadata
  logical -> physical padding
          │
          ▼
global input ids / positions
          │ embedding
          ▼
shard hidden + positions by CP rank
          │
          ▼
Transformer body on local tokens
  each attention layer:
    local Q/K/V
    local K/V --all-gather+reorder--> full KV cache
    local Q attends full visible KV
    output remains local-token layout
  attention <-> MoE may gather/reduce-scatter
          │
          ▼
last-rank hidden all-gather + restore natural token order
          │
          ▼
lm_head / logits / sampling use normal global batch semantics
```

一句话记忆：

> SGLang Prefill CP 用 CP ranks 分担同一长 prompt 的 query/hidden token，靠 KV/hidden collective 保持完整上下文语义；收益来自长 prefill 计算与 activation 分摊，代价是通信、padding、并行布局转换以及可能的 decode 退化。

---

## 18. 为什么很少在同一个服务中同时开启 Prefill CP 和 DCP

先给本节结论：

> SGLang 能明确区分 Prefill、Decode 和 Mixed forward；“很少同时开启 PCP 与 DCP”并不是因为引擎分不清两个阶段。真正的问题是：forward mode 可以逐轮切换，但 GPU 通信组、权重切分方式和 KV cache 布局是服务启动时确定的，不能到 Prefill 临时变成 PCP、到 Decode 再无代价地重组为 DCP。两者同时开启会形成一套更复杂的二维并行布局，而且当前源码没有给出经过完整 E2E 验证的通用 PCP+DCP recipe。

这里将 Prefill Context Parallel 简写为 `PCP`，避免它与 Decode Context Parallel 的 `DCP` 混在一起。

### 18.1 先理解 `EXTEND`、`DECODE` 和 `MIXED`

SGLang 源码没有名为 `PREFILL` 的 `ForwardMode`，通常所说的 Prefill 在代码里叫 `EXTEND`。三种最重要的模式定义在 [ForwardMode](../python/sglang/srt/model_executor/forward_batch_info.py#L98)：

| Forward mode | 通俗含义 | 一个请求本轮新增的 query token 数 | 是否可能进入 PCP |
| --- | --- | ---: | --- |
| `EXTEND` | 为一个或多个请求计算尚未进入 KV cache 的 prompt/suffix | 通常大于 1，也可能因为 prefix cache 只剩很短的未命中部分 | 是 |
| `DECODE` | 为正在生成的每个请求再计算一个 token | 通常每请求 1 个 | 否 |
| `MIXED` | 同一个 batch 中同时装入一些 Prefill 请求和一些 Decode 请求 | Prefill 请求是多个，Decode 请求是 1 个 | 源码把它视为 CP extend，但 strategy 还要继续判断是否可切分 |

要特别注意：

> `MIXED` 不是同一个请求同时处于 Prefill 和 Decode。自回归依赖决定了一个请求必须先算完它当前的 prompt，才能生成下一个 token。`MIXED` 指的是同一轮 GPU forward 中，请求 A/B 正在 Decode，而另一个请求 C 正在 Prefill。

#### `EXTEND`：代码中的 Prefill

假设新请求 C 有 10 个 prompt token，而且没有 prefix cache 命中：

```text
请求 C：
old KV = 空
本轮 query = c0 ... c9
本轮写入 KV = c0 ... c9
ForwardMode = EXTEND
```

如果前 6 个 token 已命中 prefix cache，那么这轮不需要重新计算它们：

```text
请求 C：
old KV = c0 ... c5
本轮 query = c6 ... c9
本轮写入 KV = c6 ... c9
ForwardMode = EXTEND
```

所以 `EXTEND` 比“从头 Prefill 整个 prompt”更准确：它表示把一段尚未计算的 token 接到已有序列后面。chunked prefill 也会把很长 prompt 切成多个 `EXTEND` forward。

#### `DECODE`：每个请求向前生成一步

请求 C 完成 Prefill 后，下一轮只输入一个新 token，并查询此前积累的 KV：

```text
请求 C：
old KV = c0 ... c9
本轮 query = c10
本轮写入 KV = c10
ForwardMode = DECODE
```

如果 batch 中有 128 个正在生成的请求，通常是每个请求各提供一个 query token，而不是某一个请求一次向前生成 128 个 token：

```text
Decode batch = A 的 1 token + B 的 1 token + ... + 第 128 个请求的 1 token
```

这也是 Prefill CP 不适合普通 Decode 的根本原因：单个请求本轮只有一个 token，无法再沿这个请求的 token 维均匀切给多个 PCP ranks。

#### `MIXED`：不同阶段的请求共用一次 forward

开启 chunked prefill 与 `--enable-mixed-chunk` 后，调度器可以把新 Prefill chunk 与已经运行的 Decode 请求合并。源码在 [scheduler.py](../python/sglang/srt/managers/scheduler.py#L3430) 调用 [mix_with_running](../python/sglang/srt/managers/schedule_batch.py#L2744)，并把 mode 设置为 `MIXED`。

例如：

```text
同一个 MIXED batch

请求 A：old KV 很长，本轮新增 1 个 Decode token
请求 B：old KV 很长，本轮新增 1 个 Decode token
请求 C：本轮处理 1024 个 Prefill token
```

这些 token 可以在内存里打包到一个 batch，但 attention metadata 会保存每个请求自己的 query 范围、prefix 长度、KV page table 和 causal 边界：

```text
A 的 query 只能看 A 的 KV
B 的 query 只能看 B 的 KV
C 的 query 只能看 C 的 prefix 和 C 当前 chunk 中位于它之前的 token
```

请求之间不会互相做 attention。可以把 `MIXED` 理解成一次 GPU forward 中包含多种 `q_len`：

```text
q_len = [1, 1, 1024]
```

Mixed chunk 的主要价值是调度折中：处理长 Prefill chunk 时，顺便让已有请求继续 Decode，避免它们等完整个 Prefill forward；同时也能提高一次 forward 的 token 利用率。代价是 metadata、kernel shape 和结果处理都更复杂，因此它默认关闭，且 return logprob、input embeddings、speculative decoding 等功能还有各自限制。启动参数定义见 [server_args.py](../python/sglang/srt/server_args.py#L973)。

如果不开 Mixed chunk，统一服务仍然会区分 Prefill 和 Decode，只是不同调度轮次通常运行纯 `EXTEND` batch 或纯 `DECODE` batch：

```text
step 1：EXTEND，新请求做 Prefill
step 2：DECODE，已有请求各生成一步
step 3：EXTEND，又有新请求进入
step 4：DECODE，已有请求继续生成
```

因此，“统一服务”不等于“每个 forward 都是 MIXED”。

### 18.2 三种 mode 分别怎样使用 PCP 和 DCP

可以先用下表建立整体认识：

| Forward mode | PCP 的作用 | DCP 的作用 |
| --- | --- | --- |
| `EXTEND` | 把当前请求的多个新 query/hidden token 分给 PCP ranks | 按 DCP owner 规则写入分片 KV；如果已有 prefix KV，还要在分片 prefix 上完成正确 attention |
| `DECODE` | 不进行 Prefill token 切分 | 每个 DCP rank 查询自己保存的部分 KV，再通过 output/LSE 通信合并完整 attention 结果 |
| `MIXED` | 理论上是 CP extend，但是否真的启用取决于 strategy；Zigzag 常被其中的单 token Decode 请求挡住 | 同时处理 Prefill token 的分片 KV 写入和 Decode token 的分片 KV 查询 |

PCP-v2 的运行判断明确要求 mode 是 `EXTEND` 或 `MIXED`，而普通 `DECODE` 不会进入：[is_cp_v2_active](../python/sglang/srt/layers/cp/utils.py#L139)。

DCP 则不能等到 `DECODE` 时才突然开启。为了让后续 Decode 每个 rank 只保存约 `1 / dcp_size` KV，KV 在 `EXTEND` 阶段生成时就必须按以下规则写入 owner rank：

```text
owner(token_position) = token_position % dcp_size
local_cache_index      = global_cache_index // dcp_size
```

Triton backend 的 [_set_kv_buffer](../python/sglang/srt/layers/attention/triton_backend.py#L1245) 会在写 KV 时计算 owner mask，源码也单独实现了 [_forward_extend_dcp](../python/sglang/srt/layers/attention/triton_backend.py#L1476)。因此，`Decode Context Parallel` 的主要收益虽然在 Decode，但其 KV cache 布局从 Prefill 写入阶段就已经生效。

这解释了为什么同时开启两者不是简单的：

```text
错误的直觉：
Prefill 时只开 PCP ──完成后──> 关闭 PCP，Decode 时再开 DCP

真实需要：
Prefill 时：PCP 切 query/hidden
          + DCP 决定新 KV 最终由哪个 rank 保存
Decode 时：DCP 查询分片 KV 并合并结果
          + PCP ranks 仍属于启动时固定的并行拓扑
```

也就是说，两者同时开启后，Prefill 阶段已经需要组合 PCP token layout 与 DCP KV ownership，而不是到 Decode 才开始接触 DCP。

### 18.3 `MIXED` 对 Prefill CP 的特殊影响

源码把 `MIXED` 归入 `is_context_parallel_extend()`，但这不代表所有 PCP strategy 都能处理它。

Zigzag 的 `can_apply()` 要求 batch 中每一个请求本轮的 `extend_len` 都至少为 `2 * cp_size`：[zigzag.py](../python/sglang/srt/layers/cp/zigzag.py#L109)。而调度器合并 Decode 请求时，会把每个 Decode 请求的 `extend_len` 设为 1：[schedule_batch.py](../python/sglang/srt/managers/schedule_batch.py#L2748)。于是：

```text
PCP size = 2
Zigzag 最小要求 = 每请求 extend_len >= 4

MIXED q_len = [1, 1, 1024]
               ↑  ↑
             Decode 请求不满足要求

结果：这个 batch 的 Zigzag PCP 通常不会启用
```

这是统一服务同时使用 PCP、DCP、Mixed chunk 时非常实际的问题：长 Prefill 请求明明有 1024 个 token 可以切分，却可能因为同 batch 中存在 `q_len=1` 的 Decode 请求，使整个 Zigzag strategy 回退到非 PCP 路径。

Interleave 的 `can_apply()` 主要检查 batch 总 token 数是否不少于 `cp_size`，不会要求每个请求都达到 `2 * cp_size`：[interleave.py](../python/sglang/srt/layers/cp/interleave.py#L64)。但当前 CP-v2 Interleave 主要是 DSA 专用路径，不能把它当作普通 dense attention 的通用替代方案。

所以从 PCP 的角度看：

- 纯 `EXTEND` 长 Prefill batch 最容易获得稳定收益；
- `MIXED` 在调度上有利于 Decode ITL，但可能妨碍 Zigzag PCP 生效；
- 是否打开 Mixed chunk，需要在“Decode 不被长 Prefill 阻塞”与“长 Prefill 能否真正使用 PCP”之间权衡。

### 18.4 为什么当前不把 PCP+DCP 作为默认推荐组合

这里的“不建议”不是说数学上绝对错误，也不是说参数一定无法启动，而是说在当前 commit 下，不应把它当成已经成熟验证的通用配置。原因按重要性排列如下。

#### 原因一：当前没有通用 E2E 测试或官方 recipe 覆盖这个组合

当前仓库可以分别找到 PCP 与 DCP 的测试、文档和模型 recipe，但没有找到同一个服务同时设置 `--enable-prefill-cp`/`--attn-cp-size` 与 `--dcp-size` 的注册 E2E 用例。

这意味着当前最多能说：

```text
CLI 参数没有显式互斥
并行组在满足条件时可以构造
某些底层路径分别支持 PCP 或 DCP
```

但不能据此推出：

```text
任意 PCP 模型 × 任意 DCP backend × EXTEND/DECODE/MIXED
都已经验证正确且性能合理
```

对于推理引擎，这一点本身就足以让它不适合作为默认生产建议。

#### 原因二：两者不是独立开关，而是二维数据布局的组合

PCP 沿 prompt/query token 分工，DCP 沿持久 KV token 分工。二者同时开启时需要同时保证：

1. PCP local hidden/Q/K/V 的 token 顺序正确；
2. PCP collective 恢复了 attention 需要的上下文语义；
3. 新 KV 最终只写入正确的 DCP owner；
4. DCP attention 对各 rank 的 partial output/LSE 做精确合并；
5. prefix cache 的逻辑 slot、DCP physical slot 与 PCP token reorder 一致；
6. `EXTEND`、`DECODE`、`MIXED` 三种 metadata 都匹配对应 backend。

两套功能单独正确，不代表它们的数据 layout 组合后自然正确。

#### 原因三：PCP 会减少留给 DCP 的 attention-TP 空间

当前并行组计算为：

```text
attn_tp_size = tp_size / (attn_dp_size * pcp_size)
```

见 [parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2451)。DCP group 还必须完整落在一个 attention-TP group 内，因此组合时至少应满足：

```text
attn_tp_size % dcp_size == 0
```

近似写成：

```text
tp_size % (attn_dp_size * pcp_size * dcp_size) == 0
```

例如：

```text
TP=8, attention DP=1, PCP=2
=> attn_tp_size=4
=> DCP 只能在这 4-rank attention-TP group 内选择可整除的 degree
```

PCP degree 和 DCP degree 的乘积很快就会要求很大的总 TP。当前启动检查对组合拓扑还不够严格，只检查原始 `tp_size % dcp_size == 0`；因此“服务能启动”也不一定说明 DCP group 没有跨越 attention-TP/DP 边界。DCP 文档对 group containment 的说明见 [dcp.mdx](../docs/docs/advanced_features/dcp.mdx#L22)。

#### 原因四：PCP ranks 到了 Decode 阶段可能没有对应收益

PCP 为同一请求的多个 Prefill token准备了多个 ranks，但普通 Decode 每请求只有一个新 token，无法继续按 token 维拆分。固定 PCP 拓扑仍然存在，于是 attention 权重可能沿 PCP 维复制，或者产生重复/低利用率的 Decode 工作。

源码提供了实验性的 `--enable-cp-decode-attn-tp`：在非 PCP forward 中临时把部分复制的 attention 权重切成 TP shard，从而提高 PCP ranks 在 Decode 阶段的利用率；但当前 whitelist 只覆盖少量 GLM-5.x/DeepSeek-V4 架构，见 [cp_decode_attn_tp.py](../python/sglang/srt/layers/cp/cp_decode_attn_tp.py#L15)。这反过来也说明“PCP 拓扑上的 Decode 如何高效执行”不是所有模型都已经通用解决的问题。

DCP 自己还需要在 Decode 中交换 query 或 partial output/LSE。PCP 的 Decode 副作用与 DCP 通信不会自动互相抵消，开在一起可能只是叠加两边的代价。

#### 原因五：`MIXED` 可能让 Zigzag PCP 直接不生效

如 18.3 所述，Mixed batch 中的 Decode 请求具有 `extend_len=1`，而 Zigzag 要求每个请求至少为 `2 * pcp_size`。这会形成一个不理想状态：

```text
为了改善 Decode ITL 开启 Mixed chunk
        ↓
Mixed batch 中加入单 token Decode 请求
        ↓
Zigzag PCP 对整个 batch 回退
        ↓
承担了 PCP 固定拓扑的 Decode 代价，却没有在该 Prefill batch 得到 PCP 收益
```

对于 DSA Interleave 可能有专用实现空间，但这不能解决普通 Zigzag 模型的通用问题。

#### 原因六：两者的最佳部署位置概念上不同，但当前 PD 还不能把两者直接拼起来

PCP 最关注长 Prompt 的 TTFT、Prefill activation 和计算吞吐；DCP 最关注长上下文 Decode 的 KV capacity、并发量与 ITL。从系统设计上看，PD 分离似乎应该允许两组 worker 独立选择：

```text
Prefill worker：按长 Prompt 选择 PCP degree
Decode worker ：按 KV capacity/ITL 选择 DCP degree
```

但当前实现尚未接通这条组合：D worker 开 DCP 时强制 P worker `attn_cp_size==1`，因此上面的“P PCP + D DCP”目前会在 bootstrap/connection 阶段失败。当前 PD 只能选择 P PCP+DCP1，或 P CP1→D DCPN（以及两端相同 DCPN）。

所以 PD 在架构上仍是未来组合两者的自然位置，但在当前 commit 上不能把它当成已经可用的 PCP+DCP 方案。

### 18.5 是否存在值得同时尝试的场景

存在，但更适合视为需要逐项验证的实验配置，而不是默认方案。至少应同时满足：

- 业务必须采用 unified serving，无法做 PD 分离；
- Prompt 很长，纯 PCP 实测能显著改善 TTFT/Prefill OOM；
- Decode 上下文也很长，纯 DCP 实测能显著增加 KV capacity 或并发；
- 模型同时落在 PCP 与 DCP 支持范围内；
- prefill、decode 以及 DSA 子 backend 都支持所需数据布局；
- `dcp_size` 完整嵌套在一个 `attn_tp_size` group 内；
- 高带宽互联足以承担 PCP all-gather 与 DCP output/LSE merge；
- 愿意对 `EXTEND`、`DECODE`、`MIXED`、prefix-cache hit/miss 分别做正确性对比。

建议按以下顺序验证，不要直接只测组合：

```text
基线：PCP=1, DCP=1
  ↓
只开 PCP：确认长 Prefill 正确性、TTFT 和显存
  ↓
只开 DCP：确认 Decode logits、ITL 和 KV capacity
  ↓
同时开启但先关闭 Mixed chunk：分别验证纯 EXTEND 与纯 DECODE
  ↓
最后开启 Mixed chunk：确认 PCP 是否实际进入、MIXED logits 与性能
```

重点观测的不只是“能不能启动”，还包括：

```text
is_cp_v2_active 在 EXTEND/MIXED 中是否为 True
每种 mode 的 attn_cp_metadata / DCP metadata
各 rank 实际 KV cache 占用
PCP all-gather 与 DCP merge 通信量
TTFT / TPOT(ITL) / throughput
prefix cache hit 后的正确性
PCP-only、DCP-only、PCP+DCP 的 logits 对齐
```

最终可以这样记忆：

> `EXTEND/DECODE/MIXED` 是每轮 forward 的运行模式，可以动态变化；PCP/DCP 是启动时确定的并行与 cache 布局，不能随 mode 无代价重组。PCP+DCP 理论上可以构造，但当前缺少通用 E2E 验证，存在拓扑挤占、Mixed 回退、Decode 利用率和二维 cache layout 等现实问题。PD 在架构上是更自然的承载位置，但当前明确不支持 `P: PCP>1 -> D: DCP>1`，因此部署时只能分别选择 PCP 优先或 DCP 优先的已支持拓扑，不能把“优先考虑 PD 分离”误解成两者已经能够直接组合。

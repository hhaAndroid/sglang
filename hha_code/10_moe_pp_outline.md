# 10. MoE PP 学习大纲：layer 维度切分和 pipeline stage

本文先写学习大纲，后面真正学习 PP 时再补细节。

PP = Pipeline Parallel。

它切的不是 head、hidden、expert，也不是 request，而是：

```text
Transformer layers
```

也就是把模型层按 stage 切开：

```text
PP stage0:
  embedding
  layer 0..15

PP stage1:
  layer 16..31

PP stage2:
  layer 32..47

PP stage3:
  layer 48..63
  norm / lm_head
```

## 0. 先给结论

PP 相对独立，可以放在 TP / EP / DPA / DeepEP 后面学。

它主要解决：

```text
模型层太多 / 参数太大
单靠 TP/EP 不适合继续横向扩展
多机 TP 通信太重
希望只在 stage 边界传 hidden states
```

在 MoE 模型里，PP 和前面并行维度的关系是：

```text
PP:
  layer 维度切分

TP / DPA / CP:
  每个 PP stage 内 attention 怎么并行

EP / DeepEP / EPLB:
  每个 PP stage 内 MoE experts 怎么并行
```

所以 PP 是一个外层维度。

## 1. 启动参数大纲

核心参数：

```text
--pp-size
```

常见多机例子：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --tp-size 8 \
  --pp-size 4 \
  --nnodes 4 \
  --node-rank 0 \
  --dist-init-addr $NODE0_IP:20000 \
  --host 0.0.0.0 \
  --port 30000
```

每个节点用不同：

```text
--node-rank 0
--node-rank 1
--node-rank 2
--node-rank 3
```

外部 HTTP 请求通常仍然发给暴露 server 的入口节点 / router。

内部 PP stage 之间传的是：

```text
hidden states
PPProxyTensors
metadata
```

## 2. PP group 怎么理解

假设：

```text
tp_size = 8
pp_size = 4
```

总 GPU 数通常可以理解成：

```text
world_size = tp_size * pp_size = 32
```

直观上：

```text
PP stage0:
  8 张卡做 TP/EP/DPA 等 stage 内并行

PP stage1:
  8 张卡做 TP/EP/DPA 等 stage 内并行

PP stage2:
  8 张卡做 TP/EP/DPA 等 stage 内并行

PP stage3:
  8 张卡做 TP/EP/DPA 等 stage 内并行
```

每个 stage 只保存部分 layers。

## 3. 一个请求怎么流动

简化流程：

```text
HTTP request
  |
  v
scheduler / first PP stage
  |
  v
PP stage0:
  embedding + early layers
  |
  |  send hidden_states
  v
PP stage1:
  middle layers
  |
  |  send hidden_states
  v
PP stage2:
  middle layers
  |
  |  send hidden_states
  v
PP stage3:
  final layers + norm + lm_head
  |
  v
logits / sampled token
```

PP 的通信发生在：

```text
stage boundary
```

而不是每一层内部。

每个 stage 内部仍然可以有：

```text
attention TP all-reduce
DeepEP dispatch/combine
DPA request routing
CP prefill token split
```

## 4. 和 MoE 的关系

PP 不直接决定 expert 怎么放。

如果某个 PP stage 包含 MoE layers：

```text
这个 stage 内部会初始化它自己的 MoE experts
这些 experts 再按 ep_size / DeepEP / EPLB 组织
```

所以要分清：

```text
PP:
  哪些 layers 在哪个 stage

EP:
  某个 MoE layer 的 experts 在 stage 内怎么分布
```

## 5. layer partition

SGLang 有环境变量：

```text
SGLANG_PP_LAYER_PARTITION
```

用于手动指定每个 PP stage 分多少层。

官方 PP 文档里提到：

```text
如果 layers 不能均分
把更多层放在更高 PP rank 有时更好
```

例如：

```text
SGLANG_PP_LAYER_PARTITION=15,15,15,16
```

后续详细学习时要看：

```text
python/sglang/srt/distributed/utils.py
```

里面有 layer partition 计算逻辑。

## 6. PP 和 chunked prefill

PP 的效率很依赖 pipeline 是否被填满。

如果只有一个完整大 batch 从 stage0 跑到 stage3，中间会有 pipeline bubble。

官方文档强调：

```text
Dynamic Chunked Prefill
Micro-batching Event Loop
non-blocking async P2P communication
```

这些都是为了：

```text
让不同 micro-batches / chunks 同时占满不同 PP stages
减少 pipeline bubble
```

后续详细学习 PP 时，需要重点看：

```text
chunked prefill size
micro-batch
stage 间 send/recv
PP bubble
```

## 7. PP 和 CP / PD 的关系

PP 可以和 CP、PD 组合，但复杂度会明显上升。

### 7.1 PP + CP

CP 切 sequence：

```text
同一个长 prompt 被多个 CP rank 切开
```

PP 切 layers：

```text
同一个 token shard 继续穿过多个 PP stages
```

所以 PP + CP 要同时处理：

```text
stage boundary hidden states
CP rank 的 token layout
KV cache layout
```

### 7.2 PP + PD

PD 分离后：

```text
prefill cluster 可以有自己的 pp_size
decode cluster 可以有自己的 pp_size
```

源码里 PD 连接逻辑会处理：

```text
prefill pp_size 和 decode pp_size 的匹配/切片
```

这部分留到第 11 篇 PD 分离学。

## 8. 后续要重点回答的问题

真正学习 PP 时，需要回答：

```text
1. SGLang 是如何把 layers 分配到不同 pp_rank 的？

2. embedding / norm / lm_head 分别在哪些 PP rank？

3. PPMissingLayer 是干嘛的？

4. PPProxyTensors 在 stage 间传什么？

5. scheduler 是一个进程还是每个 PP rank 都有？

6. stage 间 P2P 通信在哪里发起？

7. decode 阶段 PP 是怎么流水的？

8. prefill chunk 怎么填满 pipeline？

9. PP 和 DeepEP 同时开时，DeepEP group 是每个 stage 内一套，还是跨 stage？

10. PP 和 PD disaggregation 同时开时，KV transfer 怎么处理？
```

## 9. 源码阅读路径

后续详细学习时建议按这个顺序：

```text
1. 官方文档
   docs/advanced_features/pipeline_parallelism.md

2. PP group 初始化
   python/sglang/srt/distributed/parallel_state.py
   get_pp_group()

3. layer partition
   python/sglang/srt/distributed/utils.py
   get_pp_indices()
   SGLANG_PP_LAYER_PARTITION

4. 模型里的 PP 切层
   python/sglang/srt/models/transformers.py
   pipeline_parallel()
   PPMissingLayer

5. stage 间数据结构
   python/sglang/srt/model_executor/forward_batch_info.py
   PPProxyTensors

6. scheduler PP loop
   python/sglang/srt/managers/scheduler_pp_mixin.py

7. 具体 MoE 模型支持
   python/sglang/srt/models/qwen3_moe.py
   python/sglang/srt/models/deepseek_v2.py
   python/sglang/srt/models/deepseek_v4.py
```

## 10. 学习优先级

对你当前路线：

```text
优先级：中高，但可以放在 EPLB 后面
```

原因：

```text
PP 是很正统的模型并行维度
但它和 MoE expert dispatch 的关系相对间接
你目前更关心 RL rollout 和 MoE 并行，EPLB / PD 可能更贴近业务
```

建议后面学习顺序：

```text
09 EPLB
10 PP
11 PD 分离
```

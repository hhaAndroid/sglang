# 11. PD 分离学习大纲：Prefill / Decode Disaggregation

本文先写学习大纲，后面真正学习 PD 分离时再补细节。

PD = Prefill / Decode Disaggregation。

它不是模型内部的一种 tensor 并行，而是 serving 架构层面的拆分：

```text
Prefill server:
  负责长 prompt prefill
  计算 prompt 的 KV cache

Decode server:
  负责逐 token decode
  使用 prefill 产生的 KV cache 继续生成
```

中间关键动作是：

```text
KV cache transfer
```

## 0. 先给结论

PD 分离对 RL rollout 很重要，因为 rollout 里常见两类压力：

```text
prefill:
  prompt 长短差异大
  TTFT / 首 token 延迟高

decode:
  输出 token 多
  需要稳定吞吐
```

如果 prefill 和 decode 混在同一个 engine：

```text
长 prompt prefill 可能阻塞 decode
decode 长尾也会影响新请求 prefill
```

PD 分离后可以：

```text
prefill 和 decode 独立扩容
prefill 用适合长上下文的配置
decode 用适合高吞吐 decode 的配置
中间通过 KV transfer 连接
```

## 1. 和前面并行文档的关系

PD 是外层 serving 架构。

它可以包住前面这些并行：

```text
Prefill cluster:
  TP / DPA / EP / DeepEP / CP / PP

Decode cluster:
  TP / DPA / EP / DeepEP / PP
```

也就是说：

```text
TP/EP/DPA/CP/PP:
  单个 engine 内部怎么并行

PD:
  prefill engine 和 decode engine 怎么分开
```

## 2. 最小启动形态

DeepSeek V3.2 文档里有 PD 分离例子，形式大致是：

```bash
# prefill server
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 \
  --port 30000
```

```bash
# decode server
python -m sglang.launch_server \
  --model-path $MODEL \
  --tp-size 8 \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --port 30001
```

通常前面还会有 router：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  ...
```

外部业务请求不直接理解 prefill/decode 两步，而是发给 router / gateway：

```text
client -> router -> prefill -> KV transfer -> decode -> client
```

## 3. 请求生命周期

一个请求进入 PD 系统后，可以先按这个流程理解：

```text
1. client 发请求到 router

2. router 选择 prefill server

3. prefill server 做 prompt prefill

4. prefill server 生成 KV cache

5. decode server bootstrap / prealloc KV slots

6. prefill 把 KV cache transfer 到 decode

7. decode server 开始逐 token decode

8. decode 结果通过 router 返回 client
```

这里最重要的是：

```text
prefill 不是直接返回完整结果
prefill 的核心产物是 KV cache
decode 接管后继续生成
```

## 4. PD 里的几个角色

### 4.1 router / gateway

职责：

```text
接收外部 HTTP 请求
选择 prefill server
选择 decode server
维护请求状态
处理流式返回
```

对于 RL 业务来说，外部通常只关心 router 暴露的 URL。

### 4.2 prefill server

职责：

```text
做 prompt prefill
产生 KV cache
把 KV cache 传给 decode server
```

prefill server 可以更适合：

```text
长上下文
CP
PP
较大的 chunked prefill
```

### 4.3 decode server

职责：

```text
接收 KV cache
做 autoregressive decode
持续生成 token
```

decode server 可以更适合：

```text
DPA
DeepEP
较高 decode 并发
较稳定 batch
```

## 5. KV transfer 是核心

PD 分离真正难点是：

```text
prefill 算出来的 KV cache
必须准确、高效地送到 decode server 对应 KV slots
```

这里会涉及：

```text
bootstrap
KV slot allocation
rank mapping
TP / CP / PP layout
RDMA / IB device
Mooncake / NIXL / other backend
heterogeneous TP
staging buffer
```

后续详细学习时，要重点看：

```text
prefill 和 decode 的 parallel layout 是否一致
如果不一致，KV 怎么重排
```

## 6. 和 DPA + DeepEP 的关系

你的第 6 篇业务配置是单个 engine 内：

```text
DPA8 + DeepEP64
```

如果放进 PD 架构，可能变成：

```text
prefill engine:
  DPA / CP / PP / DeepEP 的某种组合

decode engine:
  DPA8 + DeepEP64
```

但这不是简单复制参数。

PD 下需要额外关心：

```text
prefill tp/dp/cp/pp layout
decode tp/dp/cp/pp layout
KV cache transfer 是否支持
router 如何选择 prefill/decode pair
```

## 7. 和 PP / CP 的关系

### 7.1 PD + CP

CP 通常更偏 prefill：

```text
长 prompt prefill 被多个 CP rank 切开
```

所以 PD + CP 常见理解是：

```text
prefill side 开 CP
decode side 不一定开 CP
```

源码里 PD 连接会处理：

```text
prefill cp_size
decode cp_size
target CP ranks
```

### 7.2 PD + PP

PP 切 layers。

PD + PP 时，KV transfer 要知道：

```text
prefill 的 pp_rank
decode 的 pp_rank
每个 PP stage 对应哪些 layers
```

源码里 disaggregation conn 会处理：

```text
decode pp size 应该等于 prefill pp size 或者为 1
```

这说明 PP + PD 是支持但复杂的组合。

## 8. 后续要重点回答的问题

真正学习 PD 时，需要回答：

```text
1. prefill server 和 decode server 分别启动哪些进程？

2. router 如何知道哪些 prefill/decode server 可用？

3. bootstrap port 是干嘛的？

4. KV cache transfer 的连接是何时建立的？

5. decode server 如何预分配 KV slots？

6. prefill 算完后如何把 KV 写到 decode 对应位置？

7. TP size 不一致时 KV 怎么切换？

8. CP / PP 同时存在时 KV layout 怎么映射？

9. PD 对 latency 和吞吐分别有什么收益和代价？

10. RL rollout 里如何做 request routing 和负载均衡？
```

## 9. 源码阅读路径

后续详细学习时建议按这个顺序：

```text
1. 官方文档
   docs/basic_usage/deepseek_v32.md
   docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md
   docs/references/multi_node_deployment/lws_pd/lws_pd_deploy.md

2. disaggregation mode 参数
   python/sglang/srt/server_args.py

3. PD 连接和 rank mapping
   python/sglang/srt/disaggregation/common/conn.py

4. prefill 侧逻辑
   python/sglang/srt/disaggregation/prefill.py

5. decode 侧逻辑
   python/sglang/srt/disaggregation/decode.py
   python/sglang/srt/disaggregation/*/conn.py

6. KV transfer 工具
   python/sglang/srt/disaggregation/utils.py

7. observability / request timing
   python/sglang/srt/observability/req_time_stats.py

8. router
   python/sglang_router
```

## 10. 学习优先级

对你的场景：

```text
优先级：高
```

原因：

```text
你做 RL rollout
真实系统里 request routing、prefill/decode 长尾、KV transfer 都会影响吞吐
PD 是从单 engine 并行走向多 engine serving 架构的关键
```

建议在 EPLB 后学习：

```text
09 EPLB:
  解决 MoE expert 内部负载不均

10 PP:
  理解 layer 维度跨节点切分

11 PD:
  理解 prefill/decode 分离和多 engine 调度
```

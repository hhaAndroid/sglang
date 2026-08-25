# 09. MoE EPLB 学习大纲：expert load balance 和冗余专家

本文先写学习大纲，后面真正开始学 EPLB 时再补源码细节。

EPLB = Expert Parallelism Load Balancer。

它解决的问题不是“模型怎么切得下”，而是：

```text
MoE token routing 不均匀
某些 experts 很热，某些 experts 很冷
热 expert 所在 GPU 变成长尾
整个 decode / prefill batch 被最慢 rank 拖住
```

所以 EPLB 属于 MoE 并行里的性能调优主题，重要性高于第 7 篇 `Hybrid EP + MoE TP`，更贴近真实生产。

## 0. 先给结论

在大 MoE + DeepEP 场景里，推荐主线通常是：

```text
DPA + DeepEP + EPLB
```

前面几篇文档关系：

```text
05:
  DPA + EP，解决不同请求流和 expert 并行怎么组合

06:
  业务配置 DPA8 + DeepEP64，全流程

09:
  在 DeepEP / EP 已经开起来之后，继续解决 expert 热点和长尾
```

EPLB 不改变用户请求入口：

```text
HTTP URL 仍然是 server / router 暴露的 URL
```

它改变的是内部：

```text
logical expert id -> physical expert location
```

## 1. 要先理解的概念

### 1.1 logical expert

模型配置里的原始 expert id：

```text
expert 0
expert 1
...
expert N-1
```

这些是 logical experts。

router/topk 一开始选出来的也是 logical expert id。

### 1.2 physical expert

实际部署在 GPU rank 上的 expert slot。

在普通 EP 里，通常可以简单理解成：

```text
logical expert 和 physical expert 一一对应
```

但开启 EPLB / redundant experts 后，会出现：

```text
一个 logical expert 可以有多个 physical copies
logical expert id 需要映射到 physical expert location
```

### 1.3 redundant experts

冗余专家。

启动参数里有：

```text
--ep-num-redundant-experts
```

它表示额外分配一些 physical expert slots，用来复制热点 experts。

直观理解：

```text
没有 redundant:
  expert 7 很热
  所有 token 都只能去 expert 7 所在 rank

有 redundant:
  expert 7 可以复制到多个 physical locations
  token 可以被分流到多个 rank
```

## 2. 启动参数大纲

官方 server arguments 里和 EPLB 相关的参数包括：

```text
--enable-eplb
--eplb-algorithm
--eplb-rebalance-num-iterations
--eplb-rebalance-layers-per-chunk
--eplb-min-rebalancing-utilization-threshold

--ep-num-redundant-experts
--ep-dispatch-algorithm
```

后续学习时要搞清楚：

```text
enable-eplb:
  是否开启动态 expert location rebalancing

ep-num-redundant-experts:
  是否增加 physical expert slots

ep-dispatch-algorithm:
  如果一个 logical expert 有多个 physical copies，topk token 具体发给哪个 copy

eplb-rebalance-num-iterations:
  多久触发一次重新均衡

eplb-rebalance-layers-per-chunk:
  每次 forward / chunk 更新多少层，避免一次性搬太多 expert 权重
```

## 3. 和 DeepEP 的关系

DeepEP 负责：

```text
token dispatch / combine
把 token 按 topk 送到 expert rank
```

EPLB 负责：

```text
决定 logical expert 应该放在哪些 physical locations
决定热点 expert 是否复制
决定 token 选择哪个 physical copy
```

所以二者不是替代关系。

更像是：

```text
DeepEP:
  让 token 去 expert 的通信更快

EPLB:
  让 token 去的 expert 分布更均衡
```

## 4. 运行时逻辑大纲

后续源码学习时可以按这个流程看：

```text
1. router/topk 得到 logical expert ids

2. expert distribution recorder 统计每层每个 expert 被选中的次数

3. EPLB 根据统计信息计算新的 expert placement

4. expert location updater 把 expert 权重迁移到新的 physical locations

5. topk ids 从 logical expert ids 映射成 physical expert ids

6. DeepEP / dispatcher 根据 physical expert ids 发送 token
```

关键点：

```text
router 仍然按 logical expert 做语义选择
EPLB 只改变执行位置，不改变模型语义
```

## 5. 一个简化图

```text
router/topk
  |
  |  logical expert ids
  v
EPLB mapping
  |
  |  logical -> physical
  v
physical expert ids
  |
  |  DeepEP dispatch / standard dispatch
  v
expert compute
  |
  v
combine
```

如果 expert 7 很热：

```text
logical expert 7
  -> physical location 7
  -> physical location 70
  -> physical location 91
```

token 可以分流到多个 physical copies。

## 6. 后续要重点回答的问题

真正学习 EPLB 时，需要逐个回答这些问题：

```text
1. EPLB 是静态 placement，还是运行时动态更新？

2. expert activation 统计在哪里记录？

3. rebalance 什么时候触发？

4. expert 权重迁移期间，正在 forward 的请求怎么办？

5. redundant expert 的 physical slot 如何加载权重？

6. topk ids 是在哪里从 logical 转 physical？

7. DeepEP backend 下，physical expert id 如何影响 dispatch？

8. EPLB 对 prefill 和 decode 的收益分别是什么？

9. RL rollout 里 expert 热点是否稳定？rebalance 频率怎么选？

10. 哪些模型已经支持 get_model_config_for_expert_location？
```

## 7. 源码阅读路径

后续详细学习时建议按这个顺序看：

```text
1. 参数
   python/sglang/srt/server_args.py
   docs/advanced_features/server_arguments.md
   docs/advanced_features/expert_parallelism.md

2. expert 统计
   python/sglang/srt/eplb/expert_distribution.py

3. expert location 元数据
   python/sglang/srt/eplb/expert_location.py

4. logical -> physical dispatch
   python/sglang/srt/eplb/expert_location_dispatch.py
   python/sglang/srt/layers/moe/topk.py

5. expert 权重迁移
   python/sglang/srt/eplb/expert_location_updater.py

6. Qwen / DeepSeek MoE 接入
   python/sglang/srt/models/qwen3_moe.py
   python/sglang/srt/models/deepseek_v2.py
   python/sglang/srt/models/deepseek_v4.py

7. FusedMoE 里 physical expert 映射
   python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

## 8. 学习优先级

对于你的 MoE 并行学习路线：

```text
优先级：高
```

原因：

```text
你已经理解了 DPA + DeepEP
下一步真实性能问题大概率不是“怎么切”，而是“负载是否均衡”
EPLB 正好解决 expert routing 长尾
```

但它依赖前置知识：

```text
EP / DeepEP
DPA 请求流
topk routing
physical expert location
```

所以建议后面单独花一轮详细学。

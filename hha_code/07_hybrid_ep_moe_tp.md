# 07. MoE Hybrid EP + MoE TP：expert 数量切分 + expert 内部切分[不支持 deepep 不是重点]

本文学习一个前面没有单独展开的 MoE 并行形态：

```text
Hybrid EP + MoE TP

EP:
  按 expert 数量切分

MoE TP:
  在一个 expert 内部继续按 intermediate 维度切分
```

最典型的例子是：

```text
tp_size = 16
ep_size = 8
moe_dp_size = 1

=> moe_tp_size = tp_size / ep_size / moe_dp_size
               = 16 / 8 / 1
               = 2
```

这个形态和前面 DeepEP 文档里的 `ep_size = tp_size` 不一样。DeepEP / mooncake / nixl / mori / flashinfer / megamoe 这类 A2A 后端在 `server_args.py` 里会把 `ep_size` 调整成 `tp_size`，因此通常会得到：

```text
tp_size = 16
ep_size = 16
moe_dp_size = 1
moe_tp_size = 1
```

所以本文主要讨论的是 `moe_a2a_backend=none` 这种更基础的路径。它不走 DeepEP 的 token all-to-all dispatch/combine，而是更适合用来理解 SGLang 内部 MoE EP 和 MoE TP group 是怎么组织的。

## 0.1 这是不是生产推荐用法

先说结论：对于大 MoE 模型的多机高吞吐部署，尤其是 RL rollout 这类场景，通常应该优先看：

```text
DPA + DeepEP

enable_dp_attention = true
moe_a2a_backend = deepep
ep_size = tp_size
moe_tp_size = 1
```

也就是第 5、6 篇文档里的主线。

本文这个：

```text
moe_a2a_backend = none
ep_size < tp_size
moe_tp_size > 1
```

不是我建议你在大规模 RL rollout 里优先采用的生产形态。它更像是：

```text
1. 学习 SGLang MoE 并行组的关键中间形态
2. 理解 expert 数量切分和 expert 内部 TP 切分的区别
3. 理解 fused MoE / standard dispatcher 路径
4. 在没有 DeepEP、或者做 kernel tuning / 单机 fused MoE 实验时可能用到
```

SGLang 仓库里的 fused MoE kernel benchmark 文档明确支持 TP 和 EP 组合，例如：

```text
tp_size = 8
ep_size = 4
=> moe_tp_size = 2
```

但在 DeepSeek / 大规模多机部署文档里，主线配置基本还是 DeepEP 或类似的 A2A EP 后端。也就是说：

```text
理解源码:
  本文很重要

真实大规模 MoE rollout:
  优先 DeepEP / DPA + DeepEP
```

## 0. 先给结论

在 SGLang 里，`moe_tp_size` 不是一个常规启动参数，而是内部推导出来的：

```text
moe_ep_size = ep_size
moe_dp_size = moe_dp_size
moe_tp_size = tp_size // moe_ep_size // moe_dp_size
```

对应源码：

```text
python/sglang/srt/distributed/parallel_state.py

moe_ep_size = expert_model_parallel_size
moe_dp_size = moe_data_model_parallel_size
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

所以只要满足：

```text
ep_size < tp_size
moe_dp_size = 1
tp_size % ep_size == 0
```

就会出现：

```text
moe_tp_size = tp_size / ep_size > 1
```

直观理解：

```text
tp_size = 16
ep_size = 8

16 张卡先按 EP 分成 8 个 expert owner
每个 expert owner 内部有 2 张卡
这 2 张卡对同一个 expert 做 MoE TP
```

也就是：

```text
EP 负责：
  expert 0/1 放在哪些 rank
  expert 2/3 放在哪些 rank
  ...

MoE TP 负责：
  一个 expert 的 gate/up/down 权重在 2 张卡上怎么切
```

## 1. 启动命令：tp16 + ep8 + moe_tp2

两节点、每节点 8 卡的例子：

node0:

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 8 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --dist-init-addr $NODE0_IP:20000 \
  --nnodes 2 \
  --node-rank 0 \
  --host 0.0.0.0 \
  --port 30000
```

node1:

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --ep-size 8 \
  --moe-a2a-backend none \
  --dp-size 1 \
  --pp-size 1 \
  --dist-init-addr $NODE0_IP:20000 \
  --nnodes 2 \
  --node-rank 1
```

这里外部业务请求仍然只发给 node0 暴露的 HTTP URL：

```text
http://$NODE0_IP:30000
```

内部 16 个 GPU rank 会通过 `dist-init-addr` 组成同一个分布式 world。`dist-init-addr` 可以理解成 torch distributed rendezvous 地址：所有节点都连接到同一个地址，SGLang 再根据 `nnodes`、`node_rank`、本机 GPU 数量推导全局 rank。

这个例子里内部并行配置是：

```text
global tp_size = 16
dp_size = 1
enable_dp_attention = false
ep_size = 8
moe_dp_size = 1
moe_tp_size = 2
```

注意不要把这个例子写成：

```bash
--moe-a2a-backend deepep
```

因为 DeepEP 会在参数处理阶段把 `ep_size` 改成 `tp_size`。如果你写：

```text
tp_size = 16
ep_size = 8
moe_a2a_backend = deepep
```

实际会变成：

```text
tp_size = 16
ep_size = 16
moe_tp_size = 1
```

这就不是本文要看的 Hybrid EP + MoE TP 了。

## 2. rank group 怎么组成

假设一个 TP world 有全局 rank：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

配置：

```text
tp_size = 16
ep_size = 8
moe_tp_size = 2
moe_dp_size = 1
```

### 2.1 MoE TP group

`parallel_state.py` 里 MoE TP group 是连续 rank：

```text
[0, 1]
[2, 3]
[4, 5]
[6, 7]
[8, 9]
[10, 11]
[12, 13]
[14, 15]
```

每一组大小都是 `moe_tp_size=2`。

它的含义是：

```text
[0, 1]:
  共同保存同一批 local experts
  但是每个 expert 的 intermediate 维度被切成 2 份

[2, 3]:
  共同保存下一批 local experts
  每个 expert 也被切成 2 份
```

### 2.2 MoE EP group

MoE EP group 不是连续 rank，而是隔着 `moe_tp_size` 取：

```text
moe_tp_rank = 0:
  [0, 2, 4, 6, 8, 10, 12, 14]

moe_tp_rank = 1:
  [1, 3, 5, 7, 9, 11, 13, 15]
```

这点非常关键。

因为每个 expert owner 内部有 2 张卡做 MoE TP，所以 EP 归并必须在相同 `moe_tp_rank` 的那一片上做：

```text
rank 0 和 rank 1:
  是同一个 expert owner 的两个 TP shard

rank 0,2,4,...,14:
  是不同 expert owner 的第 0 片 TP shard

rank 1,3,5,...,15:
  是不同 expert owner 的第 1 片 TP shard
```

画成图：

```text
                MoE TP group
              ┌──────────────┐
expert owner0 │ rank0  rank1 │  owns experts 0,1
              └──────────────┘
              ┌──────────────┐
expert owner1 │ rank2  rank3 │  owns experts 2,3
              └──────────────┘
              ┌──────────────┐
expert owner2 │ rank4  rank5 │  owns experts 4,5
              └──────────────┘
                    ...
              ┌──────────────┐
expert owner7 │ rank14 rank15│  owns experts 14,15
              └──────────────┘

MoE EP group for shard0:
  rank0 -> rank2 -> rank4 -> ... -> rank14

MoE EP group for shard1:
  rank1 -> rank3 -> rank5 -> ... -> rank15
```

这里假设模型有 16 个 routed experts，所以：

```text
num_experts = 16
ep_size = 8
num_local_routed_experts = 16 / 8 = 2
```

每个 expert owner 负责 2 个 experts，但这 2 个 experts 的权重又在 owner 内部的 2 张卡上继续切分。

## 3. 一个 layer 内部怎么走

先看一个简化 layer：

```text
input hidden
   |
   v
attention
   |
   |  attention TP16 通信
   |  例如 QKV/O projection 相关 all-reduce / all-gather
   v
post-attn hidden
   |
   v
router / gate
   |
   |  每个 rank 都可以算 topk
   |  topk 输出 global expert id
   v
MoE experts
   |
   |  EP:
   |    每个 rank pair 只拥有一部分 experts
   |
   |  MoE TP:
   |    一个 expert 内部 intermediate 维度再被 2 张卡切开
   v
post-experts reduce
   |
   |  1. MoE EP all-reduce
   |     group: [0,2,4,...,14] 或 [1,3,5,...,15]
   |
   |  2. MoE TP all-reduce
   |     group: [0,1]、[2,3]、...
   v
output hidden
```

对于 Qwen3 MoE，源码上 `Qwen3MoeSparseMoeBlock` 会读：

```text
self.tp_size = get_parallel().moe_tp_size
self.ep_size = get_parallel().moe_ep_size
```

所以这里的 `self.tp_size` 不是全局 `tp_size=16`，而是 MoE 内部的 `moe_tp_size=2`。

非 DeepEP / Ascend fuseep 时，会走：

```text
forward_normal()
```

逻辑是：

```text
1. gate 算 router_logits
2. topk 选 expert
3. experts(hidden_states, topk_output)
4. 如果 ep_size > 1:
     moe_expert_parallel_all_reduce()
5. 如果 moe_tp_size > 1:
     moe_tensor_model_parallel_all_reduce()
```

也就是说，在本文这个 `tp16 + ep8 + moe_tp2` 例子里，MoE block 后面会同时涉及两类归并：

```text
EP 归并:
  汇总不同 expert owner 的贡献

MoE TP 归并:
  汇总同一个 expert 内部不同 TP shard 的贡献
```

## 4. 为什么需要两类通信

假设 token A 的 top2 experts 是：

```text
expert 3
expert 10
```

在 `ep_size=8`、`num_experts=16` 时：

```text
expert owner0: experts 0,1  -> ranks [0,1]
expert owner1: experts 2,3  -> ranks [2,3]
expert owner2: experts 4,5  -> ranks [4,5]
...
expert owner5: experts 10,11 -> ranks [10,11]
```

那么 token A 的两个 expert 分别在：

```text
expert 3:
  owner1
  ranks [2,3]

expert 10:
  owner5
  ranks [10,11]
```

但是每个 expert 又被 MoE TP2 切开：

```text
expert 3:
  rank2 算 shard0
  rank3 算 shard1

expert 10:
  rank10 算 shard0
  rank11 算 shard1
```

所以要得到 token A 的完整 MoE 输出，需要两件事：

```text
1. 同一个 expert 的 shard0 + shard1 要合起来
   这就是 MoE TP 维度的归并

2. top2 experts 的贡献要合起来
   expert 3 和 expert 10 在不同 EP owner 上
   这就是 EP 维度的归并
```

按照 SGLang Qwen3 MoE 的 `forward_normal()` 写法，代码顺序是先做 EP all-reduce，再做 MoE TP all-reduce：

```text
if ep_size > 1:
  moe_expert_parallel_all_reduce()

if moe_tp_size > 1:
  moe_tensor_model_parallel_all_reduce()
```

从概念上看，你可以把它理解成：

```text
EP:
  解决 expert 分布在不同 owner 的问题

MoE TP:
  解决单个 expert 内部被切开的问题
```

## 5. 和 DeepEP 的区别

这篇文档最容易和第 4 篇 DeepEP 文档混淆，所以单独列一下：

| 配置 | ep_size | moe_tp_size | token 是否 all-to-all dispatch | expert 内部是否 TP 切 |
| --- | ---: | ---: | --- | --- |
| `tp16 ep16 deepep` | 16 | 1 | 是 | 否 |
| `tp16 ep8 none` | 8 | 2 | 否，基础 EP 路径 | 是 |

更直白地说：

```text
DeepEP:
  更关注 token 怎么高效 dispatch/combine 到不同 expert rank
  通常 ep_size 会被设成 tp_size
  每个 expert rank 拥有自己的 local experts
  moe_tp_size 通常是 1

Hybrid EP + MoE TP:
  更关注 expert owner 不够多时，一个 owner 内部继续切 expert 权重
  ep_size 小于 tp_size
  moe_tp_size 大于 1
```

所以如果你的目标是学习“expert 内部怎么被切”，就要看本文这种 `ep_size < tp_size` 的路径；如果目标是学习高性能 token dispatch，就看 DeepEP。

## 6. 再加 DPA：dp8 + attention tp2 + ep8 + moe_tp2

如果同时启用 DPA，可以得到另一个很有用的组合：

```bash
python -m sglang.launch_server \
  --model-path $MOE_MODEL \
  --trust-remote-code \
  --tp-size 16 \
  --dp-size 8 \
  --enable-dp-attention \
  --ep-size 8 \
  --moe-a2a-backend none \
  --pp-size 1 \
  --host 0.0.0.0 \
  --port 30000
```

内部可以理解成：

```text
global tp_size = 16
dp_size = 8
enable_dp_attention = true
attn_tp_size = 16 / 8 = 2

ep_size = 8
moe_dp_size = 1
moe_tp_size = 16 / 8 / 1 = 2
```

这个组合下有两层“2”：

```text
attention TP2:
  每一路 DP attention 内部用 2 张卡做 attention/dense TP

MoE TP2:
  每个 expert owner 内部用 2 张卡切 expert intermediate 维度
```

在这个例子里，attention TP group 和 MoE TP group 形状刚好都像：

```text
[0,1], [2,3], [4,5], ...
```

但它们的语义不同：

```text
attention TP group:
  服务某一路 DP 请求流的 attention/dense 计算

MoE TP group:
  服务某个 expert owner 内部的 expert 权重切分
```

对于 RL rollout，外部 HTTP URL 仍然通常只有一个：

```text
http://$SERVER_IP:30000
```

但这时请求可以被路由到不同 DP rank：

```text
routed_dp_rank: 0..7
```

所以它和第 1 个例子最大的差别是：

```text
tp16 ep8 dp1:
  attention 阶段还是同一路请求流
  只是 MoE expert 内部出现 EP8 + MoE TP2

tp16 ep8 dp8 enable_dp_attention:
  attention 阶段已经有 8 路 DP 请求流
  每路 attention 用 TP2
  MoE 阶段仍然是 EP8 + MoE TP2
```

## 7. 和第 6 篇业务配置的关系

第 6 篇业务配置是：

```text
tp_size = 64
dp_size = 8
enable_dp_attention = true
ep_size = 64
moe_a2a_backend = deepep
```

因为 DeepEP 会让：

```text
ep_size = tp_size = 64
```

所以：

```text
moe_tp_size = 64 / 64 / 1 = 1
```

也就是说，第 6 篇业务配置里不是本文这种 Hybrid EP + MoE TP。它是：

```text
DPA8 + DeepEP64

attention:
  dp_size = 8
  attention tp = 8

MoE:
  ep_size = 64
  moe_tp_size = 1
  token 通过 DeepEP dispatch 到 64 个 expert ranks
```

本文的配置则是：

```text
DPA 可开可不开
moe_a2a_backend = none
ep_size < tp_size
moe_tp_size > 1
```

它更适合用来理解 SGLang 对 MoE 专家权重的二维切分方式。

## 8. 约束和容易踩的点

### 8.1 `tp_size` 必须能被 `ep_size` 整除

实际配置应该满足：

```text
tp_size % ep_size == 0
```

因为内部会用：

```text
moe_tp_size = tp_size // ep_size // moe_dp_size
```

来构造 MoE TP / EP group。量化 MoE 兼容性检查里也有明确报错：

```text
if self.tp_size % self.moe_ep_size != 0:
  raise ValueError(...)
```

否则无法得到合理的整数 `moe_tp_size`。

### 8.2 `moe_intermediate_size` 必须能被 `moe_tp_size` 整除

expert 内部是按 intermediate 维度切的，所以：

```text
moe_intermediate_size % moe_tp_size == 0
```

`FusedMoE` 初始化时会 assert：

```text
assert intermediate_size % self.moe_tp_size == 0
```

如果是量化 MoE，还可能有 block size 对齐要求：

```text
(moe_intermediate_size / moe_tp_size) % weight_block_size_n == 0
```

### 8.3 routed expert 数量必须能被 `ep_size` 整除

`FusedMoE` 初始化时会检查：

```text
(num_experts - num_shared_slots) % moe_ep_size == 0
```

然后计算：

```text
num_local_routed = num_global_routed // moe_ep_size
```

所以如果模型 routed experts 数量不能被 `ep_size` 整除，就不能这样均匀切 expert owner。

### 8.4 `moe_tp_size` 不等于 attention TP size

这是本文最重要的概念之一。

不启用 DPA 时：

```text
tp_size = 16
dp_size = 1
enable_dp_attention = false

attention TP size = 16
moe_tp_size = 2
```

启用 DPA 时：

```text
tp_size = 16
dp_size = 8
enable_dp_attention = true

attention TP size = 2
moe_tp_size = 2
```

第二个例子里二者数值一样，但含义仍然不同。

### 8.5 不要和 DeepEP 混用这个心智模型

如果设置：

```text
--moe-a2a-backend deepep
```

那么 DeepEP 会把：

```text
ep_size = tp_size
```

此时你即使手动写了：

```text
--tp-size 16
--ep-size 8
```

也不能按 `moe_tp_size=2` 去理解，因为参数处理后会变成 `ep_size=16`。

## 9. 应该什么时候考虑 Hybrid EP + MoE TP

可以用下面的思路判断：

```text
想减少每张卡保存的 expert 数量:
  增大 ep_size

单个 expert 太大，想把 expert 内部 FFN 也切开:
  增大 moe_tp_size
  也就是让 ep_size 小于 tp_size

想让不同请求从 attention 阶段就走不同数据流:
  开 enable_dp_attention
  设置 dp_size > 1

想要高性能 token dispatch/combine:
  优先看 DeepEP
  但 DeepEP 通常会让 ep_size = tp_size，因此 moe_tp_size = 1
```

所以本文这个组合更像是学习和理解 MoE 权重二维切分的关键形态：

```text
expert 维度:
  EP 切

expert 内部 intermediate 维度:
  MoE TP 切
```

## 10. 源码阅读路径

建议按这个顺序看：

```text
1. 参数处理
   python/sglang/srt/server_args.py
   _handle_a2a_moe()

   重点看 deepep/mooncake/nixl/flashinfer/mori/megamoe
   为什么会把 ep_size 改成 tp_size

2. 并行组初始化
   python/sglang/srt/distributed/parallel_state.py
   initialize_model_parallel()

   重点看：
   moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
   _MOE_EP group 怎么按 moe_tp_size 间隔取 rank
   _MOE_TP group 怎么按连续 rank 组成

3. Qwen3 MoE block
   python/sglang/srt/models/qwen3_moe.py
   Qwen3MoeSparseMoeBlock

   重点看：
   self.tp_size = get_parallel().moe_tp_size
   self.ep_size = get_parallel().moe_ep_size
   forward_normal()
   moe_expert_parallel_all_reduce()
   moe_tensor_model_parallel_all_reduce()

4. FusedMoE 权重切分
   python/sglang/srt/layers/moe/fused_moe_triton/layer.py

   重点看：
   self.moe_ep_size
   self.moe_tp_size
   num_local_routed
   intermediate_size_per_partition

5. Standard dispatcher
   python/sglang/srt/layers/moe/token_dispatcher/standard.py

   重点看：
   local_expert_mapping
   global expert id 怎么映射成本 rank 的 local expert id
```

## 11. 一句话总结

`Hybrid EP + MoE TP` 的核心是：

```text
tp_size > ep_size
=> 一个 expert owner 不再是一张卡，而是一组 MoE TP ranks
=> EP 切 expert 数量，MoE TP 切 expert 内部 intermediate 维度
```

对于 `tp16 + ep8`：

```text
8 个 expert owners
每个 owner 2 张卡
每个 owner 负责一部分 experts
每个 expert 又在 owner 内部做 TP2
```

它不是 DeepEP 的高性能 A2A 路径，而是理解 SGLang MoE 并行组和专家权重切分方式的一个很好的中间形态。

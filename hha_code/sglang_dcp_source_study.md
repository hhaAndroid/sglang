# SGLang Decode Context Parallel 源码学习笔记

> 研究对象：当前工作区 SGLang 源码，commit `6f005e4da1`（2026-08-14）  
> 本文中的“DCP”默认指 **Decode Context Parallelism**；Prefill Context Parallel 简称 Prefill CP。  
> 目标：先用一个 `GQA + TP8/DCP4 + 16-token` 请求讲清权重布局、Prefill→连续 Decode 和 KV 增长；建立直觉后再讲 MLA 的 latent/weight absorption，并沿源码展开通信后端、PD 与限制。

## 先读这一章：一条请求从 Prefill 到连续 Decode 的完整 DCP 流程

第一次阅读先只看这一章。后面的拓扑、allocator、backend 和 PD 都是在解释这里的某一步。

### A. 固定例子

假设：

```text
初始 prompt：16 tokens，position 0..15
Attention 类型：GQA，Hq=64，Hkv=2
DCP 所在 TP group：TP=8
DCP size：4
物理 GPU 数：8（不是 TP8 × DCP4 = 32）
一个请求，先 Prefill，再连续 Decode
```

这里必须先说明当前 SGLang 的参数语义：**DCP 与 head/weight TP 在算法上是两个不同的分片方向，但 `--tp-size` 和 `--dcp-size` 不是两个相乘的独立 world size。DCP ranks 从 TP group 内部取得。**硬约束是：

```text
tp_size % dcp_size == 0
```

可以把一个 TP group 在 DCP attention 内部理解成二维网格：

```text
有效 query-head groups = tp_size / dcp_size
每个 head group 内的 context shards = dcp_size
物理 ranks = (tp_size / dcp_size) × dcp_size = tp_size
```

因此，如果想表达“8 张卡，query-head 方向分成 2 组，每组内部 DCP4”，当前 SGLang 参数应当是：

```text
--tp-size 8 --dcp-size 4

二维视图：effective head-TP=2，DCP=4

head group 0 / DCP group 0: ranks [0,1,2,3]
head group 1 / DCP group 1: ranks [4,5,6,7]
```

而下面这种配置不是“8 卡 TP2×DCP4”，在当前实现里会直接非法，因为 DCP group 是 TP group 的子组，`4` 不能整除进 `2`：

```text
--tp-size 2 --dcp-size 4   # 非法
```

后面的 Prefill、KV 增长和 Decode 流程都先使用这个 GQA 例子。MLA 的 latent cache、weight absorption 和 replicated-Q 都放到第 6 节，首次阅读先不要引入。

#### A.1 先看 GQA 权重怎样组成二维布局

对 `TP8/DCP4,Hq=64,Hkv=2`：

```text
effective head groups = TP / DCP = 2

DCP group [r0,r1,r2,r3]
  r0/r1/r2/r3：分别持有 8 个不同 Q heads 的权重
  r0/r1/r2/r3：都持有同一个 KV head 0 的 K/V 权重

DCP group [r4,r5,r6,r7]
  r4/r5/r6/r7：分别持有另外 8 个不同 Q heads 的权重
  r4/r5/r6/r7：都持有同一个 KV head 1 的 K/V 权重
```

也就是：

```text
Q projection：按 TP8 切
K/V projection：按 effective KV-TP2 切，再在每个 DCP4 group 内复制
```

运行时每个 group 内四份不同的 Q heads 会 AllGather：

```text
每 rank 8 Q heads -> group 内每 rank 临时拥有 32 Q heads
```

但相同的 K/V 权重不需要 gather。四个 context ranks 都能生成 KV head 0（或 head 1），随后只把属于自己的 token positions 写入 persistent cache。

#### A.2 再看同一组内 KV 怎样按 token position 分片

DCP 的核心 owner 规则：

```text
owner(position) = position % 4
local_index     = position // 4
```

因此只看一个 DCP group 内的 KV payload，目标状态是：

```text
dcp_rank0 owns positions: 0,4,8,12,...
dcp_rank1 owns positions: 1,5,9,13,...
dcp_rank2 owns positions: 2,6,10,14,...
dcp_rank3 owns positions: 3,7,11,15,...
```

映射回 8 张物理卡：

```text
KV head 0 / group [r0,r1,r2,r3]：r0/r1/r2/r3 分别是 owner 0/1/2/3
KV head 1 / group [r4,r5,r6,r7]：r4/r5/r6/r7 分别是 owner 0/1/2/3
```



### B. 阶段一：初始 16-token Prefill



#### B.1 本轮先计算什么

初始 prompt 没有历史 prefix。模型对 position `0..15` 做一次 `EXTEND`/Prefill forward，逐层产生当前 16 个 token 的 Q/K/V。

同一 DCP group 的四个 ranks 拥有相同的 K/V-head weight shard，因此都可以计算本轮全部新 token 对应的 K/V，用它完成正常 causal Prefill attention：

```text
Q(position i) attends K/V(position 0..i)
```

这里的“本轮完整”只指当前 DCP group 所负责的 **KV-head shard 的完整 token range**。例如 r0～r3 都能临时计算 `KV head 0 × positions 0..15`，r4～r7 都能临时计算 `KV head 1 × positions 0..15`；不代表任一 rank 拥有两个 KV heads，也不代表它会把 16 个 positions 都长期存进 pool。

#### B.2 写入 persistent KV pool 时立即分片

本轮新 KV 写入长期 pool 时执行 owner filter。`dcp_rank0` 不是唯一一张全局 GPU，而是每个 DCP group 中各有一个；因此必须把 KV-head 分片和 token-position 分片同时画出来：

```text
物理 rank   effective KV-head shard   persistent token positions
r0          KV head 0                 0,4,8,12
r1          KV head 0                 1,5,9,13
r2          KV head 0                 2,6,10,14
r3          KV head 0                 3,7,11,15

r4          KV head 1                 0,4,8,12
r5          KV head 1                 1,5,9,13
r6          KV head 1                 2,6,10,14
r7          KV head 1                 3,7,11,15
```

所以你的理解是对的：在这个 `Hkv=2, effective KV-TP=2` 的例子里，每个物理 rank 的 persistent KV 在 head 维只保留 `1/2`，在 token-position 维只保留 `1/4`：

```text
完整逻辑 K 或 V： [16 tokens, 2 KV heads, D]
每 rank persistent： [4 local tokens, 1 local KV head, D]
```

对某一个 position，完整的两个 KV heads 也分散在两个 DCP groups：

```text
position 0：KV head 0 在 r0，KV head 1 在 r4
position 1：KV head 0 在 r1，KV head 1 在 r5
position 2：KV head 0 在 r2，KV head 1 在 r6
position 3：KV head 0 在 r3，KV head 1 在 r7
```

owner-local index 在每个 DCP group 内按相同规则计算：

```text
global position       owner rank       owner-local index
0                     0                0
1                     1                0
2                     2                0
3                     3                0
4                     0                1
5                     1                1
...                   ...              ...
15                    3                3
```

Prefill 结束时要同时记住两种状态：

```text
逻辑请求状态：
  global sequence length = 16
  所有 rank 都知道完整 request/virtual-token mapping

物理 KV payload：
  每个 DCP group 的 rank-local lengths = [4,4,4,4]
  每个物理 rank 只保存 1 个 KV head × 4 个 local token rows
```

这一步已经完成 KV 分片。后面 Decode 不会先加载 full KV 再切。

#### B.3 Prefill 输出第一个待生成 token

Prefill 最后一个 hidden state 经过 lm_head/sampling，得到第一个生成 token，记为 `x16`。此时：

```text
x16 已经被采样出来
但 x16 自己的 KV 还没有经过 Transformer forward 写入 cache
cache 中仍是 position 0..15
```



### C. 阶段二：第一次 Decode，处理 position 16

第一次 Decode forward 的输入是刚采样出的 `x16`，它位于 position 16。

#### C.1 先为新 token 计算 Q16/K16/V16

模型计算：

```text
Q16, K16, V16
```

position 16 在每个 DCP group 内的 KV owner 都是：

```text
16 % 4 = 0
```

所以两个 DCP groups 中的 `dcp_rank0` 都会写入各自负责的 KV-head shard：

```text
group [r0,r1,r2,r3]：r0 保存 KV head 0 的 K16/V16
group [r4,r5,r6,r7]：r4 保存 KV head 1 的 K16/V16

r1/r2/r3/r5/r6/r7：本请求本轮不持久化 K16/V16
owner-local index = 16 // 4 = 4
```

逻辑上，本轮 attention 包含当前 token 自己，所以处理 position 16 时的全局可见 KV 是 position `0..16`。不同 backend 可能在 kernel 调用前后组织 write/metadata，但最终语义一致。

只画任意一个 DCP group，此时 local KV lengths 变成：

```text
dcp_rank0: positions 0,4,8,12,16 -> length 5
dcp_rank1: positions 1,5,9,13    -> length 4
dcp_rank2: positions 2,6,10,14   -> length 4
dcp_rank3: positions 3,7,11,15   -> length 4

local lengths = [5,4,4,4]
global length = 17
```



#### C.2 Decode planner 直接构造 local KV 读取表

每个 rank 都知道 global sequence length 和完整 virtual mapping，但 planner 只给 attention kernel 当前 rank 的 local metadata：

```text
dcp_rank0 local page table -> KV positions 0,4,8,12,16
dcp_rank1 local page table -> KV positions 1,5,9,13
dcp_rank2 local page table -> KV positions 2,6,10,14
dcp_rank3 local page table -> KV positions 3,7,11,15
```

这里没有：

```text
load full KV -> 临时切片 -> local attention
```

实际是：

```text
global logical metadata
        -> owner filter / local index planning
        -> attention kernel 直接读取本 GPU persistent local KV
```



#### C.3 每个 rank 都对同一个 Q16 计算 partial attention

先区分 Q 的两个维度：

```text
Q shape（概念上）= [query-token 数, query-head 数, head_dim]
```

普通 Decode 对**每个请求**确实只有 1 个 query token，但这不代表当前 rank 已经拥有完整的 query heads。在当前 `TP8/DCP4,Hq=64,Hkv=2` 例子中，Q projection 按 TP8 分片，所以每 rank 先产生 8 个 Q heads；每个 DCP group 合起来负责 32 个 Q heads：

```text
每个 rank 投影得到：Q_local  [1,  8, D]
                                  ^  ^
                                  |  └─ 当前 rank 的 TP-local heads
                                  └──── 普通 Decode 的 1 个 query token

沿 DCP-group 的 head 维 AllGather 后：Q_group [1, 32, D]
```

因此这里的 Q AllGather **不是把 query length 从 1 补成 4**，而是保持 query-token 数为 1、把 `H_local` 补成 `H_group`。当前 Triton GQA DCP 路径直接对 `[B,H,D]` 的 head 维 `dim=1` 做 AllGather：[triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L1853)。

这也不表示 TP 被取消。Q/O projection 和其他模型计算仍是 TP8；只是在 attention core 入口，为了让四个 context shards 都计算同一批 heads，临时把四份 TP-local Q heads 收齐。当前 Triton GQA 路径在 DCP attention 和 LSE merge 后对修正后的 output 做 all-reduce，再按 head slice，使每 rank 恢复 `[1,8,D]` 的 TP-local output：

```text
每个 DCP group：
TP-local Q projection:    4 × [1,  8,D]
           head AllGather:4 × [1, 32,D]   # context attention 临时复制
           local KV attention + LSE merge
           output AllReduce + local-head slice
TP-local attention out:   4 × [1,  8,D]
```

DCP 之所以需要这一步，是因为每个 context rank 都要用同一组完整的 DCP-group query heads，分别查询自己保存的 KV positions。当前 GQA 例子先只记住默认路径：`TP-local Q -> head AllGather -> group Q`。MLA 特有的 replicated-Q 优化放到第 10 节。

然后 group `[r0,r1,r2,r3]` 使用 Q heads `0..31` 和 KV head 0，分别计算：

```text
r0: Q16[heads 0..31] attends KV-head0(positions 0,4,8,12,16) -> o0, lse0
r1: Q16[heads 0..31] attends KV-head0(positions 1,5,9,13)    -> o1, lse1
r2: Q16[heads 0..31] attends KV-head0(positions 2,6,10,14)   -> o2, lse2
r3: Q16[heads 0..31] attends KV-head0(positions 3,7,11,15)   -> o3, lse3
```

group `[r4,r5,r6,r7]` 同时对 Q heads `32..63`、KV head 1 做完全相同的四份 context-shard attention。

这里每个 `o_r` 都只在自己的 context shard 内做了 softmax。不能直接相加或平均。

#### C.4 用 LSE 合并成完整 attention output

```text
global_lse = logsumexp(lse0,lse1,lse2,lse3)

O16 = exp(lse0-global_lse) * o0
    + exp(lse1-global_lse) * o1
    + exp(lse2-global_lse) * o2
    + exp(lse3-global_lse) * o3
```

合并后的 `O16` 与在完整 KV `0..16` 上直接计算 attention 数学等价。

后续 output projection、MLP/MoE、lm_head 继续执行，最后采样出下一个 token `x17`。

### D. 阶段三：第二次及后续 Decode，KV 怎样持续增长

第二次 Decode 输入 `x17`，position 17：

```text
owner(17) = 17 % 4 = 1
local_index(17) = 17 // 4 = 4
```

所以每个 DCP group 的 `dcp_rank1` 新增一个 KV row，也就是物理 rank `r1` 和 `r5` 分别写入自己负责的 KV-head shard：

```text
处理 position 17 后：local lengths = [5,5,4,4]
```

接下来：


| Decode forward 处理的位置  | 新 KV owner | 写入后的 global length | 各 rank local KV lengths |
| --------------------- | ---------- | ------------------ | ----------------------- |
| Prefill 完成，只含 `0..15` | —          | 16                 | `[4,4,4,4]`             |
| 16                    | dcp_rank0  | 17                 | `[5,4,4,4]`             |
| 17                    | dcp_rank1  | 18                 | `[5,5,4,4]`             |
| 18                    | dcp_rank2  | 19                 | `[5,5,5,4]`             |
| 19                    | dcp_rank3  | 20                 | `[5,5,5,5]`             |
| 20                    | dcp_rank0  | 21                 | `[6,5,5,5]`             |
| 21                    | dcp_rank1  | 22                 | `[6,6,5,5]`             |
| 22                    | dcp_rank2  | 23                 | `[6,6,6,5]`             |
| 23                    | dcp_rank3  | 24                 | `[6,6,6,6]`             |


这张表就是 DCP KV 增长的核心：

1. 每个新 token 在每个 DCP group 内只增加一个 owner rank 的 persistent KV；
2. owner 按 `position % dcp_size` 循环；
3. 不需要迁移旧 KV，也不需要周期性重新平衡；
4. ranks 间 local length 始终最多相差 1；
5. 每一步每个 DCP group 的所有 ranks 仍共同计算同一组 query heads 的 partial attention，并做 LSE merge。



### E. 把一个 Decode step 压缩成六步

后续每个生成 token 都重复同一流程：

```text
1. 输入上一步采样出的 token x_p，位置为 p

2. 计算 Q_p/K_p/V_p

3. owner = p % DCP
   每个 DCP group 内只有 owner rank 把本组 K_p/V_p 写入 persistent local pool

4. 每个 rank 构造自己的 local kv_lens/page table
   并直接读取 local persistent KV

5. 每个 rank 用同一个 DCP-group Q_p 计算 partial output o_r + lse_r

6. 跨 rank 精确合并 o_r/lse_r
   -> 完整 O_p -> 后续网络 -> 采样 x_(p+1)
```

最短记忆版：

```text
Prefill：full current-token compute，owner-only persistent KV write
Decode： one new owner write + all ranks local attention + LSE merge
Growth： owner round-robin，local lengths 最多相差 1
```



### F. 两个容易混淆的例外



#### F.1 Prefix Extend 临时 gather

后续 `EXTEND` 如果命中已经 DCP-sharded 的历史 prefix，某些 MLA backend 会临时 AllGather/reorder prefix KV，供期望 full prefix 的 extend kernel 使用。

这不会改变 persistent pool：

```text
临时计算 buffer：可能是 full prefix
persistent KV pool：始终是 local shard
```



#### F.2 PD transfer

如果 KV 来自 P worker：

- P DCP1→D DCPN：P sender 选择目标 rank 拥有的 token rows，直接写 D local pool；
- P/D 相同 DCPN：matching DCP ranks 的 local page 直接传；
- D 收到完成通知时，pool 已经是上述 local layout，之后从 `[4,4,4,4]` 等状态继续增长；
- 当前不支持 `P: Prefill CP>1 -> D: DCP>1`。



### G. 与 Prefill CP 只保留一个最关键区别

```text
Prefill CP：不同 ranks 负责不同 query tokens
            local Q + full KV
            attention 后不合并同一个 query 的 partial softmax

DCP：       不同 ranks 负责同一个 query 的不同 KV context shards
            full/group Q + local KV
            attention 后必须用 LSE 合并同一个 query
```

如果上面完整流程还没建立起来，先不要继续看后面的拓扑和 backend 章节。

建议阅读顺序改为：

```text
第一次：只读本章，建立 Prefill -> Decode -> KV round-robin growth
第二次：跳到第 7 章，把本章 GQA 六步映射到 Triton 源码
第三次：需要理解 MLA 时再看第 6 章；按需要看第 9 章通信后端、第 13 章 PD
第 1~5 章的动机、模型、拓扑、allocator 作为查阅资料，不要求先读
```



## 0. 补充概念对照：DCP 与 Prefill CP（首次阅读可跳过）

虽然两者都叫 Context Parallel，但它们并不是“同一种 CP 分别用在 prefill/decode”。二者切分的对象和 attention 并行方式正好相反：

```text
Prefill CP：切当前 query/token，收集完整 K/V
            = local Q + full KV

DCP：       切长期历史 K/V，让各 rank 共同计算同一批 query
            = full Q + local KV
```

对应的详细 Prefill CP 实现可看 [sglang_prefill_cp_source_study.md](./sglang_prefill_cp_source_study.md)。阅读本文前，先记住下面这个对照。

### 0.1 用同一个 attention 例子比较

假设一条请求已有 8 个 context tokens，使用两个 ranks。

#### Prefill CP：不同 rank 负责不同 query tokens

简化成连续 token shard：

```text
rank0:
  local Q = query tokens 0,1,2,3

rank1:
  local Q = query tokens 4,5,6,7
```

每个 rank 首先只根据自己的 local hidden 产生 local Q/K/V。为了让 local Q 看到完整 causal context，需要把各 rank 新产生的 K/V 收集起来：

```text
rank0 local K/V --┐
                  ├─ K/V AllGather + token reorder
rank1 local K/V --┘
                  │
                  ▼
          两个 ranks 都得到 full K/V
```

然后：

```text
rank0: Q(tokens 0..3) attends full K/V(tokens 0..7)
rank1: Q(tokens 4..7) attends full K/V(tokens 0..7)
```

同一个 query 只在一个 CP rank 上计算；两个 ranks 负责的是不同 query rows。所以 attention 后不需要跨 CP rank 合并同一 query 的 partial softmax。最后只需在模型末端收集不同 ranks 的 token outputs 并恢复全局顺序。

普通 Prefill CP 的长期 KV cache 结果是：

```text
rank0: full token range K/V
rank1: full token range K/V
```

因此它主要分摊 prefill query/hidden/activation 和计算，不会自然把 KV capacity 放大 2 倍。

#### DCP：不同 rank 负责同一 query 的不同历史 context

DCP 不把当前 decode query 分给不同 ranks。它把已经长期驻留的 KV context 按 token position 分片：

```text
rank0 local KV: token positions 0,2,4,6
rank1 local KV: token positions 1,3,5,7
```

假设当前需要为新 token 8 计算 attention。两个 ranks 都必须处理这个 query：

```text
                     Q(token 8)
                    /          \
                   /            \
rank0: Q8 attends KV(0,2,4,6)  -> partial output o0 + lse0
rank1: Q8 attends KV(1,3,5,7)  -> partial output o1 + lse1
                   \            /
                    \          /
                     LSE merge
                         │
                         ▼
                  global output O8
```

这里同一个 query 的 softmax 被 context 维切成两块。每块有不同的局部 softmax 分母，所以 `o0/o1` 不能直接相加或平均，必须携带 `lse0/lse1` 做全局归一化：

```text
global_lse = logsumexp(lse0, lse1)

global_o = exp(lse0 - global_lse) * o0
         + exp(lse1 - global_lse) * o1
```

DCP 的长期 KV cache 结果是：

```text
rank0: 1/2 token range K/V
rank1: 1/2 token range K/V
```

所以它能提高 KV capacity，并减少每 rank 每个 decode step 读取的历史 KV。

### 0.2 两条数据流为什么正好相反


| 对比项                     | Prefill CP                                       | DCP                                                                 |
| ----------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| 主要阶段                    | Prefill/extend                                   | Decode，以及 DCP-aware target verify                                   |
| 切分对象                    | 当前 forward 的 query/hidden tokens                 | 长期 KV cache 的历史 token positions                                     |
| 每个 query 由几个 CP rank 计算 | 1 个                                              | 所有 DCP ranks 各算一个 context shard                                     |
| Attention 输入形态          | local Q + full KV                                | full/group Q + local KV                                             |
| Attention 前主要通信         | K/V AllGather、token reorder                      | Q **head 维** AllGather（不是扩展 query-token length），或 replicated-Q 本地重算 |
| Attention 后主要通信         | 通常没有同一 query 的 CP output merge                   | partial output + LSE 精确合并                                           |
| KV cache token 范围       | 普通路径每 CP rank 最终是 full token range               | 每 DCP rank 只保存 owner token range                                    |
| 核心收益                    | TTFT、prefill compute、activation                  | KV capacity、长 context decode KV read                                |
| 典型代价                    | 随新增/prefix token 规模变化的 K/V gather                | 每层相对固定的 Q/output/LSE collective                                     |
| 当前 PD 组合                | 可以用于普通 decode worker，但不能直接作为 DCP decode 的 source | 要求 P 端 `attn_cp_size=1`；支持 DCP1→DCPN 或相同 DCPN→DCPN                  |


通信方向相反不是偶然，而是由 attention 公式决定的：

```text
Attention(Q, K, V)

按 Q rows 切：
  每个 Q row 可独立计算，但必须看到完整 K/V
  -> Prefill CP

按 K/V context columns 切：
  同一个 Q 的 softmax 分母跨 shards
  -> DCP，必须合并 partial output + LSE
```



### 0.3 生命周期也不同

Prefill CP 的 token sharding 主要是一次 forward 内的计算布局：

```text
全局 prompt tokens
  -> shard current hidden/query
  -> gather 本层新 K/V
  -> local query attention
  -> 最终 gather hidden/token outputs
```

DCP 的 KV sharding 是跨多个 decode steps 长期存在的 cache 布局：

```text
prefill/extend 产生 KV
  -> 根据 owner(position) 写入不同 rank
  -> 后续每个 decode step 都读取同一套分片 cache
  -> 每一步都合并各 rank partial attention
```

因此 DCP 不只是一个 attention kernel 开关。Allocator、page size、Radix Cache、KV write、prefix hit、CUDA graph、Spec 和 PD transfer 都必须理解相同的 virtual-to-physical DCP 下标。

### 0.4 一句话记忆

```text
Prefill CP：Q 分片，KV 补全；一条 query 只在一个 rank 上算。
DCP：       KV 分片，Q 补全；一条 query 在所有 ranks 上算 partial attention，再按 LSE 合并。
```

后文只要遇到容易混淆的数据流，都可以回到这两句。

### 0.5 DCP 本身的结论

SGLang 的 DCP 是一种面向长上下文 decode 的 **KV token-position 并行**：一个 DCP group 内，不同 rank 保存同一请求的不同 token 位置，每张卡只读取和计算约 `1 / dcp_size` 的历史 KV；各 rank 得到局部 attention output 和 LSE 后，再进行一次数学上精确的跨 rank softmax 合并。

它最适合：

- MLA 模型在 TP 下 latent KV 被各 attention rank 复制，例如 DeepSeek V3.1、Kimi 系列；
- 单请求 context 很长，KV cache 容量或每步读取 KV 的显存带宽成为瓶颈；
- 已经使用多卡 TP，希望复用同一 TP group 内的 ranks 分片上下文；
- rank 间有 NVLink/NVSwitch/MNNVL 等高速互联；
- decode 比 prefill 更关键，或者 PD decode worker 需要长期保存超长上下文。

它通常不适合：

- 短上下文、单请求最低延迟优先：每层固定 collective 可能大于节省的 KV 读取；
- 主要瓶颈是 prefill/TTFT：DCP 不是 Prefill CP，extend 还可能 gather 历史 prefix；
- 上下文并不占主要显存，瓶颈来自权重、MoE buffer、Mamba/KDA state 或 request slot；
- DCP group 跨低带宽网络；
- 模型、attention backend、KV dtype 或 cache/PD/spec 组合没有 DCP 接入。

最重要的实现事实是：

> DCP 不是把 query token 分给不同 rank 后各自产生不同输出，而是让所有 rank 针对同一批 query，分别计算不同 KV context shard 的 partial softmax；因此最后必须用各 shard 的 LSE 对 partial output 做加权合并。

以 `DCP=4`、全局 token 位置 `0..7` 为例：

```text
rank0 owns: 0, 4
rank1 owns: 1, 5
rank2 owns: 2, 6
rank3 owns: 3, 7

owner(v)    = v % 4
physical(v) = v // 4
```

每个 rank 的 attention kernel 得到：

```text
o_r   = softmax(local_logits_r) @ local_V_r
lse_r = logsumexp(local_logits_r)
```

全局结果不是平均值，而是：

```text
global_lse = logsumexp_r(lse_r)
global_o   = sum_r(exp(lse_r - global_lse) * o_r)
```

该结果与在完整 KV 上直接计算 attention 等价，只存在浮点归约顺序差异。

对 absorbed MLA，DCP 的价值尤其大：TP 可以切 query heads 和权重，但共享的 latent KV 往往仍在 attention TP ranks 上复制。DCP 把这份复制的 context 改成按位置分片，使每 rank 的目标 KV 容量和 decode KV 读取量近似降为 `1 / dcp_size`。

---



## 1. DCP 到底解决什么问题



### 1.1 Decode 的计算与存储形态

设已有 context 长度为 `L`，decode 每步新增一个 token。对每个 attention 层，每个请求都需要：

1. 计算新 token 的 Q/K/V 或 MLA latent；
2. 把新 KV 写入 cache；
3. 用新 Q 读取长度约为 `L` 的历史 KV；
4. 计算 attention output；
5. context 长度增长为 `L + 1`。

单步 query 数很少，attention 更接近 memory-bound：大量时间花在从 KV cache 读取历史 context，而不是大规模 Q GEMM。随着 `L` 增长：

```text
每步 KV 读取量      ~ O(L)
KV 常驻显存          ~ O(L)
每步新增 KV          ~ O(1)
```

DCP 直接切分前两个 `O(L)` 项。

### 1.2 为什么普通 TP 对 MLA KV 帮助有限

普通 MHA/GQA 可以沿 KV heads 做一定程度的 TP sharding。但 absorbed MLA 只有一份被所有 query heads 共享的 latent KV 表示。TP 切 heads 后，各 rank 仍需要读同一份 latent context，因此通常保存完整 MLA KV。

```text
TP-only MLA:
  rank0: local Q heads + full latent KV context
  rank1: local Q heads + full latent KV context
  ...

DCP MLA:
  rank0: DCP-group Q heads + token-position shard 0
  rank1: DCP-group Q heads + token-position shard 1
  ...
```

官方源码文档也把 absorbed MLA decode 和 static target verification 定义为 DCP 主路径：[dcp.mdx](../docs/docs/advanced_features/dcp.mdx#L8)。

### 1.3 收益来自哪里

设 DCP degree 为 `C`，每卡物理 KV token capacity 为 `M`。

理想情况下：

```text
每 rank 的单请求 KV 长度     L       -> L/C
每 rank 的 context KV 读取    R(L)    -> R(L/C)
共享虚拟 token capacity       M       -> M*C
DCP group 内目标 KV 总副本数  C 份     -> 1 份
```

但以下内容不会自动缩小 `C` 倍：

- 模型权重；
- 非 DCP-sharded 的 state/cache；
- speculative draft KV（当前通常复制）；
- Kimi K3 的 request-indexed KDA/Mamba state；
- 通信 workspace、MoE dispatch buffer、CUDA graph pool；
- request slot 数量。

所以 `max_total_num_tokens` 可能接近扩大 `C` 倍，不代表 `max_running_requests` 一定扩大 `C` 倍。

### 1.4 一个简单的性能 break-even

可以粗略写成：

```text
T_no_dcp(L) ≈ T_attention_read_and_compute(L)

T_dcp(L,C) ≈ T_attention_read_and_compute(L/C)
           + T_query_distribution(B,H,D,C)
           + T_partial_output_merge(B,H,V,C)
```

其中通信主要随 batch、head 数、head dim 和 DCP size 变化，不直接随 `L` 线性增长；节省的 KV 读取随 `L` 增长。因此：

```text
(1 - 1/C) * T_attention(L) > T_DCP_collectives
```

时，DCP 才可能带来 decode 性能收益。即使尚未达到速度 break-even，只要普通布局因 KV OOM 无法服务目标 context，DCP 仍可能是容量上的必要条件。

---



## 2. 哪些模型更适合



### 2.1 从 attention 结构看


| 模型结构                        | 适合度             | 原因与注意点                                                              |
| --------------------------- | --------------- | ------------------------------------------------------------------- |
| Absorbed MLA                | 很高              | latent KV 被 heads 共享，普通 TP 下容易复制；DCP 能直接去掉 context 副本               |
| GQA/MQA                     | 取决于 KV heads/TP | KV heads 很少、TP 已发生复制时有收益；否则可能只是把 head sharding 换成 position sharding |
| Dense MHA                   | 模型/后端特定         | KV width 大，容量需求高，但 Q gather 和 output merge 也更重；当前可靠主路径是 Triton      |
| Hybrid MLA + linear/SSM/KDA | 只改善 MLA 层       | recurrent/state pool 不一定分片，最终并发可能仍被 state 限制                        |
| Sliding-window attention    | 谨慎              | window index、全局/局部位置和 pool 布局都需专门接入；当前 Triton DCP 有显式限制             |
| Sparse/DSA                  | 不能直接推断          | indexer/top-k/cache 的分布语义独立，不能只凭 `--dcp-size` 推断支持                  |




### 2.2 当前仓库 E2E/accuracy test 给出的强证据


| 模型                    | 路径                           | 典型配置                                          | 硬件证据                             | 测试                                                                                              |
| --------------------- | ---------------------------- | --------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------- |
| DeepSeek-V3.1         | FlashInfer MLA DCP           | TP8/DCP8，`ag_rs` 默认                           | 8×H200 CUDA                      | [test_dsv31_dcp8_gsm8k.py](../test/registered/dcp/test_dsv31_dcp8_gsm8k.py#L58)                 |
| Qwen3.5-397B-A17B-FP8 | Triton MHA/GQA DCP           | TP4/DCP4                                      | 4×B200 CUDA、8×MI35x ROCm nightly | [test_qwen3p5_triton_dcp.py](../test/registered/dcp/test_qwen3p5_triton_dcp.py#L18)             |
| Kimi-Linear-48B-A3B   | TokenSpeed MLA DCP           | TP4/DCP4、FP8 KV、A2A、replicated Q              | 4×B200                           | [test_kimi_linear_dcp4.py](../test/registered/dcp/test_kimi_linear_dcp4.py#L23)                 |
| Kimi Linear + DSPARK  | TokenSpeed MLA target verify | DCP4、static verify，分别覆盖 replicated/gathered Q | 4×B200                           | [test_kimi_linear_dcp_dspark4.py](../test/registered/dcp/test_kimi_linear_dcp_dspark4.py#L1)    |
| Kimi Linear PD        | MLA DCP relayout             | prefill/decode DCP4                           | registered PD test               | [test_kimi_linear_pd_dcp4.py](../test/registered/disaggregation/test_kimi_linear_pd_dcp4.py#L1) |


此外，Kimi K3 有模型级 override 和大规模 cookbook recipe。它不接受任意 decode backend，而是限定为 `cutedsl_mla` 或 `tokenspeed_mla`，并按 fabric 选择 `a2a`/`fi_a2a`：[overrides.py](../python/sglang/srt/arg_groups/overrides.py#L395)。

### 2.3 不要把“源码里有 dcp_size 字段”当成端到端支持

完整 DCP 支持至少要求：

1. scheduler/allocator 提供一致的虚拟 token 空间；
2. KV pool 写入时过滤 owner token 并转换下标；
3. attention metadata 只包含本 rank KV shard；
4. attention kernel返回 partial output 和正确 LSE；
5. 模型 forward 做 Q distribution 和 LSE reduction；
6. extend、radix hit、CUDA graph、spec、PD 等使用相同布局；
7. dtype、page size 和 backend 组合经过正确性验证。

缺少任一环节，都可能出现“启动成功但首个 KV write 越界”“decode 能跑但 prefix hit 错”“输出 shape 对但 softmax 不正确”等问题。

---



## 3. DCP、TP、DPA、Prefill CP 的边界



### 3.1 DCP group 嵌套在 TP group 内

初始化时先建立 TP groups，再把每个 TP group 按连续 ranks 切成 DCP groups：[parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2428)。

例如 `TP=8, DCP=4`：

```text
TP group:  [0,1,2,3,4,5,6,7]
DCP group: [0,1,2,3]
DCP group: [4,5,6,7]
```

这 8 个 ranks 可以从 DCP attention 的角度写成一个 `2 × 4` 网格：


| effective head-group | context shard 0 | context shard 1 | context shard 2 | context shard 3 |
| -------------------- | --------------- | --------------- | --------------- | --------------- |
| 0                    | rank 0          | rank 1          | rank 2          | rank 3          |
| 1                    | rank 4          | rank 5          | rank 6          | rank 7          |


这里的 `effective head-group 数 = TP/DCP = 2`。每一行处理不同的 query-head group；一行内部四个 ranks 处理相同 heads 的四份 context/KV shards。因此在算法视图里可以说 head parallel 与 context parallel 是正交的两个轴，但在 SGLang 参数和 process-group 实现里，它们被因式分解在同一个 TP8 rank 集合中，并不是额外申请 `8×4` 张卡。

尤其不要把上述布局写成 `--tp-size 2 --dcp-size 4`。当前 `--tp-size` 表示包含这两个轴的整个 TP group 宽度，所以应写 `--tp-size 8 --dcp-size 4`。

基础硬约束：

```text
dcp_size >= 1
tp_size % dcp_size == 0
platform ∈ {CUDA, HIP}
```

运行时检查见 [parallel_state.py](../python/sglang/srt/distributed/parallel_state.py#L2366)。

### 3.2 与 Attention Data Parallel 组合

DCP group 必须完整落在一个 attention-DP replica、一个有效 attention-TP group 内。没有 Prefill CP 时可以先按下式检查：

```text
attn_tp_size = tp_size / attn_dp_size
attn_tp_size % dcp_size == 0
```

例如：

```text
TP=64, attention DP=4
每个 attention replica 宽度=16

DCP=16: 合法
DCP=32: 跨越两个 attention replica，不合法
```

若还组合 `attn_cp_size > 1`，应使用最终 resolved 的 effective attention TP 维度检查 containment，不要只看原始 `tp_size`。

当前通用启动校验主要检查 `tp_size % dcp_size == 0`，还没有完整替用户检查 DPA containment；源码文档明确标记了这一点：[dcp.mdx](../docs/docs/advanced_features/dcp.mdx#L22)。

### 3.3 Prefill CP 与 DCP 的根本区别


| 项目           | Prefill CP                             | DCP                                             |
| ------------ | -------------------------------------- | ----------------------------------------------- |
| 参数           | `--attn-cp-size`、`--enable-prefill-cp` | `--dcp-size`                                    |
| 分片对象         | 当前 prefill 的 query/hidden token        | 长期 KV cache 的 token position                    |
| 同一 query 的计算 | 一个 rank 负责该 query                      | 每个 rank 计算一个 context shard 的 partial attention  |
| 主要通信         | K/V 或 hidden gather、token layout 恢复    | Q distribution + partial output/LSE merge       |
| 主要目标         | TTFT、prefill compute、activation        | KV capacity、decode KV read、长 context throughput |
| 普通路径 KV 常驻   | 每 CP rank 往往保存完整 token 范围              | 每 DCP rank 只保存自己的 token shard                   |


DCP metadata 单独存放在 `ForwardBatch.attn_dcp_metadata`，不继承 Prefill CP 的 `BaseContextParallelMetadata`，也不参与 CP-v2 strategy contract：[metadata.py](../python/sglang/srt/layers/dcp/metadata.py#L23)。

### 3.4 DCP 不是新增一维 world size

设置 `TP=8,DCP=8` 不需要 64 张卡。DCP 复用 TP8 中的同一批 ranks：

```text
模型权重：仍按 TP8 布局
MLA KV：由这 8 个 ranks 按 token position 分片
Q/output：在这 8 个 ranks 内分发与归约
```

因此它改变的是 attention 数据布局和 collective，不是 `world_size = TP × DCP`。

更精确地说：

```text
算法二维视图：effective head groups = TP / DCP，context shards = DCP
启动参数视图：TP 已经包含 DCP ranks
```

TP 仍然有意义：模型权重和 Q/O projection 等仍保持 TP-sharded；DCP 只在 attention core 周围临时 gather 同一 DCP group 的 Q heads，并在 context/LSE 合并后把 output scatter 回原来的 TP-local head layout。

---



## 4. 虚拟 KV 布局



### 4.1 为什么需要虚拟地址

Scheduler、Radix Tree、request-to-token pool 希望看到统一的全局 token slot。如果每个 rank 自己分配不同的局部 slot，上层 cache tree 将无法用同一组 indices 描述请求。

DCP 让上层继续操作虚拟 slot `v`，到 KV pool/attention backend 边界才转换成 owner-local 物理位置。

设：

```text
C = 每 rank 的物理 token capacity
P = 普通物理 page size
c = dcp_size
```

DCP 暴露：

```text
virtual capacity  = C * c
virtual page size = P * c
owner(v)          = v mod c
physical(v)       = floor(v / c)
```

一个 widened virtual page 恰好在每个 rank 上映射到一个物理 page，因此 page allocation、eviction 和 radix lock 可以保持全 rank 一致。

### 4.2 循环分片为什么比连续分片合适

如果连续分片，长度不断增长的请求可能长期只写某一个 rank，造成尾部 rank 热点或容量不平衡。

循环分片下长度 `L` 在 rank `r` 的 local KV 长度为：

```text
local_len(r) = L // c + int(r < L % c)
```

实现见 [layout.py](../python/sglang/srt/layers/dcp/layout.py#L23)。任意时刻 ranks 间 local length 最多相差 1。

当 prefix 从非零位置 `start` 开始时，`get_dcp_lens()` 会计算第一个属于当前 rank 的 global position，再得到区间内 local token 数；这用于 window/prefix slice 等不从 0 开始的场景。

### 4.3 写入路径

对一批新 token：

```text
mask = positions % dcp_size == dcp_rank
local_loc = virtual_loc // dcp_size

只把 mask=True 的 K/V 写入 local_loc
```

MLA pool 的写入会按 DCP owner 过滤 token：[memory_pool.py](../python/sglang/srt/mem_cache/memory_pool.py#L4045)。Triton MHA backend 则显式构建 mask 和 `loc // dcp_size`：[triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L1255)。

HIP 路径还会把 `dcp_kv_mask` 放进 `ForwardBatch`：[forward_batch_info.py](../python/sglang/srt/model_executor/forward_batch_info.py#L964)。

### 4.4 源码验证：初次 Prefill 完成后 persistent pool 已经是 local KV

这是理解 DCP 生命周期时非常关键的一点：

> DCP 不是 Prefill 后在每个 rank 保存 full KV，等到 Decode 再读取 full KV 并现场切分；KV 在 Prefill/extend 写入 persistent pool 时就已经按 DCP owner 规则分片。

以初始 prompt 有 16 个 tokens、`DCP=4` 为例。Scheduler 和 request-to-token pool 维护完整的逻辑/virtual token mapping：

```text
request logical positions: 0,1,2,...,15
```

模型在 Prefill/extend forward 中产生这些 token 的新 KV。根据具体模型和 backend，各 DCP rank 可能暂时都计算出本轮全部新 token 的 latent K/V；但写入长期 KV pool 时会执行 owner filter 和 virtual-to-physical 转换：

```text
rank0 persistent pool:
  global/virtual positions 0,4,8,12
  local physical rows      0,1,2,3

rank1 persistent pool:
  global/virtual positions 1,5,9,13
  local physical rows      0,1,2,3

rank2 persistent pool:
  global/virtual positions 2,6,10,14
  local physical rows      0,1,2,3

rank3 persistent pool:
  global/virtual positions 3,7,11,15
  local physical rows      0,1,2,3
```

所以 Prefill 完成的一刻，长期状态已经是：

```text
完整逻辑 metadata：每个 rank 都知道请求长度 16 和完整 virtual mapping
KV payload：         每个 rank 只占有 4 个 owner tokens 的 local shard
```

可以把它类比为“每个 rank 都有完整目录，但实际只保存目录中属于自己的书页”。完整 metadata 用于 scheduler、Radix Tree 和跨 rank 一致性；真正占 KV 显存的数据从一开始就是 local 的。

第一次 Decode、为 position 16 计算 attention 时，也不会先读取 full KV：

```text
global seq_len = 16
        │
        ├─ rank0 local page table -> token 0,4,8,12
        ├─ rank1 local page table -> token 1,5,9,13
        ├─ rank2 local page table -> token 2,6,10,14
        └─ rank3 local page table -> token 3,7,11,15
        │
        ▼
每个 attention kernel 直接读取本 GPU persistent local pool
        │
        ▼
partial output + local LSE -> DCP merge
```

Decode planner 根据全局请求 metadata 构造 local `kv_lens/kv_indptr/kv_indices`，而不是从 full KV tensor 中做 slice：[planner.py](../python/sglang/srt/layers/dcp/planner.py#L136)。Triton backend 同样直接生成 per-rank local KV indices：[triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L384)。

position 16 的新 KV owner 是：

```text
16 % 4 = 0
```

因此只有 rank0 把它写到 local physical row `16 // 4 = 4`。下一步全局长度为 17，各 rank local lengths 是：

```text
rank0: 5
rank1: 4
rank2: 4
rank3: 4
```

需要区分一个例外：当后续 `EXTEND` 命中已有 DCP-sharded prefix 时，某些 MLA extend backend 会把历史 prefix shards 临时 AllGather/reorder 到 `dcp_kv_buffer`，供期望 full prefix 的 extend kernel 使用。这只是当前 forward 的临时计算 buffer，不会把 full KV 永久写回每张卡；persistent pool 仍保持 local shard。首次 16-token Prefill 的历史 prefix 为空，通常也没有这部分 gather。

一句话记忆：

```text
初次 Prefill：计算新 KV -> owner-filtered local persistent write
纯 Decode：    local page table -> 直接读取 local persistent KV
Prefix Extend：必要时临时 gather prefix，但 persistent KV 仍是 local
```



### 4.5 Allocator 与 page size

DCP 启用时，即使原始 page size 是 1，也会使用 paged allocator：

```text
allocator capacity  = physical_capacity * dcp_size
allocator page_size = physical_page_size * dcp_size
```

构造位置见 [kv_cache_configurator.py](../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1707)。TreeCache 使用 widened page size，避免 allocator 与 cache tree 在 eviction/page boundary 上不一致：[kv_cache_builder.py](../python/sglang/srt/mem_cache/kv_cache_builder.py#L212)。

### 4.6 手算一个 page 例子

假设 `physical_page_size=2, DCP=4`：

```text
virtual_page_size = 8
virtual page [0..7]

rank0 physical row: virtual 0,4
rank1 physical row: virtual 1,5
rank2 physical row: virtual 2,6
rank3 physical row: virtual 3,7
```

所以一个 virtual page 在每 rank 恰好占 2 个物理 token slots。

---



## 5. 启动阶段的数据流



### 5.1 参数入口

核心参数：

```bash
--dcp-size N
--dcp-comm-backend ag_rs|a2a|fi_a2a
--dcp-replicate-q-proj
# 或显式关闭模型默认
--no-dcp-replicate-q-proj
```

`--decode-context-parallel-size` 是 `--dcp-size` 的 alias。参数定义见 [server_args.py](../python/sglang/srt/server_args.py#L1010)，通信和 replicated-Q 参数见 [server_args.py](../python/sglang/srt/server_args.py#L1082)。

#### 5.1.1 先解释 `--no-dcp-replicate-q-proj`

这个参数**不会关闭 DCP**，也不会关闭 KV token-position 分片。它只明确关闭“复制 Q projection 权重来省掉逐层 Q-head AllGather”这一项优化。

先回到前面已经建立的数据流：

```text
默认 gathered-Q：
  每 rank 用 TP-local Q 权重计算 H_local 个 heads
  -> DCP group 内 Q-head AllGather
  -> 每 rank 得到 H_group 个 heads

replicated-Q：
  启动时让每 rank 准备 DCP-group 所需的 Q 权重
  -> 每 rank 冗余计算 H_group 个 heads
  -> 运行时省掉 Q-head AllGather
```

因此两条命令的直接含义是：

```bash
--dcp-replicate-q-proj
# 用更多 Q 权重显存和冗余 projection 计算，换掉每层 Q AllGather

--no-dcp-replicate-q-proj
# 不复制 group Q 权重；保留 TP-local Q projection，每层运行 Q AllGather
```

`argparse.BooleanOptionalAction` 让这个配置实际具有三态，而不是简单的 true/false：

| 用户写法 | resolved 前的含义 | 实际用途 |
|---|---|---|
| 两个 flag 都不写 | `None`，未显式决定 | 允许模型 override 选择默认值 |
| `--dcp-replicate-q-proj` | `True` | 强制请求 replicated-Q |
| `--no-dcp-replicate-q-proj` | `False` | 即使模型默认想开启，也明确要求关闭 |

例如 Kimi K3 在用户没有显式选择时默认开启 replicated-Q；如果要做 gathered-Q 对照实验或规避额外 Q 权重显存，应显式传：

```bash
--no-dcp-replicate-q-proj
```

必须同时记住四个边界：

1. 当前 replicated-Q 只实现于支持该优化的 MLA decode 路径；这里先把它当作一种 Q 分发优化，不展开 MLA 数学。
2. 前面学习的 Triton GQA 路径不使用这个优化；设置它不会把 GQA 的 Q AllGather 变成本地 replicated projection。
3. 开启它只消除 Q collective，不会消除 partial output/LSE 的跨 rank 合并。
4. 当前准备逻辑只支持合适的 BF16/FP16、非量化 Q projection/关联权重；不满足条件的 layer 会打印 warning，并继续走 Q AllGather。因此命令行是 `True` 不等于所有 layer 一定成功消除 Q AllGather。

源码在 model-runner 初始化、CUDA graph capture 之前准备 replicated weights：[model_runner.py](../python/sglang/srt/model_executor/model_runner.py#L949)。

#### 5.1.2 `--dcp-comm-backend` 控制的不是本地 attention kernel

需要区分两类 backend：

```text
attention backend：
  在单张 GPU 上读取 local KV，计算 partial output + local LSE

dcp_comm_backend：
  local attention 完成后，怎样在 DCP group 内交换/合并 partial output + LSE
```

所以 `--dcp-comm-backend` 不是在 FlashInfer、Triton、FlashMLA 等本地 attention kernel 之间做选择；它选择的是 DCP 跨 rank reduction/exchange 方案。

#### 5.1.3 三种通信后端到底有什么区别

三种方案都不会改变数学结果，也不会改变 persistent KV owner 布局：

```text
共同输入：每 rank 的 partial output o_r + local LSE_r
共同输出：合并后的正确 attention output，恢复为 TP-local heads
```

区别在于这些数据怎样交换：

| 后端 | Q 怎样得到 | partial output/LSE 怎样合并 | 主要特点 |
|---|---|---|---|
| `ag_rs` | 每层 Q-head AllGather | LSE AllGather，修正后的 FP32 output ReduceScatter | 默认、逻辑直接、适合作为正确性起点 |
| `a2a` | 默认 Q AllGather；可配 replicated-Q | 把目标 head owner 所需的 output+LSE 打包做 NCCL All-to-All，接收端本地 LSE combine | 减少/融合结果侧 collective，依赖实现与拓扑性能 |
| `fi_a2a` | 默认 Q AllGather；可配 replicated-Q | 语义同 `a2a`，但交换由 FlashInfer MNNVL kernel 完成 | 面向支持 MNNVL fabric 的 NVIDIA 平台 |

把每层主要 collective 数量写得更直接：

```text
ag_rs + gathered Q:
  Q AllGather
  LSE AllGather
  corrected output ReduceScatter

a2a + gathered Q:
  Q AllGather
  packed(output + LSE) AllToAll

a2a + replicated Q:
  packed(output + LSE) AllToAll

fi_a2a：
  collective 结构与对应 a2a 模式相同，但 output+LSE exchange 使用 MNNVL kernel
```

`a2a/fi_a2a` 不是近似算法。接收端拿到所有 context shards 对自己 head shard 的 `(partial output,LSE)` 后，仍按全局 softmax 公式精确 combine。

#### 5.1.4 硬件和参数约束

| 组合 | 是否允许 | 原因/限制 |
|---|---|---|
| `dcp_size=1 + ag_rs` | 允许，但没有 DCP collective | 默认配置 |
| `dcp_size=1 + a2a/fi_a2a` | 拒绝 | 没有多个 context ranks，无 A2A 意义 |
| `ag_rs + --dcp-replicate-q-proj` | 拒绝 | 当前 replicated-Q 只接到 A2A 路径 |
| `a2a + --dcp-replicate-q-proj` | 条件支持 | 还要求模型、dtype、权重格式和 decode path 支持 |
| `fi_a2a` | 条件支持 | NVIDIA CUDA、SM90+、MNNVL fabric、匹配的 FlashInfer 包和已初始化 workspace |

`fi_a2a` 的典型目标是 GB200/GB300 NVL fabric。普通 PCIe/NVLink 机器即使 GPU 架构足够新，只要 `is_mnnvl_fabric_supported()` 为 false，也不能使用，应退回 `a2a` 或 `ag_rs`。

#### 5.1.5 对当前 GQA 例子，用户实际应该设置什么

前面的 `TP8/DCP4,Hq64/Hkv2` Triton GQA 路径当前固定使用：

```text
Q-head AllGather
local GQA attention
LSE AllGather
corrected output AllReduce
local-head slice
```

它不根据 `--dcp-comm-backend a2a|fi_a2a` 切换到 MLA 的 packed A2A merge，也不支持 `--dcp-replicate-q-proj` 所描述的 Q-weight replication。因此学习或启动这条 GQA 路径时，核心参数是：

```bash
--tp-size 8
--dcp-size 4
```

通信参数保持默认即可，不要因为看到 `a2a` 理论 collective 更少就假设它会优化当前 Triton GQA 分支。

#### 5.1.6 常见配置怎样读

```bash
# 1. gathered-Q + 通用 AG/RS 起点
--dcp-size 4 \
--dcp-comm-backend ag_rs \
--no-dcp-replicate-q-proj

# 2. gathered-Q + packed A2A
--dcp-size 4 \
--dcp-comm-backend a2a \
--no-dcp-replicate-q-proj

# 3. replicated-Q + packed A2A
--dcp-size 4 \
--dcp-comm-backend a2a \
--dcp-replicate-q-proj

# 4. MNNVL 平台上的 replicated-Q + FlashInfer A2A
--dcp-size 4 \
--dcp-comm-backend fi_a2a \
--dcp-replicate-q-proj
```

后面三组都不是仅凭参数就保证可用；还要通过模型 override、attention backend、dtype/quantization、硬件 fabric 和运行时 probe。最终应查看 resolved 启动日志，确认实际选择，而不是只看原始命令行。

### 5.2 通用校验

`ServerArgs._handle_dcp_validation()` 检查：[server_args.py](../python/sglang/srt/server_args.py#L3921)

```text
dcp_size >= 1
a2a/fi_a2a 要求 dcp_size > 1
fi_a2a 要求 CUDA；启动 model runner 后还会做真实 MNNVL probe
dcp_replicate_q_proj 要求 dcp_size > 1
dcp_replicate_q_proj 只允许 a2a/fi_a2a
```

分布式初始化再检查 CUDA/HIP 平台以及 `tp_size % dcp_size == 0`。

### 5.3 创建 DCP group

每个 TP group 按连续 `dcp_size` 切片，创建 `_DCP: GroupCoordinator`。运行时统一通过：

```python
get_parallel().dcp_enabled
get_parallel().dcp_size
get_parallel().dcp_rank
get_parallel().dcp_group
get_parallel().dcp_comm_backend
```

访问。旧的 `layers.dcp.comm` accessor 已标记 deprecated，新的调用点应走 `runtime_context.get_parallel()`。

### 5.4 模型级 override

模型可以重写用户未显式固定的默认值。Kimi K3 是当前最完整的例子：[overrides.py](../python/sglang/srt/arg_groups/overrides.py#L395)

- DCP 下关闭 `--enable-symm-mem`，避免 decode CUDA graph 正确性问题；
- 默认 prefill backend 使用 `trtllm_mla`；
- decode backend 使用 `cutedsl_mla`；
- 也允许 `tokenspeed_mla`，并强制 FP8 E4M3 KV；
- 默认开启 replicated Q projection；
- MNNVL 设备选 `fi_a2a`，其他设备选 `a2a`；
- DSPARK 要求 static ragged verify，并把 target verify 路由到 decode backend。

因此排障时必须看 resolved 启动日志，而不是只看命令行原始值。

### 5.5 Pool sizing 与 graph/workspace 初始化

启用 DCP 后，KV configurator 使用 widened virtual capacity/page。`fi_a2a` 还必须在 CUDA graph capture 前：

1. 分配 FlashInfer MNNVL workspace；
2. 初始化跨 rank FIFO/映射；
3. 在 DCP group barrier；
4. 再进行 decode graph capture。

初始化入口见 [comm.py](../python/sglang/srt/layers/dcp/comm.py#L387)。如果某个 rank 未完成 workspace init 就开始 peer write，可能导致 deadlock。

---



## 6. 源码展开：MLA DCP Decode 六步分别落在哪里

这一章才开始引入 MLA。先建立与前面 GQA 例子的对应关系：DCP 的 token owner、local page table、partial attention 和 LSE merge 都没有变；变化的是“长期 K/V payload 到底是什么”。

### 6.0 MLA 为什么是特殊情况

MLA 把每个 token 的 K/V 压缩为所有 query heads 共享的 latent：

```text
hidden h_t
  ├─ q_a（常为 replicated）-> q_b（按 TP heads 切）-> per-head Q
  └─ kv_a（replicated）-> c_t^KV + k_t^rope

persistent MLA cache shape（概念上）：
[token positions, 1, kv_lora_rank + rope_head_dim]
```

`kv_b` 虽然是按 head/TP 切的，但 absorbed decode 不把它展开成 per-head K/V 后写 cache，而是在加载权重后拆成：

```text
W_UK 的 TP-local head 切片 -> w_kc：attention 前吸收到 Q
W_UV 的 TP-local head 切片 -> w_vc：attention 后作用于 latent output
```

对应的结合律是：

```text
Q[h] · (W_UK[h] · c_j) = (Q[h] · W_UK[h]) · c_j

sum_j a[h,j] · (W_UV[h] · c_j)
= W_UV[h] · (sum_j a[h,j] · c_j)
```

所以 MLA 的 head-specific 信息在 `q_b/w_kc/w_vc`，persistent cache 只保存共享 latent。源码中 `q_a/kv_a` 常融合为 `ReplicatedLinear`，`q_b/kv_b` 是 column-parallel：[deepseek_v2.py](../python/sglang/srt/models/deepseek_v2.py#L1778)。加载后拆出 `w_kc/w_vc` 的逻辑见 [deepseek_weight_loader.py](../python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py#L481)，KV pool 宽度见 [memory_pool.py](../python/sglang/srt/mem_cache/memory_pool.py#L3932)。

对 `TP8/DCP4`，它仍然是两个 query-head groups、每组四个 context shards：

```text
group [r0,r1,r2,r3]：一组 Q heads + 一份按 positions 分成 4 片的 latent context
group [r4,r5,r6,r7]：另一组 Q heads + 另一份按 positions 分成 4 片的 latent context
```

与 GQA 相比，MLA 更特殊的地方只是 KV payload 没有普通 KV-head 维，以及可以用 replicated-Q projection 替代逐层 Q-head AllGather。下面再沿源码展开。



### 6.1 Batch/attention metadata 先变成 local KV 视图

普通 decode metadata 包含全局 `kv_lens/kv_indptr/kv_indices`。DCP planner：

1. 根据 `dcp_rank/dcp_size` 计算每个请求的 local KV length；
2. 对 local lengths 做 prefix sum，生成 local `kv_indptr`；
3. 从全局 indices 中筛出 owner positions；
4. 把 virtual indices 转换为 local physical indices；
5. 原地更新 backend 将使用的 `kv_lens/kv_indptr/kv_indices`。

实现见 [plan_dcp_decode_metadata](../python/sglang/srt/layers/dcp/planner.py#L136)。

因此 attention kernel 实际看到的是：

```text
global request length: L
rank r kernel KV length: L//C + (r < L%C)
kernel page table: only rank-r physical KV rows
```



### 6.2 生成完整 DCP-group Q

普通 MLA TP 下，每个 rank 只计算自己的 local query heads。但每个 DCP rank 只持有部分 context，为了让各 context shard 都对完整 heads 贡献 partial result，需要完整 DCP-group Q。

默认路径：

```text
local q_nope + q_rope
        │ DCP all-gather along head dim
        ▼
full DCP-group q_nope + q_rope
```

函数见 [all_gather_q_for_mla_decode](../python/sglang/srt/layers/dcp/comm.py#L174)。

这里 gather 的是 query **head 维**：普通 Decode 即使每个请求只有 1 个 query token，TP projection 之后各 rank 仍只有 `H_local`，因此仍需把它补成 `H_group`。replicated-Q 路径则由每 rank 本地完整计算这组 heads，跳过每层 Q all-gather，后文单独分析。

### 6.3 本地 attention kernel

每个 rank 用相同的 full DCP-group Q 查询自己的 local KV shard：

```text
rank r:
  Q = full DCP-group query heads
  K/V = token positions where position % C == r
  output = partial_o_r
  stats  = local_lse_r
```

FlashInfer MLA 在 DCP decode 时显式设置 `return_lse=True`：[flashinfer_mla_backend.py](../python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L721)。FlashMLA BF16 路径同样返回 output+LSE：[flashmla_backend.py](../python/sglang/srt/layers/attention/flashmla_backend.py#L462)。

### 6.4 为什么 partial output 不能直接相加

设全局 logits 被 context shard 切为 `x_0, x_1, ...`：

```text
softmax([x_0,x_1]) @ [V_0,V_1]
```

局部 kernel 算的是：

```text
o_0 = softmax(x_0) @ V_0
o_1 = softmax(x_1) @ V_1
```

因为 `softmax(x_0)` 和 `softmax(x_1)` 各自除以不同分母：

```text
o_0 + o_1 != global_attention
average(o_0,o_1) != global_attention
```

必须保留：

```text
lse_r = log(sum(exp(x_r)))
```

用全局归一化权重修正。

### 6.5 LSE 精确合并

令：

```text
L = logsumexp(lse_0, ..., lse_C-1)
w_r = exp(lse_r - L)
```

则：

```text
global_o = sum_r(w_r * o_r)
```

源码还必须知道 backend 的 LSE log base：

- FlashInfer MLA 常返回 base-2 LSE；
- FlashMLA、CuteDSL MLA 使用自然对数路径；
- 使用错误的 `exp/exp2` 会得到 shape 正确但数值错误的结果。

AG/RS 实现在 [cp_lse_ag_out_rs_mla](../python/sglang/srt/layers/dcp/comm.py#L112)，模型侧 dispatch 在 [forward_mla.py](../python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L753)。

### 6.6 恢复 TP-local head layout

local attention 为完整 DCP-group heads 产生 partial output。跨 rank合并时不只要合并 context，还要让每个 rank 最终只拿回自己原来的 TP-local head shard。

`ag_rs` 用 output reduce-scatter；`a2a` 则按目标 head owner 把 partial output+LSE 发给对应 rank，再在接收端合并。因此最终模型后续层仍看到普通 TP-local output shape，不需要知道 attention 曾按 context 分片。

### 6.7 新 KV 写入

当前 decode token 的 virtual position 在所有 ranks 一致，但只有 owner rank 写入：

```text
owner = position % C
physical = position // C
```

下一 decode step 构建 local page table 后，只有 owner rank 会在自己的 context shard 中看到该 token。

---



## 7. Triton MHA/GQA DCP 路径



### 7.1 为什么它与 MLA 不完全相同

MHA/GQA 原本可以沿 KV heads 做 TP。DCP 加入后，模型需要重新协调：

- Q 仍按 attention TP heads 分片；
- DCP group 内 K/V heads 可能复制；
- token positions 再由 DCP ranks 分片。

Qwen3.5 的模型初始化计算：[qwen3_5.py](../python/sglang/srt/models/qwen3_5.py#L853)

```python
kv_tp_size = attn_tp_size // dcp_size
kv_tp_rank = attn_tp_rank // dcp_size
```

因此不能笼统地说“Q/K/V 权重全部按 TP8 切”。以 `TP=8,DCP=4` 为例：

```text
Q/O/MLP 的 TP 宽度：        8
K/V head 的有效 TP 宽度：   kv_tp_size = TP/DCP = 2
DCP group 数：              2
每个 group 内 context ranks：4
```

具体是：

- Q projection 仍按 TP8 切，每 rank 持有 `Hq/8` 个 Q heads 的权重；
- K/V projection 只在两个 effective KV-head groups 之间切；
- 同一个 DCP4 group 内的四个 ranks 使用完全相同的 K/V-head weight shard；
- 这四个 ranks 再分别保存该 K/V-head shard 的不同 token positions。

假设 `Hq=64,Hkv=8`：

```text
DCP group [r0,r1,r2,r3]
  Q weights:  r0/r1/r2/r3 分别拥有 8 个不同 Q heads
  K/V weights:四个 ranks 都拥有 KV heads 0..3
  KV cache:   四个 ranks 分别存 positions % 4 == 0/1/2/3

DCP group [r4,r5,r6,r7]
  Q weights:  r4/r5/r6/r7 分别拥有另外 8 个不同 Q heads
  K/V weights:四个 ranks 都拥有 KV heads 4..7
  KV cache:   四个 ranks 分别存 positions % 4 == 0/1/2/3
```

因此 fused `qkv_proj` 在一个 rank 上实际混合了两种布局：Q shard 由 `tp_rank/tp_size=rank/8` 选择，而 K/V shard 由 `kv_tp_rank/kv_tp_size=(rank//4)/2` 选择。实现见 [qwen3_5.py](../python/sglang/srt/models/qwen3_5.py#L904) 和 [linear.py](../python/sglang/srt/layers/linear.py#L921)。

#### KV heads 少于并行宽度时怎样复制

要比较的不是原始 `TP=8`，而是：

```text
G = effective KV-head TP = TP / DCP = 2
```

每 rank 的 K/V head 数为：

```text
num_kv_heads_per_rank = max(1, Hkv / G)
```

同时要求 `Hkv` 与 `G` 之间可以整除。分三种情况：

1. `Hkv > G`：K/V heads 在 `G` 个 groups 之间正常切分；每份再沿 DCP 维复制 `DCP` 次。
2. `Hkv == G`：每个 group 一个 K/V head；组内四个 context ranks 复制这一 head 的权重。
3. `Hkv < G`：一个 K/V head 还要被多个 effective head groups 共享，因此会额外跨 groups 复制；每个 group 内又有 DCP 复制。

对 `TP8/DCP4`：

| 总 KV heads `Hkv` | group 0 的 KV weights | group 1 的 KV weights | 每个 DCP rank 的 KV heads |
|---:|---|---|---:|
| 8 | heads 0..3 | heads 4..7 | 4 |
| 4 | heads 0..1 | heads 2..3 | 2 |
| 2 | head 0 | head 1 | 1 |
| 1 | head 0 | head 0（跨 group 再复制） | 1 |

`QKVParallelLinear` 对 `kv_tp_size >= total_num_kv_heads` 的情况显式设置 `num_kv_heads=1`，并计算 `num_kv_head_replicas=kv_tp_size/total_num_kv_heads`：[linear.py](../python/sglang/srt/layers/linear.py#L985)。所以答案是：**会复制，但判断“KV heads 是否不够”的门槛是 `TP/DCP`，不是原始 TP。**此外，同一 DCP group 内为了让每个 token owner 都能本地生成 K/V，K/V 权重本来就会沿 DCP 维复制。

#### 为什么只能说 attention core 是 effective TP2

Decode 时每 rank 先产生 `Hq/8` 个 Q heads，DCP group 内 AllGather 后，每个 rank 临时拥有：

```text
(Hq/8) × DCP4 = Hq/2
```

于是两个 DCP groups 各负责一半 Q heads，看起来是 effective head-TP2；但 local attention 完成并按 LSE 合并后，output 会 slice/reduce-scatter 回 `Hq/8`，后续 `o_proj` 和其余模型计算仍是 TP8。**不是整个运行从 TP8 变成了 TP2，只是 DCP attention core 的二维视图是 head-groups 2 × context-shards 4。**

#### GQA 的 KV 容量收益为什么未必等于 DCP4

因为启用 DCP 后，每 rank 的 token 数缩为 `L/4`，但每 rank 持有的 K/V heads 可能增加。粗略比较单层每 rank cache payload：

```text
DCP1: max(1, Hkv/TP)       × L
DCP4: max(1, Hkv/(TP/DCP)) × L/DCP
```

仍以 `TP8/DCP4` 为例：

| `Hkv` | DCP1 每 rank cache | DCP4 每 rank cache | 容量改善 |
|---:|---:|---:|---:|
| 8 | `1×L` | `4×L/4 = 1×L` | `1×` |
| 4 | `1×L` | `2×L/4 = 0.5×L` | `2×` |
| 2 | `1×L` | `1×L/4 = 0.25×L` | `4×` |
| 1 | `1×L` | `1×L/4 = 0.25×L` | `4×` |

因此 GQA/MQA 原先的 KV-head replication 越严重，DCP 越有机会把这些副本转换为 token-position shards；如果 KV heads 原本已经能被 TP8 完整切开，DCP4 只是用“每 rank 更多 KV heads”交换“每 rank 更少 positions”，KV cache 容量未必增加。

最终收益取决于总 KV heads、TP degree 和原本是否已经复制 KV heads，不能像 MLA 那样一律认为容量提升恰好为 `C` 倍。

### 7.2 Decode metadata

Triton backend 的 `_dcp_kv_indices()`：

1. 计算每请求 local DCP lens；
2. 生成 local `kv_indptr`；
3. 通过 Triton kernel 从 `req_to_token` 生成 owner-local indices；
4. eager 创建新 tensor，CUDA graph replay 写入 address-stable buffer。

入口见 [triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L382)。

### 7.3 Decode attention

Triton MHA DCP：[triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L1853)

```text
TP-local Q
  -> DCP group all-gather Q heads
  -> each rank decode_attention_fwd(local K/V indices)
  -> local split-K LSE 再 logsumexp
  -> cp_lse_ag_out_rs_mha
  -> TP-local output heads
```

MHA merge 当前使用：

1. LSE all-gather；
2. 根据 global LSE 修正 local output；
3. output all-reduce；
4. 每 rank slice 自己的 head range。

实现见 [cp_lse_ag_out_rs_mha](../python/sglang/srt/layers/dcp/comm.py#L83)。它没有使用 MLA 的 `--dcp-comm-backend a2a/fi_a2a` dispatch。

### 7.4 Extend 路径

Triton DCP extend 把 attention 分成：

- 已有 prefix：按 DCP local KV shard 计算 partial attention；
- 当前 extend tokens：当前 forward 中直接计算；
- prefix partial outputs 先跨 rank LSE merge；
- 再用 `logaddexp(prefix_lse, current_lse)` 合并 prefix 与当前段。

相关实现见 [triton_backend.py](../python/sglang/srt/layers/attention/triton_backend.py#L1568)。

### 7.5 当前显式限制

Triton DCP 路径目前显式拒绝或限制：

- decode `score_mod`；
- extend `score_mod`；
- extend sinks；
- extend custom masks；
- extend sliding window。

因此“普通 Triton backend 支持某功能”不等于“DCP Triton 分支也支持该功能”。

---



## 8. Extend、prefix hit 与 DCP metadata



### 8.1 为什么 extend 比 decode 更复杂

纯 decode 每请求通常只有一个新 query，历史 KV 已按 DCP 布局稳定存在。Extend 可能同时包含：

- radix/prefix cache 命中的历史 tokens；
- 本轮新增的多个 tokens；
- 多请求、不同 prefix length；
- chunked prefill；
- target verification 的多个 query tokens。

已有 prefix 分散在 ranks 上，而很多成熟 prefill/extend kernel 期望完整、自然 token 顺序的 KV，因此某些 MLA 路径会临时 all-gather prefix shards。

### 8.2 `DecodeContextParallelMetadata`

当前 dataclass 字段：[metadata.py](../python/sglang/srt/layers/dcp/metadata.py#L30)

```text
dcp_kv_indptr
dcp_kv_buffer
dcp_kv_indices
dcp_local_prefix_kv_indices
dcp_extend_prefix_lens_sum
```

它描述：

- 每个请求在 DCP buffer 中的边界；
- 临时收集/重排 prefix KV 的 buffer；
- 哪些 prefix physical rows 属于当前 rank；
- prefix 总长度与新增 token 区域分界。



### 8.3 Metadata 构建

`prepare_decode_context_parallel_metadata()`：[planner.py](../python/sglang/srt/layers/dcp/planner.py#L32)

1. 从 request-to-token pool 取出全局 prefix virtual indices；
2. 根据 `% dcp_size == dcp_rank` 过滤本 rank owner indices；
3. 做 `// dcp_size` 转换为 local physical indices；
4. 构造 DCP KV indptr/indices；
5. 分配临时 `dcp_kv_buffer`。



### 8.4 MLA extend 的 prefix gather

`all_gather_kv_cache_for_dcp()` 会：

1. 为每请求计算各 rank local length；
2. 对非整除长度做左右 padding；
3. all-gather local KV；
4. 把 rank-major 结果恢复为原始 token 顺序；
5. 去掉 padding。

调用链集中在 [comm.py](../python/sglang/srt/layers/dcp/comm.py#L203)。

这解释了为什么 DCP 的简化性能模型主要适用于 decode：

```text
decode: local KV read + context-independent output merge
extend: 可能增加与 prefix length 成比例的 KV gather/reorder
```



### 8.5 Prefix Cache/Radix Cache 的关键条件

Radix tree 必须使用 widened virtual page 边界。否则不同 rank 可能在不同物理 page 边界上 lock/evict 同一逻辑 prefix，导致：

- 部分 rank cache hit，部分 rank miss；
- free/eviction 数量不一致；
- req-to-token mapping 与物理 pool 不一致。

因此 DCP 不是只改 attention kernel，还深入 allocator、tree cache、host pool 和 PD transfer 的下标语义。

---



## 9. 三种通信后端



### 9.1 `ag_rs`：通用 fallback

MLA 每层主要通信：

```text
Q AllGather
LSE AllGather
FP32 corrected output ReduceScatter
```

优点：

- 逻辑直接；
- 通用 GroupCoordinator collective；
- 是通用默认值；
- DeepSeek-V3.1 DCP8 CI 使用默认路径。

缺点：collective 次数较多，output correction/RS 使用 FP32 scratch/通信。

### 9.2 `a2a`：packed NCCL all-to-all

`a2a` 把目标 head owner 需要的：

```text
partial output + FP32 LSE
```

打包到同一个 tensor，通过一次 `all_to_all_single` 交换。接收端拥有所有 context ranks 对自己 head shard 的 partials，再本地运行 Triton LSE combine。

实现见 [dcp_a2a_lse_reduce](../python/sglang/srt/layers/dcp/comm.py#L452)。代码以 raw bytes 传输，避免 FP8/pynccl dtype enum 问题。

两种 Q 模式：

```text
gathered Q:
  Q AG + packed A2A = 每层两个主要 collective

replicated Q:
  packed A2A = 每层一个主要 collective
```



### 9.3 `fi_a2a`：FlashInfer MNNVL

`fi_a2a` 复用同样的本地 LSE combine，但把跨 rank exchange 交给 FlashInfer DCP MNNVL kernel。

要求：

- NVIDIA CUDA；
- SM90+；
- `is_mnnvl_fabric_supported()` 为 true；
- FlashInfer 包含 `flashinfer.comm.dcp_alltoall`；
- graph capture 前初始化 workspace。

典型平台是 GB200 NVL72。没有 MNNVL 时应使用 `a2a` 或 `ag_rs`。

### 9.4 怎么选


| 场景                       | 建议起点                                   |
| ------------------------ | -------------------------------------- |
| 通用 DeepSeek MLA、先验证正确性   | `ag_rs`                                |
| B200/B300 等高速互联、无 MNNVL  | `a2a`                                  |
| GB200/GB300 MNNVL fabric | `fi_a2a`                               |
| Kimi K3                  | 让模型 override 自动选                       |
| Triton MHA/GQA           | 当前使用固定 MHA AG+all-reduce merge，不由该参数选择 |


真正最优项取决于 batch、heads、DCP size、节点边界和是否启用 replicated Q；必须 profile，不能只按理论 collective 次数决定。

---



## 10. Query projection replication



### 10.1 默认 gathered-Q 路径

普通 TP 已将 Q projection output heads 分片。DCP decode 需要 full DCP-group heads，于是每层执行：

```text
local q_nope/q_rope -> DCP all-gather -> full query heads
```

优点是：

- 沿用普通 TP weight layout；
- 不增加投影权重显存；
- 不增加 Q projection FLOPs。

缺点是每个 MLA 层多一次 Q collective。

### 10.2 Replicated-Q 路径

`--dcp-replicate-q-proj` 在启动时收集每 rank 的：

- Q projection weights；
- absorbed MLA 使用的 `w_kc` 等关联权重；

并让每个 DCP rank 本地计算完整 group query heads。

权衡：

```text
增加：Q 权重显存 + 重复 Q GEMM
减少：每层一次 Q all-gather
```

它适合 decode batch 较小、collective latency 显著、Q projection 相对便宜的场景。

### 10.3 当前限制

- 只用于 MLA；
- 只允许 `a2a`/`fi_a2a`；
- 只对未量化 BF16/FP16 eligible weights 生效；
- 不支持的层会回退到 Q all-gather；
- 增加的 weight memory 可能挤占 DCP 节省下来的 KV 空间。

Kimi K3 在 DCP 下默认开启该模式；Kimi Linear acceptance test 同时显式覆盖该配置：[test_kimi_linear_dcp4.py](../test/registered/dcp/test_kimi_linear_dcp4.py#L59)。

---



## 11. Attention backend 与硬件支持



### 11.1 MLA attention backends


| Backend          | DCP 状态                  | 重要限制/证据                                                            |
| ---------------- | ----------------------- | ------------------------------------------------------------------ |
| `flashinfer`     | 支持 MLA decode DCP       | DeepSeek-V3.1 H200 CI；DCP decode 要求 kernel 返回 LSE                  |
| `flashmla`       | BF16/FP16 KV 路径支持       | FP8 KV + DCP 显式 assert 不支持                                         |
| `trtllm_mla`     | 普通 DCP decode 可用路径      | DCP + speculative decode 被 registry 拒绝                             |
| `cutedsl_mla`    | 支持 decode/target verify | Kimi K3 canonical DCP decode backend；依赖新版 FlashInfer DCP signature |
| `tokenspeed_mla` | 支持 Kimi MLA DCP         | Blackwell、FP8 KV、CUDA graph/eager acceptance test                  |


backend registry 对 `trtllm_mla + DCP + spec` 的拒绝以及替代建议见 [attention_registry.py](../python/sglang/srt/layers/attention/attention_registry.py#L69)。

FlashMLA 的明确限制：[flashmla_backend.py](../python/sglang/srt/layers/attention/flashmla_backend.py#L425)

```text
FlashMLA does not support DCP for FP8 kv cache
```

因此不能只看 backend 名称，还要同时确认 KV dtype。

### 11.2 MHA/GQA attention backend

当前有端到端强证据的是 `triton`：

- Qwen3.5 TP4/DCP4 B200 nightly；
- Qwen3.5 MI35x ROCm nightly；
- decode、extend、CUDA graph metadata 都有专门 DCP 分支。

普通 `flashinfer` MHA backend 中出现 `attn_dcp_size` 不足以证明完整 DCP 数据流；当前学习和部署应以 Triton registered recipe 为基线。

### 11.3 NVIDIA CUDA

CUDA 当前覆盖最完整：

- H200 DeepSeek-V3.1 FlashInfer MLA；
- B200 Kimi Linear TokenSpeed MLA；
- B200 Qwen3.5 Triton；
- Kimi K3 CuteDSL/TokenSpeed recipes；
- NCCL A2A 和 FlashInfer MNNVL A2A。



### 11.4 AMD ROCm

并行组初始化允许 HIP，Qwen3.5 Triton DCP 有 MI35x nightly accuracy test。MLA ROCm forward 也包含 `ag_rs/a2a` dispatch，但具体模型/backend/dtype 应以对应 AMD test recipe 为准，不能泛化为全部 CUDA 组合都可直接迁移。

### 11.5 其他平台

当前模型并行初始化明确拒绝非 CUDA、非 HIP 的 DCP。也就是说 CPU、XPU、NPU、MUSA 即使局部代码出现相关字段，也不能通过通用 DCP group 初始化路径。

---



## 12. Speculative decoding



### 12.1 Target KV 与 draft KV 的布局不同

当前典型组合：

```text
target MLA KV: 按 DCP token position 分片
draft KV:      每个 DCP rank 复制完整 draft context
```

所以总 KV per-token cost 更接近：

```text
target_kv_bytes / dcp_size + draft_kv_bytes
```

而不是：

```text
(target_kv_bytes + draft_kv_bytes) / dcp_size
```

DCP degree 越大，未分片 draft KV 在总成本中的比例越高，实际容量收益越偏离理想 `C` 倍。

### 12.2 Target verification 为什么也需要 DCP metadata

Target verify 一次处理多个 candidate query tokens，但 target 历史 KV 仍是 cyclic shards。DCP-aware kernel必须获得：

- rank-local page table；
- local KV lengths；
- global sequence lengths，用于完整 causal boundary；
- `cp_world/cp_rank`；
- rank-local output 与 LSE。

然后使用与普通 decode 相同的跨 rank LSE merge。

### 12.3 Kimi K3 / DSPARK 约束

Kimi K3 DCP + DSPARK 当前要求：

- `SGLANG_RAGGED_VERIFY_MODE=static`；
- target verify 路由到 decode backend；
- decode backend 是 `cutedsl_mla` 或合适的 `tokenspeed_mla`；
- `trtllm_mla` 不能作为 DCP speculative decode backend。

模型 override 逻辑见 [overrides.py](../python/sglang/srt/arg_groups/overrides.py#L406)。

---



## 13. PD disaggregation



### 13.1 先分清 P 端的 “CP” 指什么

PD 场景里很容易把两个参数都简称为 CP：

```text
P 节点 Prefill CP：--attn-cp-size > 1
P/D 节点 DCP：     --dcp-size > 1
```

当前代码有一个重要硬限制：

> 当 D 节点启用 DCP 时，P 节点必须满足 `attn_cp_size == 1`。也就是说，当前不支持 `P: Prefill CP > 1 -> D: DCP > 1`。

D 节点在接收 P 节点 bootstrap 信息时会直接检查并拒绝该组合：[disaggregation/common/conn.py](../python/sglang/srt/disaggregation/common/conn.py#L551)。

这不是说 Prefill CP 与 PD 本身不兼容。Prefill CP 可以用于普通 PD prefill/decode 拓扑；受限的是“Prefill CP source 直接向 DCP-sharded decode pool 传 KV”这一组合，因为当前 relayout 只实现了 dense/DCP source 到 DCP destination，没有实现：

```text
Prefill-CP token/head/rank layout
            ->
DCP destination owner/local-physical layout
```

当前支持矩阵：


| P 节点              | D 节点              | 当前状态                | 传输方式                                                  |
| ----------------- | ----------------- | ------------------- | ----------------------------------------------------- |
| `attn_cp=1, DCP1` | `DCP1`            | 支持                  | 普通 PD page transfer                                   |
| `attn_cp=1, DCP1` | `DCPN, N>1`       | 仅 MLA/hybrid-MLA 支持 | sender 按目标 DCP rank 做 token relayout，直接写 D local pool |
| `attn_cp=1, DCPN` | 相同 `DCPN`         | 支持                  | matching DCP rank 的 local page 直传                     |
| `attn_cp=1, DCPM` | 不同 `DCPN`，且 `M>1` | 不支持                 | `Unsupported PD DCP topology`                         |
| `attn_cp>1`       | `DCPN, N>1`       | 不支持                 | bootstrap/connection 阶段拒绝                             |




### 13.2 为什么 dense prefill page 不能原样复制到 DCP pool

Prefill worker 可能保存 dense、自然顺序 KV；DCP decode worker 的目标 layout 是：

```text
owner(v) = v % dcp_size
physical(v) = v // dcp_size
```

所以传输必须知道 decode rank，只选择属于该 rank 的 token rows，并 pack 到目标物理 page。直接 memcpy 完整 page 会：

- 写入非 owner rows；
- physical offset 错误；
- partial final page 携带 stale rows。

因此 DCP PD transfer 的原则不是“D 收到 full KV 后自己切”，而是：

> 发送前就根据目标 `dst_dcp_size/dst_dcp_rank` 选择 source token rows，并把它们直接写到目标 rank 的 local physical rows。



### 13.3 路径一：P DCP1 → D DCPN

以 16-token、DCP4 为例。P 节点是 DCP1，持有 dense KV：

```text
P source pool:
  token 0,1,2,3,4,5,...,15
```

D 节点已经分配好 widened virtual slots，但各 rank 的 physical pool 只需要自己的 shard：

```text
D-rank0 expects: 0,4,8,12
D-rank1 expects: 1,5,9,13
D-rank2 expects: 2,6,10,14
D-rank3 expects: 3,7,11,15
```

每个目标 DCP rank 的注册信息包含：

```text
dst_dcp_size
dst_dcp_rank
destination virtual/page indices
destination KV data pointers
```

P 端据此构建 token-level transfer plan：

```text
                      P full/dense KV
                   token 0,1,2,...,15
                             │
           select rows by destination owner rule
                             │
       ┌──────────────┬──────┴───────┬──────────────┐
       ▼              ▼              ▼              ▼
    0,4,8,12       1,5,9,13      2,6,10,14      3,7,11,15
       │              │              │              │
       ▼              ▼              ▼              ▼
 D-rank0 local   D-rank1 local  D-rank2 local  D-rank3 local
 physical 0..3   physical 0..3  physical 0..3  physical 0..3
```

以 D-rank1 为例，概念上的映射是：

```text
source token rows:       [1,5,9,13]
destination virtual:     [1,5,9,13]
destination physical:    [0,1,2,3]
```

Mooncake/NIXL 直接从 selected source row 写入 D-rank1 的 local pool，没有 D 端 full-KV 临时 buffer，也没有“接收完成后再切分”。NIXL 的 DCP send path 会调用 `build_dcp_token_transfer_plan()`，随后以 flat token descriptors 发送：[disaggregation/nixl/conn.py](../python/sglang/srt/disaggregation/nixl/conn.py#L1651)。

计划中还携带真实 `num_kv_tokens`，避免 partial final page 把未初始化/stale rows 一起传过去。

### 13.4 路径二：P DCPN → D 相同 DCPN

例如两端都是 DCP4：

```text
P-rank0 local: 0,4,8,12  -> D-rank0 local
P-rank1 local: 1,5,9,13  -> D-rank1 local
P-rank2 local: 2,6,10,14 -> D-rank2 local
P-rank3 local: 3,7,11,15 -> D-rank3 local
```

source/destination 已经有相同 owner rule 和 physical page geometry，不需要 token relayout，走已有 page path。但连接必须是 matching DCP ranks：

```text
P dcp_rank 0 只能对接 D dcp_rank 0
P dcp_rank 1 只能对接 D dcp_rank 1
...
```

不匹配会直接报错，校验见 [disaggregation/common/conn.py](../python/sglang/srt/disaggregation/common/conn.py#L281)。

P worker 上开启 DCP 在相同布局下是受支持的，但它通常不能提高 prefill 性能，反而增加通信；启动时会给 warning：[arg_groups/pd_disaggregation_hook.py](../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L30)。因此部署上不应仅为了“和 D 对齐”就默认给 P 打开 DCP，应比较 P-D direct layout 与 DCP1→DCPN relayout 的整体代价。

### 13.5 不支持的 DCP-size 转换

```text
source DCP == destination DCP:
  要求匹配 DCP rank，走已有 page path

source DCP1 -> destination DCP>1:
  MLA/hybrid-MLA 可按 token relayout

其他 DCP size 变化:
  拒绝
```

连接校验见 [disaggregation/common/conn.py](../python/sglang/srt/disaggregation/common/conn.py#L285)，token relayout 规划见 [disaggregation/utils.py](../python/sglang/srt/disaggregation/utils.py#L994)。

### 13.6 D 节点收到以后就是 local pool

无论是 DCP1→DCPN relayout，还是相同 DCPN 的 rank-matched direct transfer，完成通知发出时，D 端 persistent pool 都已经是最终 DCP local layout：

```text
D-rank0 pool: token 0,4,8,12
D-rank1 pool: token 1,5,9,13
D-rank2 pool: token 2,6,10,14
D-rank3 pool: token 3,7,11,15
```

第一次 decode 时，DCP planner 直接构造 local KV lengths/page table，attention kernel 直接读取 local pool，计算 partial output+LSE。不会先在 D 端重建 full KV。

因此完整链路是：

```text
P computes/persists KV
        │
        ├─ same DCP: matching local page direct transfer
        │
        └─ DCP1→DCPN: sender selects token rows + virtual→local mapping
        │
        ▼
D persistent local KV pool
        │
        ▼
local decode attention + cross-rank LSE merge
```



### 13.7 Transfer backend 与其他条件

当前 DCP PD relayout 支持 Mooncake 和 NIXL，共享同一份 token-level plan。要求：

- physical page size 匹配；
- KV dtype/geometry 匹配；
- prefill attention CP=1；
- decode 使用 chunk cache；
- MLA 或 hybrid-MLA KV pool；
- 当前不与 decode radix cache、HiCache 同时使用。

KDA/Mamba state 不按 DCP token owner 过滤，而是维持自己的 attention-TP/request mapping。

这里的限制要分别理解：

- `prefill attention CP=1`：禁止 P Prefill CP→D DCP；
- `MLA/hybrid-MLA pool`：限制 DCP1→DCPN token relayout；
- `Mooncake/NIXL`：其他 transfer backend 没有 DCP transfer path；
- matching physical page/KV dtype：否则 source row bytes 无法安全落到 destination；
- decode chunk cache：当前 DCP PD 不支持 decode radix cache；
- HiCache：当前不能与 PD decode DCP 组合。

---



## 14. HiCache、L3、LMCache 与其他 memory 功能



### 14.1 HiCache L1/L2

当前 HiCache + DCP 只接入 MLA host pool。Controller/Radix 层看到 widened logical page：

```text
logical page size = physical host page size * dcp_size
```

每个 GPU/host rank 真正传输前：

```text
owned = indices[indices % dcp_size == dcp_rank] // dcp_size
```

每 rank 独立做 H2D/D2H，无需额外 DCP collective。host pool 的转换见 [pool_host/base.py](../python/sglang/srt/mem_cache/pool_host/base.py#L335)。

### 14.2 当前不支持的组合

HiCache + DCP 当前不支持：

- L3 storage backend；
- LMCache；
- HiSparse；
- 非 MLA KV host pool；
- speculative decoding；
- PD decode。

ServerArgs 的 fail-fast 校验见 [server_args.py](../python/sglang/srt/server_args.py#L7308)。

L3 不只是“暂时没测”，而是 storage keys 和 rank-0 replicated MLA backup 还没有完整 DCP-rank 语义，必须在启动时拒绝，避免跨 rank shard 覆盖或取错。

### 14.3 Unified memory pool

`--enable-unified-memory` 当前与 DCP 不兼容，因为 unified MHA pool 没有 DCP-aware masked write path；启动校验会直接 assert，见 [server_args.py](../python/sglang/srt/server_args.py#L8226)。

### 14.4 特殊 KV dtype

- FlashMLA FP8 KV：不支持 DCP；
- MXFP8 MHA KV pool：不支持 DCP write mask，[memory_pool.py](../python/sglang/srt/mem_cache/memory_pool.py#L3419)；
- TokenSpeed Kimi 路径：显式使用 FP8 E4M3；
- 普通 FlashInfer MLA FP8/byte transport 需看具体 pool/kernel path。

所以 DCP 的 dtype 支持是“attention backend × pool type × model”三者交集，不是一张全局 KV dtype 白名单。

---



## 15. 可复用启动 recipe



### 15.1 DeepSeek-V3.1：先学通用 MLA `ag_rs`

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 \
  --trust-remote-code \
  --tp-size 8 \
  --dcp-size 8 \
  --attention-backend flashinfer \
  --disable-piecewise-cuda-graph
```

未写 `--dcp-comm-backend` 时默认 `ag_rs`。这个配置适合观察：

1. DCP group；
2. widened allocator/page；
3. local decode page table；
4. Q all-gather；
5. FlashInfer output+LSE；
6. LSE AG + output RS。



### 15.2 Kimi Linear：A2A + replicated Q

```bash
python -m sglang.launch_server \
  --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --trust-remote-code \
  --tp-size 4 \
  --dcp-size 4 \
  --attention-backend tokenspeed_mla \
  --kv-cache-dtype fp8_e4m3 \
  --dcp-comm-backend a2a \
  --dcp-replicate-q-proj \
  --cuda-graph-backend-prefill disabled
```

它适合对比：

```text
gathered Q + A2A
replicated Q + A2A
CUDA graph captured decode
超过 graph max batch 的 eager decode
```



### 15.3 Qwen3.5：Triton MHA/GQA DCP

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --trust-remote-code \
  --tp-size 4 \
  --dcp-size 4 \
  --attention-backend triton \
  --context-length 1048576 \
  --disable-radix-cache
```

这个 recipe 主要用于学习 MHA/GQA 路径，不要照搬 MLA 的 `a2a/replicated-Q` 推理。

### 15.4 Kimi K3

Kimi K3 建议先只设置拓扑，让模型 override 选择 backend：

```bash
python -m sglang.launch_server \
  --model-path moonshotai/Kimi-K3 \
  --trust-remote-code \
  --tp-size 8 \
  --dcp-size 8
```

启动后确认 resolved 值：

```text
prefill backend: trtllm_mla（默认路径）
decode backend:  cutedsl_mla
dcp_replicate_q_proj: true
dcp_comm_backend: fi_a2a on MNNVL, otherwise a2a
```

大规模 DPA/EP 配置应从 [Kimi K3 cookbook](../docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx#L138) 的目标硬件 cell 出发。

---



## 16. 关键兼容性与限制清单



### 16.1 通用拓扑

- `dcp_size >= 1`；
- `tp_size % dcp_size == 0`；
- 仅 CUDA/HIP；
- DPA 下 DCP group 必须位于单个 attention replica 内；
- 当前启动检查没有完整覆盖更强的 DPA containment 条件；
- 跨节点 DCP 是否有收益取决于互联，数学可构组不等于性能合理。



### 16.2 Attention/backend

- MLA backend必须返回 rank-local LSE；
- LSE log base 必须正确；
- `trtllm_mla + DCP + speculative decode` 不支持；
- FlashMLA FP8 KV + DCP 不支持；
- Triton DCP extend 不支持 score_mod/sinks/custom masks/sliding window；
- Kimi K3 decode backend 只允许 CuteDSL/TokenSpeed MLA；
- `fi_a2a` 要求真实 MNNVL fabric，不只是 GPU compute capability 满足。



### 16.3 Cache/memory

- virtual capacity/page 必须按 DCP widened；
- draft KV 通常复制，不按 target DCP 分片；
- KDA/Mamba state 不按 DCP token owner 分片；
- HiCache 只支持 MLA L1/L2；
- L3、LMCache、HiSparse、unified memory pool 当前不支持；
- MXFP8 masked write 不支持。



### 16.4 CUDA graph

- decode CUDA graph 有 DCP metadata/address-stable buffer 路径；
- `fi_a2a` workspace 必须在 capture 前初始化；
- DCP 会禁用 prefill piecewise/breakable graph 的若干路径，因为 dummy extend capture 没有 DCP metadata；
- Kimi K3 DCP 当前关闭 symmetric memory 以规避 decode graph 正确性问题；
- eager 与 captured batch 都要分别验证。



### 16.5 Spec/PD

- speculative draft KV 复制；
- Kimi K3 DSPARK 当前要求 static verify；
- D worker 启用 DCP 时要求 P worker `attn_cp_size==1`，不支持 `P: Prefill CP>1 -> D: DCP>1`；
- PD 仅支持相同 DCP 或 DCP1→DCP relayout；
- 相同 DCP 要求 P/D DCP rank 匹配；DCP1→DCPN 只支持 MLA/hybrid-MLA；
- relayout 在 sender 端选择 token rows并直接写入 D local pool，不会把 full KV 发给 D 后再切；
- relayout transfer 只接入 Mooncake/NIXL；
- PD DCP 要求匹配 physical page/KV dtype 和特定 decode cache 模式。

---



## 17. 性能分析方法



### 17.1 一个更完整的简化模型

MLA 每层、每 decode step 可粗略写成：

```text
T_dcp ≈ T_q_projection
      + T_q_allgather_or_redundant_gemm
      + T_local_attention(B, L/C, H_group, D)
      + T_output_lse_exchange(B, H_group, V, C)
      + T_local_lse_combine
```

显存可粗略写成：

```text
M_per_rank ≈ M_weights
           + M_target_kv(L)/C
           + M_unsharded_state
           + M_draft_kv(L)
           + M_graph_and_comm_workspace
```

不要只观察 `max_total_num_tokens`；它不能说明：

- state pool 是否先耗尽；
- max-running-requests 是否被 request slots 限制；
- A2A workspace 是否抵消 KV 节省；
- draft KV 是否成为主项。



### 17.2 建议测试矩阵

固定模型、dtype 和硬件，至少测试：

```text
context length: 1K, 4K, 16K, 64K, 128K, ...
batch size:     1, 低并发, graph max, graph max+1, 目标线上并发
DCP degree:     1, 2, 4, 8
comm backend:   ag_rs, a2a, fi_a2a（硬件允许时）
Q mode:         gathered, replicated
prefix mode:    no hit, partial hit, large radix hit
deployment:     unified, PD decode
spec:           off, target recipe
```

记录：

- `max_total_num_tokens`；
- effective max running requests；
- 单请求最大可服务 context；
- decode tokens/s；
- ITL/TPOT p50/p90/p99；
- 单卡 KV pool bytes、graph pool、comm workspace；
- attention kernel 时间；
- Q AG、LSE AG、RS/A2A 时间；
- extend/prefix gather 时间；
- 各 rank local KV length 是否均衡。



### 17.3 如何判断瓶颈


| 现象                        | 可能原因                                                     |
| ------------------------- | -------------------------------------------------------- |
| DCP 后容量接近扩大 C 倍，但速度变慢     | context 尚短，固定 collective 占主导                             |
| 长 context 加速、短 context 退化 | 正常 break-even 行为                                         |
| `ag_rs` 慢、`a2a` 明显改善      | collective 次数/FP32 RS 成为瓶颈                               |
| replicated Q 反而慢          | Q GEMM/权重带宽代价大于 Q AG                                     |
| DCP8 比 DCP4 容量高但吞吐低       | A2A/同步或跨节点边界成本过大                                         |
| max token 提高但并发不提高        | request slot、KDA/Mamba state 或 draft KV 限制               |
| decode 正确、prefix hit 错    | virtual/physical index、page widening 或 extend reorder 问题 |
| eager 正确、CUDA graph 错     | address-stable metadata/buffer、capture 前 workspace 初始化问题 |
| 只在非整除长度错误                 | local length、padding、partial final page 处理错误             |
| 输出无 NaN 但 logits 偏差大      | LSE base-e/base-2 或跨 rank softmax 权重错误                   |


---



## 18. 推荐源码阅读顺序



### 第一阶段：概念、参数和拓扑

1. [官方 DCP 文档](../docs/docs/advanced_features/dcp.mdx#L8)
2. [server_args.py：DCP 参数](../python/sglang/srt/server_args.py#L1010)
3. [server_args.py：通用校验](../python/sglang/srt/server_args.py#L3921)
4. [parallel_state.py：DCP group](../python/sglang/srt/distributed/parallel_state.py#L2366)
5. [runtime_context.py：live parallel accessors](../python/sglang/srt/runtime_context.py#L236)

读完应能回答：DCP 是否新增 world-size 维度、哪些 ranks 在一个 group、DPA containment 为什么比 `TP%DCP` 更强。

### 第二阶段：布局和 allocator

1. [layout.py：owner/local length](../python/sglang/srt/layers/dcp/layout.py#L23)
2. [kv_cache_configurator.py：widened allocator](../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1707)
3. [kv_cache_builder.py：TreeCache page](../python/sglang/srt/mem_cache/kv_cache_builder.py#L212)
4. [memory_pool.py：MLA write filtering](../python/sglang/srt/mem_cache/memory_pool.py#L4045)
5. [triton_backend.py：MHA masked write](../python/sglang/srt/layers/attention/triton_backend.py#L1255)

建议手算 `DCP=4, physical_page=2, virtual positions=0..15` 的 owner 和 physical index。

### 第三阶段：MLA decode 主链

1. [planner.py：local decode metadata](../python/sglang/srt/layers/dcp/planner.py#L136)
2. [forward_mla.py：Q gather/replication 分支](../python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L621)
3. [flashinfer_mla_backend.py：local output+LSE](../python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L721)
4. [forward_mla.py：comm backend dispatch](../python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L753)
5. [comm.py：AG/RS LSE merge](../python/sglang/srt/layers/dcp/comm.py#L112)
6. [comm.py：A2A merge](../python/sglang/srt/layers/dcp/comm.py#L452)

读完应能从公式解释：为什么只传 output 不够、为什么必须有 LSE、为什么最后还能恢复 TP-local heads。

### 第四阶段：Extend 与 prefix cache

1. [metadata.py](../python/sglang/srt/layers/dcp/metadata.py#L23)
2. [prepare_decode_context_parallel_metadata](../python/sglang/srt/layers/dcp/planner.py#L32)
3. [all_gather_kv_cache_for_dcp](../python/sglang/srt/layers/dcp/comm.py#L276)
4. [forward_mla.py：extend prefix gather](../python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L621)
5. [triton_backend.py：DCP extend](../python/sglang/srt/layers/attention/triton_backend.py#L1568)



### 第五阶段：模型和复杂组合

1. [Qwen3.5 KV-head/DCP layout](../python/sglang/srt/models/qwen3_5.py#L853)
2. [Kimi K3 overrides](../python/sglang/srt/arg_groups/overrides.py#L395)
3. [PD DCP connection validation](../python/sglang/srt/disaggregation/common/conn.py#L285)
4. [PD relayout](../python/sglang/srt/disaggregation/utils.py#L994)
5. [HiCache DCP host translation](../python/sglang/srt/mem_cache/pool_host/base.py#L335)
6. registered DCP tests：[test/registered/dcp](../test/registered/dcp)

---



## 19. 调试清单



### 19.1 启动时先确认

```text
tp_size / pp_size / dp_size / ep_size
effective attn_tp_size / attn_dp_size / attn_cp_size
dcp_size / dcp group ranks
prefill attention backend
decode attention backend
dcp_comm_backend
dcp_replicate_q_proj 的 resolved 值
KV cache dtype / pool class / page size
max_total_num_tokens
decode CUDA graph 是否 capture
fi_a2a workspace/fabric probe 是否成功
模型 override 改写了哪些参数
```



### 19.2 Forward 时建议打印/断点

```text
forward_mode
positions / out_cache_loc
dcp_rank / dcp_size
global seq_lens
local dcp kv lens
global/local kv_indptr
virtual kv indices -> local physical indices
dcp_kv_mask true count
Q gather 前后 shape
local attention output/LSE shape
LSE base-e or base-2
merge 后 output shape
extend prefix local/gathered length
```

关键位置：

```text
get_dcp_lens
plan_dcp_decode_metadata
prepare_decode_context_parallel_metadata
all_gather_q_for_mla_decode
attention_backend.forward_decode
cp_lse_ag_out_rs_mla / dcp_a2a_lse_reduce
token_to_kv_pool.set_*kv_buffer
```



### 19.3 正确性问题的缩小顺序

1. `DCP1` 与 `DCP2`，单请求、短整除长度，逐 token 比 logits；
2. 关闭 radix cache、spec、HiCache、PD；
3. eager decode，关闭 CUDA graph；
4. 只测 decode，无 prefix hit；
5. 加入非整除长度；
6. 加入 multi-request 不同长度；
7. 加入 radix prefix hit；
8. 对比 `ag_rs` 与 `a2a`；
9. 对比 gathered Q 与 replicated Q；
10. 恢复 graph、spec、PD/HiCache。

特别检查：

- local lengths 总和是否等于 global length；
- owner filter 后是否恰好每个 virtual position 只有一个 rank 写入；
- partial final page 是否没有 stale rows；
- global output 与单卡/非 DCP reference 的 logprob 差异；
- FlashInfer/FlashMLA LSE base 是否匹配 correction kernel。



### 19.4 可直接利用的测试

- 纯布局单测：[test_dcp_layout_unit.py](../test/registered/dcp/test_dcp_layout_unit.py#L1)
- LSE combine kernel：[test_dcp_lse_combine.py](../test/registered/kernels/test_dcp_lse_combine.py#L1)
- reduce-scatter：[test_reduce_scatter_along_dim.py](../test/registered/dcp/test_reduce_scatter_along_dim.py#L1)
- DeepSeek V3.1 DCP/non-DCP parity：[test_dsv31_dcp8_gsm8k.py](../test/registered/dcp/test_dsv31_dcp8_gsm8k.py#L1)
- Qwen3.5 Triton accuracy：[test_qwen3p5_triton_dcp.py](../test/registered/dcp/test_qwen3p5_triton_dcp.py#L1)
- HiCache host index math：[test_hicache_dcp_host_pool.py](../test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py#L1)

---



## 20. 常见问题



### Q1：DCP 会把所有 KV cache 都缩小 `dcp_size` 倍吗？

不会。它只缩小接入 DCP owner/index translation 的 target KV。Spec draft KV 通常复制；KDA/Mamba state 不分片；部分 MHA/GQA 原本已有 KV-head TP，净收益取决于 head layout。

### Q2：DCP 会加速 prefill 吗？

它的目标不是 prefill。Extend 可能需要 gather 已分片 prefix，长 prefix 下甚至增加通信。要优化 TTFT，应研究 Prefill CP、chunked prefill 或 PD prefill worker。

### Q3：DCP degree 越大越好吗？

不是。容量通常继续增长，但 local attention work 下降后，Q distribution、output/LSE collective 和同步占比会上升。最佳 degree 与 context、batch 和互联相关。

### Q4：为什么采用 round-robin token layout？

在线 decode 每步只增加少量 token。Round-robin 能让 ranks 的 local KV 长度始终最多差 1，并让一个 widened virtual page 均匀映射到每个 rank 的一个物理 page。

### Q5：为什么 partial attention output 不能直接 all-reduce？

因为每个 shard 内已做局部 softmax，归一化分母不同。必须先用 local LSE 计算其在全局 softmax 中的权重，再对修正后的 output 求和。

### Q6：`ag_rs` 名称下 MHA 为什么代码是 all-reduce？

MLA `ag_rs` 主路径使用 output reduce-scatter。Triton MHA helper 当前是 LSE all-gather、output all-reduce，再 slice local heads。两者语义相同但具体 collective 实现不同；`--dcp-comm-backend` 主要控制 MLA dispatch。

### Q7：DCP 与 TP 是相乘关系吗？

算法上它们是 head 与 context 两个不同的分片轴；但当前 SGLang 的启动参数不是两个独立 world size 相乘。DCP group 使用 TP group 内已有 ranks，且要求 `TP % DCP == 0`。

```text
--tp-size 8 --dcp-size 4
=> 8 张卡，可理解为 effective head-group 2 × context shard 4

--tp-size 2 --dcp-size 4
=> 非法，不表示 8 张卡
```

同理，`TP8,DCP8` 仍是 8 张卡，不是 64 张卡。

### Q8：什么时候一定需要 DCP？

当目标 context 的 shardable KV 在普通布局下无法装入，且模型/backend 支持时，DCP 是容量方案之一。若 KV 能装下，则是否启用取决于长 context decode 的带宽收益能否覆盖 collective。

### Q9：DCP 能提高 Kimi K3 并发吗？

它能提高 MLA KV token capacity；但 KDA state pool 不分片。如果并发先被 KDA request state 限制，DCP 不会继续提高并发，需要单独调整/评估 state pool。

### Q10：为什么 DCP 下 spec 的容量收益不再接近 C 倍？

因为 target KV 是 `target/C`，draft KV 仍是完整复制项。当 `C` 增大后，draft KV 可能成为每 token 的主要成本。

### Q11：能不能只设置 `--dcp-size` 给任意模型？

不能。必须同时有模型 Q/KV layout、pool masked write、backend local LSE、merge、extend/cache/graph 等完整接入。当前应从 registered test 或模型 cookbook recipe 出发。

### Q12：DCP 与 HiCache 能一起用吗？

目前只支持 MLA L1/L2 host tier。L3、LMCache、HiSparse、spec 和 PD decode 等组合仍不支持。

### Q13：PD 能不能让 P 开 Prefill CP、D 开 DCP？

当前不能。D worker 的 `dcp_size>1` 会要求 P worker `attn_cp_size==1`。目前支持 P DCP1→D DCPN 的 sender-side token relayout，或者 P/D 使用相同 DCPN 并按 matching DCP rank 直传；两条路径都直接落到 D local pool。

---



## 21. 一页复习图

```text
CLI
  --tp-size T
  --dcp-size C
  --dcp-comm-backend ag_rs|a2a|fi_a2a
  [--dcp-replicate-q-proj]
          │
          ▼
ServerArgs validation + model overrides
          │
          ├─ C divides TP
          ├─ CUDA/HIP
          ├─ create DCP groups inside TP
          ├─ choose attention/comm backend
          └─ allocate widened virtual KV pages
          │
          ▼
Shared virtual token space
  owner(v)    = v % C
  physical(v) = v // C
  virtual page = physical page * C
          │
          ▼
Decode metadata planning
  global seq_lens / req_to_token
          │ owner filter + index translation
          ▼
  rank-local kv_lens / indptr / page table
          │
          ▼
Per MLA attention layer
  local TP Q heads
          │
          ├─ Q all-gather
          │      or
          └─ replicated Q projection
          │
          ▼
  full DCP-group Q heads
          │
          ▼
  attention on rank-local KV shard
          │
          ├─ partial output o_r
          └─ local LSE lse_r
          │
          ▼
  cross-rank exact softmax merge
    global_lse = logsumexp(lse_r)
    output = Σ exp(lse_r-global_lse) * o_r
          │
          ├─ ag_rs: LSE AG + output RS
          ├─ a2a: packed NCCL A2A + local combine
          └─ fi_a2a: MNNVL A2A + local combine
          │
          ▼
  restore normal TP-local head output
          │
          ▼
New KV write
  only owner rank stores virtual_position // C
```

一句话记忆：

> SGLang DCP 用 TP group 内的 ranks 按 token position 分片长期 KV context，让每 rank 只读约 `1/C` 的历史 KV；它通过 Q 分发和带 LSE 的精确 partial-softmax 合并恢复普通 TP 输出，以固定的每层通信换取长上下文 KV 容量与带宽收益。

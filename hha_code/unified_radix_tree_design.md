# UnifiedRadixTree：SWA、Mamba/SSM 与 DSA 缓存统一设计梳理

> 本文结合当前仓库代码，以及 [#30468](https://github.com/sgl-project/sglang/pull/30468)、[#30636](https://github.com/sgl-project/sglang/pull/30636)、[#30626](https://github.com/sgl-project/sglang/pull/30626)、[#31643](https://github.com/sgl-project/sglang/pull/31643) 梳理设计意图。重点不是逐文件解释，而是说明这套设计为什么出现、核心抽象是什么，以及 ReplaySSM、int8 checkpoint 和 Mamba LRU 为什么都能自然接入。

## 1. 一句话结论

UnifiedRadixTree 的关键并不是发明一种新的 radix tree，而是把系统拆成：

- **一棵只表达 token 前缀关系的逻辑树**；
- **挂在树节点上的多种 cache component**，分别管理 Full KV、SWA KV 和 Mamba/SSM checkpoint；
- **一个统一 controller**，负责 GPU/CPU/Storage 之间的实际内存分配、释放和搬运。

它要解决的本质问题是：**同一段 token 前缀，对不同类型的状态有不同的可复用边界和生命周期，但它们仍然必须以同一个逻辑前缀为一致性锚点。**

因此，它不是要求 Full、SWA、Mamba 使用完全相同的缓存策略；恰恰相反，它允许它们共享树结构，但独立定义命中、锁定、LRU、淘汰、COW 和 HiCache 搬运语义。

## 2. 为什么需要统一树

旧设计针对不同模型维护专用实现，例如普通 `RadixCache`、`SWARadixCache`、`MambaRadixCache`。这些实现面对的逻辑前缀其实相同，但物理状态不同：


| 状态                | 真正被复用的内容                      | 命中时需要什么                  |
| ----------------- | ----------------------------- | ------------------------ |
| Full Attention KV | 从 root 到命中点的整条 KV path        | 整段前缀连续可用                 |
| SWA KV            | 命中点之前的最后一个 sliding window     | 最近窗口连续可用，更老的 KV 可以不存在    |
| Mamba/SSM         | 某个边界上的完整 recurrent state      | 只需要命中边界的一个 checkpoint    |
| DSA               | 仍以 token 对应的 KV/indexer 数据为主体 | 尤其在 HiCache 中需要统一的多级缓存控制 |


一旦继续为每种组合写专用树，复杂度会呈组合式增长：SWA/Mamba × HiCache × speculative decoding × ReplaySSM × int8 checkpoint × session cache × PD/PP/CP。类似的 prefix walk、节点 split、锁、LRU、淘汰、host backup/load-back 逻辑也会被反复实现，很容易出现某项新能力只同步到旧树、没有同步到新树的问题。

所以 #30468 把 hybrid SWA 和 hybrid SSM/Mamba 默认切到 `UnifiedRadixCache`。当前 `registry.py` 中：

- hybrid SWA（纯 SWA 特例除外）默认创建统一树；
- hybrid SSM/Mamba 默认创建统一树；
- DSA 在开启 hierarchical cache 时默认通过统一树接入 HiCache；
- DSA 当前没有独立的 `DSAComponent`，主要仍使用 `FULL` component，只是底层 KV pool 可以是 DSA 特有布局。

最后一点很容易被 PR 标题掩盖：**“DSA 默认使用 UnifiedRadixTree”在当前代码里主要指 DSA + HiCache 的路由，不代表 DSA 被建模成了第四种 tree component。**

## 3. 核心架构：逻辑树与物理资源解耦

当前实现可以概括为：

```text
请求 token IDs
     │
     ▼
UnifiedTreeCore
  - 只管理 radix 拓扑、prefix walk、node split
  - 根据所有 component 的 validator 选择一致的命中边界
  - 决定锁、LRU、级联淘汰等树级动作
     │
     ├── FullComponent   ：整条 path 的 Full/MLA/DSA KV
     ├── SWAComponent    ：最近 sliding window 的 KV
     └── MambaComponent  ：边界点上的 recurrent checkpoint
     │
     ▼
UnifiedRadixCache controller
  - 执行 allocator free/alloc
  - 执行 GPU ↔ CPU ↔ Storage 搬运
  - 执行 Mamba COW、int8 checkpoint store/load
```

`UnifiedTreeNode` 不再硬编码 `value`、`swa_value`、`mamba_value` 等字段，而是有统一的：

```python
node.component_data[ComponentType.FULL]
node.component_data[ComponentType.SWA]
node.component_data[ComponentType.MAMBA]
```

每份 `ComponentData` 都有独立的：

- `value`：device pool 中的 slot/index；
- `host_value`：HiCache host pool 中的 slot/index；
- `lock_ref` / `host_lock_ref`：device/host 资源保护；
- `metadata`：例如 SWA 的窗口边界 UUID；
- component 自己的热度与容量统计；SWA/Mamba 使用独立 LRU，Full 使用 access time 和 leaf set 驱动淘汰。

这里节点存的主要是 **pool index/所有权句柄**，真正的大 tensor 在各自的 memory pool 中。这样树只负责判断“哪个逻辑前缀拥有哪些资源”，不直接耦合某种 tensor layout。

当前代码又进一步把 `UnifiedTreeCore` 与 `UnifiedRadixCache` controller 分开，并使用 deferred `CacheAction`/`ComponentAction`：**tree 决定要做什么，controller 在树边界之外执行真实 I/O 和 allocator 操作。** 这降低了树算法与 CUDA stream、host transfer、具体 pool 实现之间的耦合，也为替换 TreeCore 后端留下空间。

## 4. 一次 prefix hit 是怎样达成“多状态一致”的

统一树最重要的语义不是“找到 token 最长公共前缀”，而是：

> 找到 token 能匹配，并且所有启用 component 都认为状态可复用的最深边界。

prefix walk 时，每个 component 创建自己的 validator：

- `FullComponent`：该节点必须存在 device value；HiCache 模式下 host value 也可构成 host match；
- `SWAComponent`：从候选命中点向前必须能凑出一段连续的 sliding window；窗口之外的 tombstone 不影响命中；
- `MambaComponent`：候选边界必须存在一个完整的 Mamba state checkpoint，device 或 host 上有一份即可。

TreeCore 只有在所有 validator 都通过时才更新 `best_match_node`。开启 HiCache 后，它还同时记录：

- 最深的 device-resident match，调度器可以立刻复用；
- 最深的 device-or-host match，后续可以 load-back/prefetch。

这就是统一设计的核心好处：**各状态保留不同的有效性规则，但最终复用边界由统一树做交集，避免“Full KV 认为命中了，Mamba state 却对应不上 token 长度”的错误。**

### 不同 component 的锁也不同

命中以后，三种状态真正被使用的范围不同，所以不能套用同一种锁：

- Full：锁定从命中节点到 root 的整条 path；
- SWA：只锁定最后一个 window；
- Mamba：只锁定命中边界的单个 checkpoint。

这也是 component 化的重要价值：共享 prefix tree，不等于共享资源生命周期。

## 5. 节点 split 与 tombstone：为什么物理状态可以比逻辑树“稀疏”

radix tree 的一个 edge 可能覆盖多个 token。若请求只匹配 edge 的一部分，TreeCore 会 split 节点。不同 component 对 split 的处理不同：

- Full KV 可以按 token 切分，父子节点各持对应切片；
- SWA KV 也可以按窗口边界重新分配；
- Mamba state 是某个位置之后的整体 recurrent state，不能把一个 checkpoint 切成前后两半，因此 split 后 checkpoint 留在原 child，新产生的 parent 对 Mamba 是 tombstone。

类似地，内部节点的某个 component 可以被单独淘汰，节点结构和其他 component 继续存在。`value is None` 表示该 component 在这里是 tombstone，而不是整棵树必须删除这个节点。

这使得各类资源能按自身压力独立淘汰：

- Mamba pool 紧张时，可以先丢某些中间 checkpoint，保留 Full/SWA；
- SWA pool 紧张时，可以丢窗口数据，但逻辑 token path 仍可保留；
- 只有 leaf 被删除时，所有 component 才需要一起收缩，因为逻辑节点本身不复存在。

内部节点的淘汰优先级为 `Full > SWA > Mamba`。淘汰高优先级 component 时会级联清掉同节点上更低优先级的数据，保证不会留下逻辑上不可达或无法复用的孤立状态；而淘汰 Mamba checkpoint 不必连带淘汰仍有价值的 Full/SWA 数据。

## 6. ReplaySSM 如何与统一树对齐

ReplaySSM 的出发点是减少 recurrent decode 每一步对巨大 state matrix 的读写带宽。它把某个 active slot 的状态表达成：

```text
当前逻辑状态 = 已提交的 checkpoint S0 + ring 中尚未 fold 的最近若干步更新
```

非 flush step 只记录较小的更新信息，周期性 flush/fold 后才把 ring 合并回 `temporal[slot]`。因此有一个非常重要的不变量：

> radix tree 中 key 的 token 长度，必须与它保存的 checkpoint 所代表的序列长度严格一致。

在 `no_buffer` 路径上，请求结束时 `temporal[slot]` 可能比 live state 落后 `write_pos` 步。如果直接把当前完整 token key 和这个旧 checkpoint 一起插入树，未来命中会从错误状态继续计算。

[#30636](https://github.com/sgl-project/sglang/pull/30636) 在 `MambaComponent.prepare_for_caching_req()` 中补齐了这个同步：

```python
cache_len = token_ids_len - replayssm_write_pos[slot]
replayssm_write_pos[slot] = 0
```

也就是说，只把 key 截到最后一个已经 flush 的边界，使 checkpoint 和 prefix 长度重新一致。cache hit 做 COW 时，source checkpoint 必须是 fully-flushed state；复制到新的 request-private active slot 后，目标 slot 的 ReplaySSM cursor 从 0 开始，表示 ring 为空。

这里 UnifiedRadixTree 不需要理解 ReplaySSM 的 recurrence 数学，它只需要让 `MambaComponent` 在“状态进入树”和“状态从树恢复”两个边界维护上述一致性协议。这正是 component hook 的设计价值。

### 普通 decode ReplaySSM 与 Spec-Verify ReplaySSM 不是一回事

上面的 `write_pos`/flush-boundary 讨论主要对应普通 decode ReplaySSM：它优化每个 decode step 读写完整 recurrent state 的 HBM 带宽。

另一个容易混淆的功能是 [#28695](https://github.com/sgl-project/sglang/pull/28695) 引入的 ReplaySSM Spec-Verify。它优化的是 speculative decoding 的 target verify 临时显存，核心是取消 per-draft full-state snapshot。两者共享 ReplaySSM 的“checkpoint + 小记录”思想，但服务于不同阶段，当前两个开关也互斥，因为它们共享 ring storage，却使用不同的更新协议。

## 7. ReplaySSM Ring Spec-Verify：为什么能把 11.5 GB 降到 1.8 GB

在 v0..5.16 中开始支持，应该非常好用，大幅减少 mtp 开启所带来的显存占用

> **先明确最关键的对象：这里 snapshot、ring 和 replay 的都是 target 模型的 GDN/SSM state，不是 draft 模型的 state。** Draft model 只负责提出候选 token；target model 用自己的 committed state 验证这些候选 token。因为 verify forward 执行时还不知道 sampler 最终会接受到哪个位置，传统实现才需要暂存 target model 在每个候选位置之后的完整 SSM state。ReplaySSM 优化掉的正是这些 target-side per-draft snapshots。

整个角色关系是：

```text
Draft model
  产生候选 token：t1 -> t2 -> t3 -> t4
                         |
                         v
Target model
  从自己的 committed SSM state S0 出发执行 target verify
  产生 target logits，并记录 target GDN 层的 raw update records
                         |
                         v
Sampler
  根据 target logits 决定 accept_len
                         |
                         v
Target model
  只 replay accepted records，得到自己新的 committed SSM state
```

即使 draft model 自身也包含 SSM/GDN 层，它也有独立的 draft state 生命周期，不是本节 `intermediate_ssm: 11.48 GB -> 0` 所指的对象。

### 7.1 传统 target verify 为什么要保存多份完整状态

假设 draft model 一次提出一条长度为 4 的候选链：

```text
t1 -> t2 -> t3 -> t4
```

target model 在一次 verify forward 中依次处理它们。对于 target 的 GDN recurrent layer，它自己的状态会按 token 更新：

```text
target S0 --t1--> target S1
target S1 --t2--> target S2
target S2 --t3--> target S3
target S3 --t4--> target S4
```

verify forward 执行时还不知道 sampler 最终接受几个 token。假设最后只接受 `t1、t2`，target 模型正确的 committed state 应该是 `target S2`。传统实现因此把 target 每一步的完整 state 都写进 `intermediate_ssm`：

```text
target intermediate_ssm = [target S1, target S2, target S3, target S4]
```

采样结束后再选择正确的中间状态写回 target model 的 persistent `temporal`。不能提交 `S4`，因为 rejected token 并未进入最终序列；也不能事先只保存某一个 `S_i`，因为 verify 时还不知道最终的 `accept_len`。

需要注意的是，这块显存通常不是每轮 verify forward 根据当前 batch 临时申请的。`MambaPool` 在服务初始化时，就按照最大 speculative state slot 容量预分配 `intermediate_ssm_state_cache`，使后续 verify 可以直接写入固定地址，也便于 CUDA Graph 使用。当前代码中的逻辑 shape 近似为：

```python
intermediate_ssm.shape = [
    num_linear_layers,
    spec_state_slots + 1,
    speculative_num_draft_tokens,
    num_heads_per_tp,
    K,
    V,
]
```

也就是说，每个可并发运行的 target state slot，都提前为每个 draft position、每个 target GDN layer 留好一份完整状态：

```text
                   draft position
              t1       t2       t3       t4
target layer 0  S1_0     S2_0     S3_0     S4_0
target layer 1  S1_1     S2_1     S3_1     S4_1
target layer 2  S1_2     S2_2     S3_2     S4_2
...             ...      ...      ...      ...
target layer N  S1_N     S2_N     S3_N     S4_N
```

上表只是一个 target request slot；真实 buffer 还要再乘最大并发 `spec_state_slots`。GDN 的单个 recurrent state 近似是一个 `[K, V]` 大矩阵，因此这块 scratch 的显存为：

```text
Memory ≈
num_linear_layers
× spec_state_slots
× speculative_num_draft_tokens
× num_heads_per_tp
× K × V
× dtype_size
```

这就是显存爆炸的来源。其中 draft token 数、最大并发 slot 数和 target GDN 层数都是线性放大项；最昂贵的是每个快照本身包含 `K×V` 大矩阵。比如 draft window 从 4 增加到 8，其他条件不变时，这部分预分配显存近似翻倍。

Qwen3.5-35B-A3B、TP1 的 PR 测试配置中，仅 `intermediate_ssm` 就占 11.48 GB。这个数字与模型维度、draft length、slot 数、TP 和 dtype 都有关，不是所有模型的固定开销。

开启 ReplaySSM Spec-Verify 后，初始化阶段直接令：

```python
intermediate_ssm_state_cache = None
```

因此后续 verify kernel 也不再写 per-draft full-state snapshots。需要区分的是，短卷积状态为了在 accept/reject 后恢复正确 window，仍保留较小的 `intermediate_conv_window`；它经过重叠窗口去重，且尺寸远小于 recurrent `K×V` state，不是 11.48 GB 的主要来源。

### 7.2 ReplaySSM 保存“更新日志”，不保存“每一步快照”

ReplaySSM 保留一份 target committed checkpoint `S0`，并为每个 draft token 记录 target GDN 层中足以重放 recurrence 的小输入：

```text
target R_i = (target raw_v_i, target raw_k_i, target g_i, target beta_i, ...)
```

直观上：

```text
旧方案：保存 target 起点之后每一步的整张照片 S1、S2、S3、S4
新方案：保存 target 起点照片 S0，以及 target 的四条操作日志 R1、R2、R3、R4
```

完整 state 的大小是 `O(K×V)`，而单步 raw record 主要是 `O(K+V)`。因此 scratch 从近似：

```text
O(draft_len × K × V)
```

降为：

```text
O(record_len × (K + V))
```

#28695 合入时，`intermediate_ssm` 从 11.48 GB 变为 0，六组 ring buffers 合计约 1.81 GB，即 speculative scratch 约缩小 6.4 倍。persistent `S0` 并没有消失；省掉的是“每个 draft token 一份完整 state”的临时副本。

### 7.3 不物化 S1...S4，verify output 怎么算

target verify 仍然需要每个位置的 output/logits，才能让 sampler 判断 accept/reject。ReplaySSM 从：

```text
S0 + ring history R1...Ri
```

直接重建第 `i` 步的 output，而不真正写出完整 `S_i`。GDN 路径使用 chunked delta-rule 的 `(I + A)^-1` 变换，一次处理线性 draft window 内的因果交互。

这个 output 是一次性的：生成 logits、交给 sampler，随后丢弃；它不会作为下一轮 persistent state。因此 output reconstruction 的微小数值误差不会沿生成长度累积。

### 7.4 sampler 决定接受长度后，只精确 replay accepted prefix

假设最终：

```text
accept_len = 2
```

commit 阶段只按照原 recurrent update 的顺序重放 `R1、R2`：

```text
S0 --exact replay(R1)--> S1 --exact replay(R2)--> S2
```

`R3、R4` 对应 rejected suffix，不参与 fold，直接被覆盖或丢弃。整体流程可以画成：

```text
                 output-only verify
S0 + R1 R2 R3 R4 --------------------> sampler
                                          |
                                          | accept_len = 2
                                          v
S0 -- exact replay(R1, R2) ------------> S2 (new committed checkpoint)
```

传统 rollback 是“从多份完整状态中选择 `S2`”；ReplaySSM rollback 是“只把 accepted records replay 到 `S0`”。

当前 GDN 实现采用 `fold-every-commit`：每次 accept length 确定后立即把 accepted prefix replay 进 `temporal`，使 `temporal` 始终表示最新 committed state。相比让 committed records 长期滞留在 circular ring 中，这更容易与 radix checkpoint、`extra_buffer` 和 state tracking 组合。

### 7.5 为什么 state 要 exact fold，而 output 可以用 chunked reconstruction

#28695 的早期版本曾直接把 chunked delta open-loop fold 进 checkpoint。短输出看起来正常，但误差会进入未来 state，长 reasoning 序列中不断积累，最终出现 accuracy 下降和 repetition loop。

最终设计将两条数值路径分开：

- **Output 路径**：一次性消费，使用 chunked reconstruction；误差不反馈到未来 state。
- **State 路径**：保存 raw `v/k/g/beta`，commit 时逐 token 执行 closed-loop exact fold；运算顺序刻意克隆 recurrent baseline。

代码中的 `gdn_replayssm_exact_fold_kernel` 保持相同的归一化、decay→delta→update 顺序、tile 和 reduction tree。配合 fp32 SSM checkpoint，目标是让 committed state 与 recurrent baseline bit-identical。因此当前打开 spec ReplaySSM 时会自动选择 `--mamba-ssm-dtype float32`；强制使用低精度会收到长序列 drift 风险警告。

所谓 accuracy parity 的准确理解是：

- committed state 通过 exact replay 保持一致；
- verify output 的误差只存在于一次性路径，通常低于最终 bf16 cast 的粒度；
- PR 的端到端准确率测试与 recurrent baseline 持平。



### 7.6 为什么只支持 linear draft chain

该算法要求：

```text
--speculative-eagle-topk in {None, 1}
```

即 draft 是唯一父子关系的线性链。ring 中第 `i` 条 record 明确依赖 `1...i-1`，可以用严格下三角 causal mask 表达。

若 `topk > 1`，EAGLE verify 是一棵多分支树：

```text
          t1
        /    \
      t2a    t2b
     /  \      \
   t3a  t3b    t3c
```

不同节点依赖不同 parent state，一个线性 ring/cursor 无法描述所有分支。若要支持，需要携带 parent/path 元数据或复制分支状态，会明显增加 kernel 和内存管理复杂度。

### 7.7 6.4× 更小不等于所有场景都快 6.4×

6.4× 指 speculative scratch memory，不是吞吐提升倍数。ReplaySSM 省掉大量 full-state snapshot write，但增加了 ring write、output reconstruction 和 accepted-prefix fold。

在 #28695 的 Qwen3.5-35B-A3B H20 对照测试中：

```text
ReplaySSM：1954.1 output tok/s
Recurrent：1926.4 output tok/s
```

两边基本处于 throughput parity。最稳定的收益是释放约 11.5 GB HBM，用于更大的 KV cache、更高并发或给其他 kernel 留出安全余量；不同模型、batch size 和硬件仍需单独 benchmark。

### 7.8 当前参数与最初 PR 已有变化

#28695 最初使用：

```bash
--enable-gdn-replayssm-spec
```

当前它只是 deprecated alias，推荐使用：

```bash
--enable-linear-replayssm-spec
```

因为当前 spec-verify 路径已经从 GDN 扩展到 GDN/KDA。它仍然默认关闭，并保持 linear-chain 限制。

`--linear-replayssm-cache-len` 的作用也发生了细化：

- 最初 GDN circular-ring 方案用它控制 ring length；
- 当前 GDN `fold-every-commit` 直接按最大 draft token 数分配 record window；
- KDA spec ring 仍使用该长度并要求能容纳 draft window；
- 普通 decode ReplaySSM 仍使用它控制周期性 flush 间隔。



### 7.9 它和 UnifiedRadixTree 的边界

Spec-Verify ring 是运行中请求的临时 scratch，不是 radix prefix cache：

```text
ReplaySSM verify ring
        |
        | exact-fold accepted prefix
        v
committed GDN/Mamba checkpoint
        |
        | track / cache unfinished / request finish
        v
UnifiedRadixTree.MambaComponent
```

UnifiedRadixTree 只应接收与 token 长度一致、能够独立恢复的 committed checkpoint。ReplaySSM 负责在 verify 内消除 snapshots 并完成 accept/reject；MambaComponent 负责 checkpoint 入树、COW、LRU 和多级缓存生命周期。两套设计通过 committed-state boundary 组合，而不是让 radix tree 直接管理 speculative ring。

## 8. 普通 decode 也可以开启 ReplaySSM

ReplaySSM 不只用于 speculative target verify。当前实现还提供了一条普通自回归 decode 路径：

```bash
--enable-linear-replayssm
--linear-replayssm-cache-len 16
```

它和 `--enable-linear-replayssm-spec` 共享“保存小记录、延迟物化完整状态”的核心思想，但解决的问题不同：

```text
普通 decode ReplaySSM：
减少每个 decode token 对完整 K×V SSM state 的 HBM 读写
→ 主要是显存带宽和吞吐优化

Spec-Verify ReplaySSM：
消除每个 draft token 的完整 target SSM snapshot
→ 主要是 speculative scratch 显存优化
```

普通 decode 原本每生成一个 token，都要读取、更新并写回一份完整的 recurrent state。ReplaySSM 改为保留一个完整 checkpoint，并把最近若干 token 的 `d/k/g` 等更新记录写入 ring：

```text
完整 checkpoint S0
  + record(t1)
  + record(t2)
  + record(t3)
  + record(t4)
        |
        | fold / flush
        v
新的完整 checkpoint S4
```

这里的 record 类似“状态更新日志”，但不是简单的 `S += delta`。kernel 仍要按照 GDN/Delta Rule recurrence 的衰减、外积和状态相关更新，把这些记录作用到 checkpoint 上。数学意义上的最新状态每一步都确定了，只是完整的 `K×V` 矩阵暂时没有写回 `temporal`；达到 fold 边界时才将日志压实成新的 checkpoint。

`--linear-replayssm-cache-len=L` 控制普通 decode 最多积累多少步再 fold。它是显存与效率之间的折中：

| `L` | ring 显存 | 完整 state 写回频率 | 每步 replay/ring 开销 |
| --- | --- | --- | --- |
| 较小 | 较小 | 较高 | 较小 |
| 较大 | 随 `L` 线性增加 | 较低 | 较大 |

GDN decode ring 的显存近似为：

```text
Memory ≈
num_linear_layers
× state_slots
× L
× per-token-record-size
```

单条 record 主要是 `O(K+V)`，没有完整 state 的 `O(K×V)` 那么大。增大 `L` 可以把昂贵的完整状态写回摊薄到更多 token 上，但 `L` 过大又会增加 output reconstruction 需要读取和处理的 pending records。因此它通常存在最佳平衡点，并非越大越好；当前默认值 `16` 可以作为 benchmark 起点，再对比 `8/16/32` 的吞吐、ITL 和显存。

这条路径当前主要适合 GDN。代码中的预期是 batch size 较大时缓解 state bandwidth bottleneck；KDA 虽然已接通并保证正确性，但由于 per-K gate ring 更大、每步重建更重，目前可能慢于 packed baseline，不建议为了性能开启。其他当前限制包括：

- 只优化 decode，不替换 prefill 路径；
- 需要 Triton linear-attention decode backend；
- 需要 `--mamba-radix-cache-strategy no_buffer`；
- 暂不支持 PD disaggregation；
- 已支持 Radix Cache，遇到 radix track boundary 会强制 fold，确保树接收的是完整 committed checkpoint；
- `--enable-linear-replayssm` 与 `--enable-linear-replayssm-spec` 当前互斥，因为两者共享 ring storage，但 cursor 和 commit 协议不同。

因此，普通 decode ReplaySSM 不会复现 spec-verify 中“11.48 GB 降到 1.81 GB”的巨大显存收益：普通 decode 本来就没有每个 draft token 一份完整 target state 的中间快照。它的核心价值是用小 ring 换取更少的完整 `K×V` state HBM 流量。

## 9. int8 Mamba checkpoint 如何接入

Mamba recurrent state 很大。如果 radix cache 长期把每个历史 checkpoint 都放在 active bf16/fp16 pool 中，会直接压缩可并发请求数和可缓存前缀数。

int8 方案将两类生命周期拆开：

```text
运行中的请求：active MambaPool，保持完整精度，供 kernel 持续更新
历史前缀缓存：MambaCheckpointPool，temporal state 用 int8 保存
```

状态进入树时，`MambaComponent`：

1. 从 active slot 读取完整状态；
2. 将 temporal state 做 per-(head, k-channel) 对称 int8 量化；
3. 很小的 conv window 保持原 dtype；
4. 树节点的 Mamba `value` 指向 int8 checkpoint pool 的 slot，而不是 active pool slot。

prefix hit 时：

1. 为新请求分配一个 full-precision active slot；
2. 将命中的 int8 checkpoint 解量化到 active slot；
3. 后续 recurrence 始终在完整精度下继续。

因此它不是每步 quant/dequant，也不会让状态反复经历量化循环；每个缓存点通常只在 store 时量化一次，在 hit 时解量化一次。收益是固定显存下大约翻倍的 cached-state 容量，同时也能减少相应 host-offload 占用。

[#30626](https://github.com/sgl-project/sglang/pull/30626) 的核心不是添加量化公式本身，而是把统一树的 **slot 所有权与 allocator 路由** 补完整：

- 创建 checkpoint 时向 int8 pool 分配；
- tree eviction 时归还给 int8 allocator，而不是 active mamba allocator；
- insert 发现已有相同 checkpoint 时释放刚创建的重复 slot；
- request finish 时释放 active slot，但保留已经被 tree 接管的 int8 slot；
- cache hit 的 deferred COW 在 forward stream 上选择 `load_to_active()`，避免走普通 full-precision `copy_from()`。

这说明统一树中 `component_data[MAMBA].value` 的真正抽象是“一个可恢复 Mamba checkpoint 的句柄”，而不是固定等同于 active pool index。

## 10. “cache hit 只 reset 用到的 state”到底是什么意思

这里的 `reset` 是 **把 LRU 位置刷新到 MRU**，不是把 Mamba tensor 清零。

一次 Mamba prefix hit 实际只消费命中边界的那个 state：

- 只有它被 COW 到 request-private slot；
- Mamba lock 也只锁这个节点；
- path 上的祖先 checkpoint 并没有参与这次恢复。

旧逻辑却调用 `reset_node_and_parents_mru()`，把整条 ancestor path 的 Mamba states 都标成“刚刚使用”。在多轮会话里，这会让同一 session 的一串中间 checkpoint 聚集在 LRU 热端；内存压力下，真正仍有复用价值的 leaf checkpoint 反而容易随一个冷 session 整串被淘汰。

[#31643](https://github.com/sgl-project/sglang/pull/31643) 将策略改为：

- Full KV 仍刷新整条匹配 path，因为整条 path 确实被复用；
- SWA 只刷新 window 范围；
- Mamba 只刷新 `best_match_node` 上实际被 COW 的 checkpoint；
- insert walk 也不再把已有的所有 Mamba ancestors 刷热，新生成的 checkpoint 才进入 MRU。

这个修改最能体现 UnifiedRadixTree 的设计哲学：**LRU 的“访问”不是 token tree 的访问，而是物理 component 的真实消费。** 当前统一实现把这一语义放在 `MambaComponent.refresh_lru()` 中，而不是让 TreeCore 对整条 path 做一刀切的刷新。

PR 给出的 128 并发多轮会话压力测试中，整体 cache hit rate 从 0.609 提升到 0.828，平均 TTFT 从 0.19s 降到 0.11s。这个收益来自更准确的缓存价值判断，不是 kernel 计算变快。

## 11. 这套设计带来的主要好处



### 11.1 功能演进从“同步多套树”变成“扩展一个 component”

ReplaySSM、int8 checkpoint、HiCache 等能力主要在 Mamba component 的边界协议中实现，prefix walk 和 radix split 不需要再复制一份。

### 11.2 一致性边界更清楚

token key、Full KV、SWA window 和 Mamba checkpoint 必须共同确认 match；ReplaySSM 的 flush 边界也在 checkpoint 入树前被校准。错误更容易被限制在某个 component 的 prepare/finalize hook 中。

### 11.3 资源利用更好

不同 pool 有独立的容量、锁与 LRU。Mamba 中间 checkpoint 可以先淘汰，SWA 只保留窗口，int8 checkpoint 与 active slot 分池，避免最稀缺的资源被统一粗粒度策略绑死。

### 11.4 HiCache 组合能力更强

同一棵逻辑树可以同时记录 device 和 host slot，并由 component 生成各自的 transfer spec。Full、SWA、Mamba 不必各自维护一套 L1/L2/L3 前缀拓扑。

### 11.5 更适合继续扩展

TreeCore 与 controller 已经解耦；未来更换 TreeCore 实现、添加新 cache component、引入新 checkpoint 表示形式，都不必重写整个调度与缓存栈。

## 12. 需要注意的边界与代价

- 统一树降低的是重复实现和组合复杂度，并没有消除状态机复杂度；component 之间的 validator、锁和级联淘汰仍需维护严格不变量。
- “最长 token match”不一定等于“最终可复用长度”，最后结果取所有 component 可用边界的交集。
- Mamba checkpoint 不能像 Full KV 一样任意 split，树上出现 Mamba tombstone 是正常设计，不代表缓存损坏。
- int8 checkpoint 用一次量化误差换容量；后续 recurrence 虽为完整精度，但不能理解成数学上的完全无损。
- DSA 当前主要复用 `FULL` component 和统一 HiCache controller，不应误解为已有通用 DSA component。
- ReplaySSM 的 ring/cursor 是 checkpoint 表示的一部分；任何 COW、donate、finish、offload/load-back 路径都必须明确它保存的是 fully-flushed snapshot，还是 snapshot + ring。



## 13. 建议 review 时抓住的代码入口

不需要逐文件看，按下面顺序即可验证本文的核心判断：

1. `[python/sglang/srt/mem_cache/registry.py](../python/sglang/srt/mem_cache/registry.py)`
  - 看默认选树规则，以及 `FULL/SWA/MAMBA` component 的组装。
2. `[python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py](../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py)`
  - 看 `match_prefix()` 如何汇总所有 validator，以及 node split、级联淘汰。
3. `[python/sglang/srt/mem_cache/unified_cache/components/tree_component.py](../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py)`
  - 看 component hook 契约和 `ComponentData`。
4. `[python/sglang/srt/mem_cache/unified_cache/components/swa_component.py](../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py)`
  - 看 window-bounded validator、LRU refresh 和 tombstone。
5. `[python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py](../python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py)`
  - 看 single-node match/COW、ReplaySSM finish 对齐、int8 checkpoint 生命周期和 single-state LRU refresh。
6. `[python/sglang/srt/mem_cache/mamba_checkpoint_pool.py](../python/sglang/srt/mem_cache/mamba_checkpoint_pool.py)`
  - 看 active state 与 cached int8 checkpoint 的分池设计。
7. `[python/sglang/srt/model_executor/model_runner.py](../python/sglang/srt/model_executor/model_runner.py)`
  - 看 deferred COW 如何在 forward stream 上选择普通 copy 或 int8 `load_to_active()`。
8. `[python/sglang/srt/layers/attention/linear/gdn_backend.py](../python/sglang/srt/layers/attention/linear/gdn_backend.py)`
  - 看 target verify 如何在 snapshot、circular ring 和 fold-every-commit 路径间分派。
9. `[python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py](../python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py)`
  - 看 accepted prefix 的 closed-loop exact fold。
10. `[python/sglang/srt/speculative/spec_utils.py](../python/sglang/srt/speculative/spec_utils.py)`
  - 看 sampler 得到 accept length 后如何提交 ReplaySSM state。



## 14. 最后总结

我认为这次设计最关键的变化可以概括成三句话：

1. **统一的是 token 前缀拓扑，不是不同 cache state 的物理语义。**
2. **每种 state 只为自己真正消费的资源负责：Full 看 path，SWA 看 window，Mamba 看一个 checkpoint。**
3. **ReplaySSM 和 int8 checkpoint 证明了 component 边界是有效的：只要维护好“prefix 长度 ↔ checkpoint 状态”的协议，底层状态表示可以独立演进。**

所以 UnifiedRadixTree 的长期价值不只是少维护几个旧类，而是给 hybrid model 的 cache state 建立了一个统一的控制平面：逻辑共享、物理隔离、边界协商、生命周期自治。

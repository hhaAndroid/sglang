# EAGLE Speculative Decoding 与 PD Disaggregation

> 基于当前 SGLang 源码的 spec v2 路径分析。重点配置示例：
> `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`

当前实现已经统一走 spec v2 worker。旧的 `eagle_worker.py` / `multi_layer_eagle_worker.py` 不是主路径，本文以这些文件为准：

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/speculative/spec_info.py` | speculative algorithm 分发、PD draft input 入口 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | EAGLE 主流程：draft、verify、draft-extend |
| `python/sglang/srt/speculative/base_spec_worker.py` | EAGLE/MTP 共用的 draft/draft-extend batch 准备 |
| `python/sglang/srt/speculative/eagle_info.py` | `EagleDraftInput`、`EagleDraftExtendInput`、`EagleVerifyInput` |
| `python/sglang/srt/speculative/eagle_info_v2.py` | spec v2 decode 前 KV 预留逻辑 |
| `python/sglang/srt/speculative/eagle_utils.py` | tree 构建、verify sampling、verify batch 准备 |
| `python/sglang/srt/speculative/eagle_disaggregation.py` | PD disaggregation 下组装 decode 侧 EAGLE draft input |
| `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | Multi-layer EAGLE / MTP 主流程 |
| `python/sglang/srt/models/llama_eagle.py` | LLaMA EAGLE draft 模型结构 |
| `python/sglang/srt/disaggregation/prefill.py` | PD prefill 侧保存 spec 元数据 |
| `python/sglang/srt/disaggregation/decode.py` | PD decode 侧恢复 spec 元数据 |
| `python/sglang/srt/disaggregation/utils.py` | PD metadata buffer 定义与拷贝 |

---

## 1. 参数含义

| 参数 | 示例值 | 含义 |
|------|--------|------|
| `speculative_algorithm` | `EAGLE` | 使用 EAGLE draft 模型做投机解码 |
| `speculative_num_steps` | `3` | draft 树最大深度，不含根节点 bonus token |
| `speculative_eagle_topk` | `1` | 每步保留的候选数。`topk=1` 时退化为线性链 |
| `speculative_num_draft_tokens` | `4` | target verify 一次处理的 token 数，等于 `1 + num_steps` |

`topk=1` 时，`speculative_num_draft_tokens` 会被自动校正为 `speculative_num_steps + 1`。例如 `num_steps=3` 时 verify 序列为 `[bonus, d1, d2, d3]`，共 4 个 token。

注意：如果外层系统使用 `--sglang-speculative-*` 这类参数名，那应理解为外层封装参数；SGLang 原生命令行参数是 `--speculative-*`。

---

## 2. 核心数据结构

### 2.1 `EagleDraftInput`

用于 decode 轮开始前的 draft 阶段：

```python
EagleDraftInput(
    topk_p=...,
    topk_index=...,
    hidden_states=...,
    bonus_tokens=...,
)
```

字段语义：
- `bonus_tokens`：上一轮 target 无条件接受的最后一个 token，也就是下一轮 verify tree 的根。
- `topk_p/topk_index`：上一轮 draft-extend 已经准备好的下一步 draft 候选。
- `hidden_states`：普通 EAGLE 中通常是 draft model 的最后 hidden state，不是 target hidden state。

### 2.2 `EagleDraftExtendInput`

用于 target prefill 或 target verify 之后填充 draft KV cache：

```python
EagleDraftExtendInput(
    hidden_states=target_hidden_states,
    num_correct_drafts=accept_lens - 1,
    num_accept_tokens=accept_lens,
)
```

这里的 `hidden_states` 是 target model 的 hidden state。EAGLE draft 模型需要用 token embedding 拼接 target hidden state 来更新自己的 KV cache。

### 2.3 `EagleVerifyInput`

用于 target verify：

```python
EagleVerifyInput(
    draft_token=[bonus, d1, d2, d3],
    custom_mask=tree_mask,
    positions=position,
    retrieve_index=...,
)
```

`EagleVerifyInput.draft_token_num` 是 verify token 总数，`max_tree_depth = spec_steps + 1`。`accept_lens` 返回时包含 bonus token，所以 `accept_lens = accepted_draft_tokens + 1`。

---

## 3. EAGLE Draft 模型结构

以 `python/sglang/srt/models/llama_eagle.py` 为例，EAGLE draft 模型不是完整 target LLM，而是较轻的 assistant 模型：

```python
hidden_states = self.embed_tokens(input_ids)
hidden_states = self.fc(
    torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)
)
for layer in self.layers:
    hidden_states, residual = layer(...)
return hidden_states + residual
```

关键点：
- draft 输入同时需要 `input_ids` 和 `spec_info.hidden_states`。
- prefill/draft-extend 阶段喂给 draft 的 hidden states 来自 target。
- decode draft loop 内，后续步会使用 draft 上一步输出的 hidden states 继续链式生成。

---

## 4. Prefill 阶段

普通 EAGLE prefill 的核心函数是 `EagleDraftWorker._draft_extend_for_prefill()`。

### 4.1 Target prefill

target model 先处理 prompt，产生：
- target hidden states
- next token，即第一轮的 `bonus_tokens`

### 4.2 Draft prefill / draft-extend

draft 模型需要为 prompt 建立自己的 KV cache。代码会把每个请求的 `input_ids` 做左移，并把最后一个位置替换成 target 生成的 next token：

```text
prompt:       [t0, t1, t2, ..., t(n-1)]
target next:  v0
draft input:  [t1, t2, ..., t(n-1), v0]
target h:     [h0, h1, ..., h(n-2), h(n-1)]
```

直觉是：target 在位置 `i` 的 hidden state 预测位置 `i+1` 的 token，所以 draft 使用 `embed(t(i+1)) + h(i)` 来学习预测更后面的 token。

当前代码还处理 chunked prefill 的尾 token：非最后 chunk 时，tail token 不是 target next token，而是下一段 prompt 的第一个 token，避免 chunked prefill 下 draft KV 链断开。

### 4.3 返回下一轮 draft input

prefill 的 draft-extend 完成后，会返回：

```python
EagleDraftInput(
    topk_p=topk_p,
    topk_index=topk_index,
    hidden_states=logits_output.hidden_states,
    bonus_tokens=next_token_ids,
)
```

注意这里返回的 `hidden_states` 是 draft forward 的输出 hidden states，用于下一轮 decode draft。

---

## 5. Decode 一轮的完整流程

当前 spec v2 下，一轮 decode 主要分成四步：

```text
1. draft()
   用上一轮准备好的 topk 和 hidden states 生成 draft tree

2. verify()
   target model 一次 forward 验证整棵 tree，得到 accept_lens 和 bonus token

3. _draft_extend_for_decode()
   用 target verify 的 hidden states 更新 draft KV cache，并准备下一轮 topk

4. scheduler result processor
   提交输出 token，更新 kv_committed_len 和统计信息
```

---

## 6. Draft 阶段：topk=1, num_steps=3

入口：`EagleDraftWorker.draft()` 和 `draft_forward()`。

假设进入本轮时：

```text
bonus_tokens:  [v0]
topk_index:    [d1]
topk_p:        [p(d1)]
hidden_states: [draft_h_at_v0]
draft KV:      已包含历史 committed token
```

`draft_forward()` 循环 `range(num_steps)`：

### step 0

`select_top_k_tokens()` 从上一轮 topk 中选出 `d1`。随后会执行一次 draft forward：

```text
input_ids = d1
spec_info.hidden_states = hidden_states_for_d1
draft forward -> logits -> topk_index = d2
draft hidden -> hidden_states
```

所以“d1 免费”只表示 d1 本身不用本轮 draft forward 生成；但 step 0 仍会跑 draft forward，用 d1 生成 d2。

### step 1

继续用上一步的 draft hidden：

```text
input_ids = d2
spec_info.hidden_states = draft_h_at_d1
draft forward -> logits -> topk_index = d3
draft hidden -> hidden_states
```

### step 2

收集 `d3` 作为树节点后直接 `break`，不再 forward。

因此 `num_steps=3` 时，本轮实际 draft forward 次数是 `num_steps - 1 = 2`。生成的 verify 序列为：

```text
[v0, d1, d2, d3]
```

---

## 7. Tree 构建

入口：`build_tree_kernel_efficient()`。

draft 阶段只产出 draft tokens `[d1, d2, d3]`。构建 verify tree 时，代码会在最前面拼上 `bonus_tokens`：

```python
draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
```

`topk=1` 时树退化成单链：

```text
v0 -> d1 -> d2 -> d3
```

对应的 verify attention 是普通因果链：

```text
v0: attend history
d1: attend history + v0
d2: attend history + v0 + d1
d3: attend history + v0 + d1 + d2
```

当前实现对 `topk=1` 有 fast path：`organize_draft_results()` 的排序和 gather 会退化成 identity，代码直接拼接每步 token，并复用预分配的 parent/index buffer。

---

## 8. Verify 阶段

入口：`EAGLEWorkerV2.verify()`。

target model 对 `[v0, d1, d2, d3]` 一次 forward，输出每个位置的 target logits。随后 `eagle_sample()` 决定接受长度。

### 8.1 Greedy 路径

当 `sampling_info.is_all_greedy`，或当前设备是 NPU/HIP 时，使用 greedy verify：

```text
target@v0 == d1 ? accept d1
target@d1 == d2 ? accept d2
target@d2 == d3 ? accept d3
```

最后还会追加一个 target 自己预测的 bonus token。因此：

| 情况 | 输出 token |
|------|------------|
| d1 被拒绝 | `[target@v0]` |
| d1 接受、d2 拒绝 | `[d1, target@d1]` |
| d1/d2 接受、d3 拒绝 | `[d1, d2, target@d2]` |
| d1/d2/d3 全接受 | `[d1, d2, d3, target@d3]` |

### 8.2 Sampling 路径

非 greedy 且 kernel 可用时，使用 `tree_speculative_sampling_target_only()`：

```python
target_probs = softmax(logits / temperature)
target_probs = top_k_renorm_prob(target_probs, top_k)
target_probs = top_p_renorm_prob(target_probs, top_p)
draft_probs = torch.zeros_like(target_probs)
```

这里是 target-only sampling 模式，不使用 draft model 的概率分布做精确 speculative sampling 校正。两个阈值来自 server args：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--speculative-accept-threshold-single` | `1.0` | 单点直接接受阈值 |
| `--speculative-accept-threshold-acc` | `1.0` | 接受概率缩放阈值 |

TP 多卡下，sampling 结果会从 rank 0 broadcast，避免不同 rank 因浮点差异产生不一致 token。

---

## 9. Verify 后更新 Draft KV

入口：`EagleDraftWorker._draft_extend_for_decode()`。

verify 得到：

```text
predict:      每个请求 verify tree 上的输出 token buffer
accept_lens:  每个请求接受 token 数，包含 bonus token
hidden_states: target verify 时每个 tree node 的 hidden states
```

然后 draft-extend 使用 `predict` 作为 input ids，用 target verify 的 hidden states 更新 draft KV cache：

```python
EagleDraftExtendInput(
    hidden_states=batch_result.logits_output.hidden_states,
    num_correct_drafts=batch_result.accept_lens - 1,
    num_accept_tokens=batch_result.accept_lens,
)
```

draft-extend 会对整块 `num_draft_tokens` 宽度运行，并在最后用：

```python
select_index = arange(bs) * speculative_num_draft_tokens + accept_lens - 1
```

选出每个请求最后接受位置对应的 logits/hidden states，作为下一轮 `EagleDraftInput.topk_*` 和 `hidden_states`。

---

## 10. KV Cache 管理

当前 spec v2 的 KV 管理不要按旧版 `backup_state=True` 理解。

### 10.1 decode 前预留

`EagleDraftInputV2Mixin.prepare_for_decode()` 会基于每个请求的：

```text
kv_committed_len
kv_allocated_len
get_alloc_reserve_per_decode()
```

预留足够的 KV 空间。它还会先把 `kv_committed_len += 1`，这 1 个位置是给 bonus token 预占的。

### 10.2 draft 写入位置

`EagleDraftWorkerBase.prepare_for_draft()` 根据 `req_to_token`、`seq_lens`、`topk`、`num_steps` 计算本轮 draft 写入位置：

```text
out_cache_loc shape = bs * topk * num_steps
```

`topk=1` 或 `page_size=1` 时走 contiguous 分配；`topk>1 且 page_size>1` 时走 page-aligned tree region，并复制 prefix tail 到各分支，保证按页读取正确。

`draft_forward()` 再把 `out_cache_loc` 变换成 per-step 布局：

```python
out_cache_loc.view(bs, topk, num_steps)
    .permute(2, 0, 1)
    .reshape(num_steps, -1)
```

### 10.3 verify 后提交

scheduler 处理结果时：

```python
req.kv_committed_len += accept_lens[i] - 1
```

因为 decode 前已经为 bonus 预占了 1 个位置，所以这里加的是 `accepted_draft_tokens = accept_lens - 1`。如果请求已经结束，会把预占的 bonus slot 减回去。

---

## 11. PD Disaggregation 下的 EAGLE

PD 模式下，prefill server 和 decode server 分离。EAGLE 需要除了 KV cache 以外，再把下一轮 draft 所需的元数据从 prefill 侧传到 decode 侧：

```text
output_id
output_topk_p
output_topk_index
hidden_states_tensor
```

### 11.1 prefill 侧保存

在 `python/sglang/srt/disaggregation/prefill.py` 中，prefill 完成后：

```python
req.output_ids.append(next_token_id)
if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
    req.output_topk_p = batch.spec_info.topk_p[i]
    req.output_topk_index = batch.spec_info.topk_index[i]
    req.hidden_states_tensor = batch.spec_info.hidden_states[i].cpu().clone()
```

这里的 `batch.spec_info` 是 prefill draft-extend 返回的 `EagleDraftInput`。因此传输给 decode 侧的是：
- 下一轮 draft 的 topk 候选
- 下一轮 draft 需要的 draft hidden state
- 当前输出 token，也就是下一轮 tree root / bonus token

### 11.2 metadata buffer 传输

`python/sglang/srt/disaggregation/utils.py` 里 `MetadataBuffers` 为 PD + spec decode 准备了：

```python
output_topk_p:     (size, 16), float32
output_topk_index: (size, 16), int64
output_hidden_states: (size, hidden_size)
```

当前代码里注释说明 `speculative_eagle_topk` 暂不应超过 16。multi-layer EAGLE 会传 `topk * num_steps` 个状态，也需要落在这个 buffer 宽度内。

### 11.3 decode 侧恢复

在 `python/sglang/srt/disaggregation/decode.py` 中，decode 侧收到 metadata 后：

```python
decode_req.req.output_topk_p = output_topk_p
decode_req.req.output_topk_index = output_topk_index
decode_req.req.hidden_states_tensor = output_hidden_states
```

这些字段先挂到 `Req` 上，后续再组装成真正的 `EagleDraftInput`。

### 11.4 组装 `EagleDraftInput`

入口是 `SpeculativeAlgorithm.build_disagg_draft_input()`，EAGLE family 会调用：

```python
build_eagle_disagg_draft_input(batch, server_args, last_tokens_tensor, future_map)
```

实现位于 `python/sglang/srt/speculative/eagle_disaggregation.py`：

```python
num_states = server_args.speculative_eagle_topk
if server_args.enable_multi_layer_eagle:
    num_states *= server_args.speculative_num_steps

topk_p = stack(req.output_topk_p[:num_states])
topk_index = stack(req.output_topk_index[:num_states])
hidden_states = stack(req.hidden_states_tensor).to(batch.device)

spec_info = EagleDraftInput(
    topk_p=topk_p,
    topk_index=topk_index,
    hidden_states=hidden_states,
    bonus_tokens=last_tokens_tensor,
)
spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
```

这里 `last_tokens_tensor` 就是 prefill 侧生成并传过来的第一个 output token。对 decode server 来说，它是本请求第一轮 speculative decode 的 tree root。

### 11.5 overlap 模式

如果 `batch.enable_overlap` 为真，`build_eagle_disagg_draft_input()` 还会：

```python
spec_info.future_indices = batch.req_pool_indices
future_map.publish(spec_info.future_indices, batch.seq_lens)
future_map.stash(spec_info.future_indices, spec_info)
```

作用是把 PD 恢复出的 `EagleDraftInput` 放进 overlap 调度的 future map。后续 worker 可以用 `req_pool_indices` 作为 key 取回对应 spec input，避免异步调度中 batch 重排导致状态错配。

---

## 12. Multi-layer EAGLE / MTP

当 `server_args.enable_multi_layer_eagle=True` 时，worker 切到 `MultiLayerEagleWorkerV2`。

### 12.1 draft 阶段

普通 EAGLE 的 `topk_p/topk_index` 形状是：

```text
(bs, topk)
```

multi-layer EAGLE / MTP 的 `topk_p/topk_index` 会保存多步结果：

```text
(bs, topk * num_steps)
```

因此 PD disaggregation 中 `num_states = topk * num_steps`。

### 12.2 draft-extend 阶段

MTP 的多步 draft 主要发生在 verify 之后的 `_draft_extend_for_decode()`，而不是下一轮 `draft()` 中临时逐步生成。代码会循环：

```python
for step in range(speculative_num_steps):
    draft_runner_list[step].forward(...)
    ret_topk_p, ret_topk_index = fast_topk(...)
    rotate_input_ids_triton(...)
```

最后把所有 step 的 topk 拼起来，写入下一轮 `EagleDraftInput`：

```python
next_draft_input.topk_p = torch.cat(ret_topk_p_list, dim=1)
next_draft_input.topk_index = torch.cat(ret_topk_index_list, dim=1)
next_draft_input.hidden_states = None
```

所以更准确的说法是：MTP 把多步 draft 成本搬到了 verify 后的 draft-extend 阶段；下一轮 `draft()` 主要负责把已经准备好的多步 topk 组织成 verify tree。

---

## 13. topk=1, steps=3 的一轮示例

假设 prefill 后得到：

```text
bonus = v0
topk_index = d1
hidden = draft_h(v0)
```

一轮 decode：

```text
Draft:
  step0: select d1, forward(d1, draft_h(v0)) -> d2, draft_h(d1)
  step1: select d2, forward(d2, draft_h(d1)) -> d3, draft_h(d2)
  step2: select d3, break

Build tree:
  [v0, d1, d2, d3]

Verify:
  target forward once over [v0, d1, d2, d3]
  accept_lens in [1, 4]

Draft extend:
  use accepted target hidden states to update draft KV
  select last accepted position
  prepare next topk and draft hidden

Commit:
  append accepted output tokens
  kv_committed_len += accept_lens - 1
```

Case 表：

| draft 接受数 | `accept_lens` | 输出 token |
|--------------|---------------|------------|
| 0 | 1 | `target@v0` |
| 1 | 2 | `d1, target@d1` |
| 2 | 3 | `d1, d2, target@d2` |
| 3 | 4 | `d1, d2, d3, target@d3` |

---

## 14. 旧文档中最容易混淆的点

1. `EagleDraftInput.hidden_states` 不应笼统说成 target hidden states。decode draft 阶段它通常是 draft hidden states；draft-extend 阶段的 `EagleDraftExtendInput.hidden_states` 才是 target hidden states。
2. `d1` 是上一轮 topk 直接给出的，但 `step0` 仍然会跑 draft forward，用 `d1` 产生 `d2`。
3. 当前主实现是 spec v2，没有旧版 `backup_state=True` 那套 KV 回滚描述。应按预留、写入、提交 `kv_committed_len` 来理解。
4. PD disaggregation 不只传 KV cache，还必须传 `topk_p/topk_index/hidden_states_tensor`，否则 decode 侧无法构造第一轮 `EagleDraftInput`。
5. MTP 不是“完全零成本 draft”，而是把多步 topk 准备放在 verify 后 draft-extend 阶段。

# SGLang Speculative Decoding: Spec V1 与 Spec V2 实现对比

> 这份文档按 `v0.5.10` 的源码解释为什么当时要区分 Spec V1 / Spec V2；再补充当前 main 上 V2 的演进。重点以 EAGLE 为例，因为 `v0.5.10` 里 EAGLE 同时保留了 `eagle_worker.py` 和 `eagle_worker_v2.py`。

---

## 1. 版本时间线

在 `v0.5.10` 中，Spec V1 和 Spec V2 是并存的：

```text
SGLANG_ENABLE_SPEC_V2=False
  -> disable_overlap_schedule=True
  -> create_worker() 返回 eagle_worker.py / standalone_worker.py
  -> Spec V1

SGLANG_ENABLE_SPEC_V2=True
  -> disable_overlap_schedule=False
  -> create_worker() 返回 eagle_worker_v2.py / standalone_worker_v2.py
  -> Spec V2 + overlap scheduler
```

关键源码：

- `v0.5.10:python/sglang/srt/server_args.py`
  - `_handle_speculative_decoding()` 中检查 `envs.SGLANG_ENABLE_SPEC_V2.get()`。
  - 开启时设置 `disable_overlap_schedule=False`。
  - 关闭或不支持时设置 `disable_overlap_schedule=True`。
  - 当时 Spec V2 只支持 `topk=1`，`topk>1` 会报错。

- `v0.5.10:python/sglang/srt/speculative/spec_info.py`
  - `create_worker()` 根据 `enable_overlap = not server_args.disable_overlap_schedule` 选择 worker。
  - EAGLE:
    - overlap on -> `EAGLEWorkerV2`
    - overlap off -> `EAGLEWorker`

当前 main 上已经不同：

- `SGLANG_ENABLE_SPEC_V2` 已废弃。
- speculative decoding 总是使用 V2 worker。
- `--disable-overlap-schedule` 只决定 V2 worker 是 overlap 驱动还是同步驱动。
- 旧 V1 worker path 已在 `c0480a88be [Spec] Retire Spec V1 (#27964)` 之后移除。

---

## 2. Spec V1 的核心思路

Spec V1 是“worker 内部同步闭环”：

```text
Scheduler
  -> EAGLEWorker.forward_batch_generation(batch)
       if prefill:
         target prefill
         draft prefill
         return
       else:
         draft
         target verify
         draft_extend_after_decode
         return
  -> Scheduler process result
```

这里 scheduler 把 `ScheduleBatch` 直接交给 speculative worker。worker 在函数内部会大量修改 `batch` 状态，例如：

- `batch.forward_mode`
- `batch.spec_info`
- `batch.seq_lens`
- `batch.out_cache_loc`
- `batch.return_hidden_states`

所以 `v0.5.10:eagle_worker.py` 的注释明确说：`forward_batch_generation()` 会修改很多 batch state，最终 batch 不保证和输入状态一致。

---

## 3. Spec V1 的 prefill 流程

入口：`v0.5.10:python/sglang/srt/speculative/eagle_worker.py`

```python
def forward_batch_generation(self, batch):
    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        logits_output, next_token_ids, seq_lens_cpu, can_run_cuda_graph = (
            self.forward_target_extend(batch)
        )
        self.forward_draft_extend(
            batch,
            logits_output.hidden_states,
            next_token_ids,
            seq_lens_cpu,
            logits_output.mm_input_embeds,
        )
        return GenerationBatchResult(...)
```

### 3.1 target prefill

`forward_target_extend()`：

```python
model_worker_batch = batch.get_model_worker_batch()
model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
```

target prefill 必须捕获 FULL hidden states，因为 draft prefill 要用 target 每个 prompt 位置的 hidden state 建立 draft KV cache。

### 3.2 draft prefill

`forward_draft_extend()`：

```python
batch.spec_info = EagleDraftInput(
    hidden_states=hidden_states,
    verified_id=next_token_ids,
)
batch.spec_info.prepare_for_extend(batch)
...
logits_output = self.draft_model_runner.forward(forward_batch).logits_output
self.capture_for_decode(logits_output, forward_batch.spec_info)
```

这里的 `verified_id` 是 target prefill 采样出的第一个 token。`prepare_for_extend()` 会做 shift：

```text
prompt:      [t0, t1, ..., t(n-1)]
verified_id: v0
draft input: [t1, ..., t(n-1), v0]
hidden:      [h0, ..., h(n-1)]
```

最后 `capture_for_decode()` 把 draft prefill 输出转成下一轮 decode 的初始状态：

```python
draft_input.topk_p, draft_input.topk_index = fast_topk(...)
draft_input.hidden_states = logits_output.hidden_states
```

---

## 4. Spec V1 的 decode 流程

Spec V1 decode 在同一个 `forward_batch_generation()` 里顺序执行：

```python
spec_info = self.draft(batch)
logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
    self.verify(batch, spec_info)
)
self.forward_draft_extend_after_decode(batch)
return GenerationBatchResult(...)
```

### 4.1 draft 阶段

入口：`EAGLEWorker.draft()`。

先做 `_draft_preprocess_decode()`：

```python
out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
    batch.tree_cache,
    num_seqs * alloc_len_per_decode,
    backup_state=True,
)
...
assign_draft_cache_locs(...)
...
self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)
```

这就是 V1 的一个重要特点：draft token 的 KV 写入位置是临时分配出来的，并且用 allocator backup/restore 把 allocator 状态回滚。也就是说，draft KV 会真实写到某些 slot，但 allocator 不把这些 slot 当作最终 committed allocation。后面只有被接受的 token 才会被搬运或保留到 target KV 视角。

之后 `draft_forward()` 按 `num_steps` 循环：

```python
for i in range(self.speculative_num_steps):
    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(...)

    if i == self.speculative_num_steps - 1:
        break

    forward_batch.input_ids = input_ids
    forward_batch.out_cache_loc = out_cache_loc[i]
    spec_info.hidden_states = hidden_states
    logits_output = self.draft_model_runner.forward(...)
    topk_p, topk_index = fast_topk(...)
    hidden_states = logits_output.hidden_states
```

`topk=1, steps=3` 时：

```text
step 0: 选择上一轮 topk 的 d1，forward(d1) -> d2
step 1: forward(d2) -> d3
step 2: 收集 d3，break
```

然后 `build_tree_kernel_efficient()` 把 `verified_id` prepend 到 draft tokens 前面，得到 verify tree。

### 4.2 verify 阶段

入口：`EAGLEWorker.verify()`。

```python
spec_info.prepare_for_verify(batch, self.page_size)
batch.forward_mode = ForwardMode.TARGET_VERIFY
batch.spec_info = spec_info
model_worker_batch = batch.get_model_worker_batch(...)
batch_result = self.target_worker.forward_batch_generation(
    model_worker_batch,
    is_verify=True,
)
```

target 一次 forward verify tree。随后：

```python
spec_info.hidden_states = logits_output.hidden_states
res = spec_info.verify(
    batch,
    logits_output,
    self.token_to_kv_pool_allocator,
    self.page_size,
    vocab_mask,
)
```

`res` 是 `EagleVerifyOutput`，里面包含：

- `verified_id`
- `accept_length_per_req_cpu`
- `accepted_indices`
- `draft_input`

V1 verify 完成后会直接把 `batch.spec_info` 更新成下一轮 draft input：

```python
batch.forward_mode = ForwardMode.DECODE
batch.spec_info = res.draft_input
```

### 4.3 draft_extend_after_decode

入口：`forward_draft_extend_after_decode()`。

这个阶段用 target verify 捕获的 hidden states 更新 draft KV cache，并准备下一轮 topk。

V1 的写法很“就地修改 batch”：

```python
seq_lens_backup = batch.seq_lens.clone()
seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
...
batch.spec_info.prepare_extend_after_decode(batch, self.speculative_num_steps)
batch.forward_mode = ForwardMode.DRAFT_EXTEND
...
logits_output = self.draft_model_runner.forward(...)
self.capture_for_decode(logits_output, forward_batch.spec_info)
...
batch.forward_mode = ForwardMode.DECODE
batch.seq_lens = seq_lens_backup
batch.seq_lens_cpu = seq_lens_cpu_backup
...
```

也就是说 V1 为了跑 draft_extend 会临时改变 batch 的 seq_lens、forward_mode 等字段，跑完后再恢复。这种模式能工作，但对 overlap、异步调度、batch 生命周期管理都比较不友好。

---

## 5. Spec V1 的优缺点

优点：

- 实现直接：一个 worker 内同步完成 draft、verify、draft_extend。
- 状态传递简单：`batch.spec_info` 在 worker 内直接改。
- 对早期 topk>1 tree drafting 支持较完整。

问题：

- 调度粒度粗。scheduler 只能看到整个 speculative forward，不能自然 overlap draft/verify/draft_extend。
- batch 被 worker 大量 in-place 修改，需要备份和恢复状态。
- KV 管理依赖临时分配和 allocator restore，语义不够显式。
- result processor 很难统一处理普通 decode 和 speculative decode。
- PD disaggregation / overlap / adaptive spec 等特性要接入时，状态边界不清晰。

---

## 6. Spec V2 的核心目标

Spec V2 不是换了 EAGLE 算法，而是重构执行框架：

```text
目标 1: worker 接口改成 ModelWorkerBatch / ForwardBatch，便于 scheduler 管理
目标 2: speculative 状态显式化成 next_draft_input
目标 3: 用 FutureMap 让下一轮 batch 可以引用上一轮尚未完全 CPU resolve 的结果
目标 4: 支持 overlap scheduler，让 schedule stream 和 forward stream 更少互相等待
目标 5: KV 管理从临时回滚变成预留 + commit
```

在 `v0.5.10` 中，Spec V2 还处在早期阶段：

- 只支持 EAGLE/EAGLE3/STANDALONE。
- 只支持 `topk=1`。
- 通过 `SGLANG_ENABLE_SPEC_V2=True` 开启。

当前 main 中，V2 已经成为默认路径，并支持更多算法和更多 topk/page 组合。

---

## 7. Spec V2 的 scheduler 入口

`v0.5.10:python/sglang/srt/managers/scheduler.py` 中：

```python
self.enable_overlap = not server_args.disable_overlap_schedule
```

运行 batch 时：

```python
if self.spec_algorithm.is_none() or self.enable_overlap:
    worker_batch_or_batch = batch.get_model_worker_batch()
else:
    # speculative decoding v1
    worker_batch_or_batch = batch
```

这行非常关键：V1 把 `ScheduleBatch` 直接传给 worker；V2/overlap 则把 batch 转成 `ModelWorkerBatch`，让 worker 的 forward 接口更接近普通 model worker。

overlap 路径：

```python
future_indices = self.future_map.alloc_future_indices(bs)

with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)
    self.future_map.resolve_future(model_worker_batch)
    batch_result = self.model_worker.forward_batch_generation(model_worker_batch)

    if batch_result.delay_sample_func is None:
        self.future_map.store_to_map(future_indices, batch_result)
        batch_result.copy_to_cpu(...)
    else:
        batch_result.future_indices = future_indices

if batch.is_spec_v2:
    batch.spec_info = batch_result.next_draft_input
    batch.spec_info.future_indices = future_indices
    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
```

直观理解：

- scheduler 先给本轮结果分配一段 future slot。
- 下一轮 batch 可以先拿到 `future_indices`，不一定要等所有 CPU resolve 完成。
- 真正 forward 前，`future_map.resolve_future()` 把 future slot 中的 token/spec 状态填回 `model_worker_batch`。

---

## 8. Spec V2 的 FutureMap

`v0.5.10:python/sglang/srt/managers/overlap_utils.py` 中，FutureMap 对普通 decode 和 speculative decode 分开处理。

普通 decode 只需要保存 token id：

```python
token_ids_buf
```

Spec V2 需要保存下一轮 draft input：

```python
topk_p_buf
topk_index_buf
verified_id_buf
new_seq_lens_buf
hidden_states_buf
```

存储：

```python
draft_input = batch_result.next_draft_input
self.topk_p_buf[intv] = draft_input.topk_p
self.topk_index_buf[intv] = draft_input.topk_index
self.verified_id_buf[intv] = draft_input.verified_id
self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
self.hidden_states_buf[intv] = draft_input.hidden_states
```

恢复：

```python
indices = draft_input.future_indices.indices
draft_input.topk_p = self.topk_p_buf[indices]
draft_input.topk_index = self.topk_index_buf[indices]
draft_input.verified_id = self.verified_id_buf[indices]
draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
draft_input.hidden_states = self.hidden_states_buf[indices]
```

这就是 V2 overlap 的核心：batch 中先保存 future index，真正 forward 前再把上一轮结果 materialize 出来。

---

## 9. Spec V2 的 prefill 流程

入口：`v0.5.10:python/sglang/srt/speculative/eagle_worker_v2.py`

```python
def forward_batch_generation(self, model_worker_batch):
    if model_worker_batch.forward_mode.is_extend() or model_worker_batch.is_extend_in_batch:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(model_worker_batch)

        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
            model_worker_batch,
            batch_output.logits_output.hidden_states,
            batch_output.next_token_ids,
            batch_output.logits_output.mm_input_embeds,
        )
        return batch_output
```

和 V1 算法逻辑相同：

```text
target prefill -> target hidden + next token
draft prefill  -> draft KV + next_draft_input
```

但返回值不同。V2 会把下一轮 draft 所需状态显式放入：

```python
batch_output.next_draft_input
```

而不是只在 worker 内部原地更新 `ScheduleBatch`。

---

## 10. Spec V2 的 decode 流程

`EAGLEWorkerV2.forward_batch_generation()`：

```python
if model_worker_batch.spec_info is None:
    model_worker_batch.spec_info = EagleDraftInput.create_idle_input(...)

verify_input = self.draft_worker.draft(model_worker_batch)
model_worker_batch.spec_info = verify_input
batch_output = self.verify(model_worker_batch)
self.draft_worker._draft_extend_for_decode(model_worker_batch, batch_output)
return batch_output
```

阶段仍然是：

```text
draft -> verify -> draft_extend
```

但它返回的 `GenerationBatchResult` 带有：

```python
next_token_ids=predict
next_draft_input=next_draft_input
accept_lens=accept_length
```

V2 把“下一轮 draft 输入”作为显式结果交还 scheduler，而不是让 scheduler 从被修改过的 batch 中猜。

---

## 11. Spec V2 的 KV 管理

`v0.5.10:python/sglang/srt/speculative/eagle_info_v2.py` 中，`EagleDraftInputV2Mixin.prepare_for_decode()` 做了 over-allocation：

```python
x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
r.kv_allocated_len += x
r.decode_batch_idx += 1
...
assign_req_to_token_pool_func(...)
```

和 V1 的差异：

```text
V1:
  draft 时临时 alloc_token_slots(..., backup_state=True)
  assign_draft_cache_locs(...)
  restore allocator state
  verify 后再处理接受 token

V2:
  decode 前按 req 的 kv_committed_len / kv_allocated_len 预留未来可能用到的 slot
  req_to_token 直接记录预留区域
  draft/verify/draft_extend 都在这个预留区域内工作
  result processor 根据 accept_lens 更新 committed 长度
```

V2 这种方式更适合 overlap，因为 scheduler 可以在请求级别维护已提交和已预留的 KV 边界。

当前 main 中这部分又进一步演进：

- `prepare_for_decode()` 使用 `get_alloc_reserve_per_decode()`。
- EAGLE 会提前 pre-claim bonus slot：`kv_committed_len += 1`。
- result processor 用 `accept_lens - 1` settle committed length。
- page_size > 1 + topk > 1 的 draft tree 预留也被纳入 V2。

---

## 12. Spec V2 的 verify

`v0.5.10:EAGLEWorkerV2.verify()`：

```python
verify_forward_batch, can_run_cuda_graph = (
    verify_input.prepare_for_v2_verify(
        self.req_to_token_pool,
        batch,
        self.target_worker,
    )
)

forward_batch_output = self.target_worker.forward_batch_generation(
    model_worker_batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
    skip_attn_backend_init=True,
)

predict, accept_length, accept_index = verify_input.sample(...)
new_seq_lens = batch.seq_lens + accept_length
```

然后构造下一轮的壳：

```python
next_draft_input = EagleDraftInput(
    verified_id=verified_id,
    new_seq_lens=new_seq_lens,
    verify_done=verify_done,
)
```

注意：这时 `next_draft_input` 还没有完整的 `topk_p/topk_index/hidden_states`。这些要由紧随其后的 `_draft_extend_for_decode()` 填进去。

---

## 13. Spec V2 的 draft_extend

`_draft_extend_for_decode()` 的作用是：

```text
用 target verify 产生的 hidden states 更新 draft KV cache
从最后接受位置选出下一轮 topk_p / topk_index / hidden_states
填入 batch_result.next_draft_input
```

这一步完成后，scheduler 才能把完整的 `next_draft_input` 写入 FutureMap。

这也是 V2 的关键边界：

```text
worker 内部负责生成完整 next_draft_input
scheduler 负责把 next_draft_input 放入 future_map 或同步塞回 batch
```

---

## 14. V1 与 V2 的核心差异

| 维度 | Spec V1 (`v0.5.10`) | Spec V2 (`v0.5.10`) |
|------|----------------------|----------------------|
| 开启方式 | 默认 speculative 路径 | `SGLANG_ENABLE_SPEC_V2=True` |
| worker | `eagle_worker.py` | `eagle_worker_v2.py` |
| scheduler 传参 | 直接传 `ScheduleBatch` | 传 `ModelWorkerBatch` |
| batch 状态 | worker 内大量 in-place 修改 | 通过 `GenerationBatchResult.next_draft_input` 显式返回 |
| overlap | 基本没有 | 支持 overlap scheduler |
| 状态 relay | `batch.spec_info` 原地更新 | `FutureMap` 保存/恢复 spec 状态 |
| KV 管理 | 临时分配 + allocator backup/restore | 请求级 over-allocation + committed/allocated 边界 |
| topk 支持 | topk>1 tree 路径较成熟 | `v0.5.10` 只支持 topk=1 |
| 复杂度位置 | worker 内复杂 | scheduler/FutureMap/worker 协作复杂 |

---

## 15. 为什么要从 V1 演进到 V2

一句话：V1 能跑，但不适合高性能调度扩展。

更具体地说：

1. **overlap 需要明确的跨轮状态边界**

   V1 把状态藏在 `batch.spec_info` 和 worker 内部修改中。V2 把下一轮状态整理成 `next_draft_input`，scheduler 可以把它写入 FutureMap。

2. **KV cache 需要请求级 bookkeeping**

   V1 的 backup/restore 更像“临时借 slot”。V2 把 `kv_committed_len` / `kv_allocated_len` 当作请求生命周期的一部分，更适合 preemption、retract、PD disaggregation、KV offload。

3. **scheduler 需要统一普通 decode 和 spec decode**

   V1 是特殊分支：spec v1 直接用 `ScheduleBatch`。V2 尽量回到 `ModelWorkerBatch` / `ForwardBatch` 的统一 forward 模型。

4. **后续算法需要统一接口**

   DFLASH、NGRAM spec v2、Frozen-KV MTP、multi-layer EAGLE 都更适合挂在 V2 的阶段化接口上。

---

## 16. 当前 main 上的 V2 相比 v0.5.10 的变化

当前 main 已经不是 `v0.5.10` 那个“实验 V2”：

| 项 | `v0.5.10` | 当前 main |
|----|-----------|-----------|
| 开关 | `SGLANG_ENABLE_SPEC_V2=True` | 环境变量移除，默认 V2 |
| non-overlap | 回退 V1 worker | 仍用 V2 worker，同步驱动 |
| EAGLE topk>1 | V2 不支持 | V2 已支持更多 topk/page 组合 |
| 算法 | EAGLE/EAGLE3/STANDALONE 主要试点 | EAGLE、EAGLE3、STANDALONE、NGRAM、DFLASH、Frozen-KV MTP 等 |
| FutureMap | 以 `verified_id/new_seq_lens` 为核心 | 语义更通用，EAGLE 使用 `bonus_tokens/accept_lens/next_draft_input` 等更清晰字段 |
| V1 worker | 存在 | 已移除 |

因此：

```text
看 v0.5.10:
  必须区分 V1 / V2。

看当前 main:
  主要区分 overlap V2 / non-overlap V2。
```

---

## 17. 阅读源码建议

如果你基于 `v0.5.10` 学：

1. 先看 `server_args.py::_handle_speculative_decoding()`，理解 `SGLANG_ENABLE_SPEC_V2` 如何决定 `disable_overlap_schedule`。
2. 再看 `spec_info.py::SpeculativeAlgorithm.create_worker()`，确认 worker 选择。
3. 读 `eagle_worker.py::forward_batch_generation()`，这是 Spec V1 的主线。
4. 读 `eagle_worker_v2.py::forward_batch_generation()`，这是 Spec V2 worker 主线。
5. 读 `scheduler.py::run_batch()` 中 `enable_overlap` 分支，理解 FutureMap 怎么接入。
6. 最后读 `overlap_utils.py::FutureMap`，理解跨轮状态如何保存和恢复。

如果你基于当前 main 学：

1. 直接从 `spec_info.py::create_worker()` 看所有算法都返回 V2 worker。
2. 看 `arg_groups/speculative_hook.py` 里 `SGLANG_ENABLE_SPEC_V2 has been removed` 的 warning。
3. 看 `eagle_worker_v2.py`、`base_spec_worker.py`、`eagle_info_v2.py`。
4. 对照 scheduler 的 overlap 和 non-overlap V2 分支。

---

## 18. 一个简化心智模型

Spec V1：

```text
worker 是一个大黑盒：
  输入 batch
  内部修改 batch
  内部完成 draft/verify/draft_extend
  输出 tokens
```

Spec V2：

```text
worker 是阶段化执行器：
  输入 ModelWorkerBatch + spec_info
  输出 GenerationBatchResult + next_draft_input
  scheduler/FutureMap 管理跨轮状态
```

这就是为什么 Spec V2 更复杂，但也更容易支持 overlap、PD、KV offload、adaptive spec 和更多 speculative algorithm。

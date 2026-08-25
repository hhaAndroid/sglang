# SGLang Pause Generation 三种模式说明

这份文档总结 SGLang `pause_generation` 的三种模式：`abort`、`retract`
和 `in_place`，重点说明它们和 black-box agent partial rollout 的关系。

## 问题背景

这次讨论主要围绕以下问题：

- 听说 SGLang 有一个 `retrect` 模式，主要用于 partial rollout，它在哪里？
- 它和 `abort` 接口是什么关系？
- 在 black-box agent 场景下，如果想打断请求做 partial rollout，`retract`
  是否正好满足需求？
- 在 black-box 场景里，我们并不是真的打断 agent 运行过程，而是希望：
  连接不要断开，暂停期间不返回流式数据，恢复后继续生成。
- 如果发送暂停时 agent 正在执行工具，这个工具执行无法被打断是合理的；
  但一旦工具执行完成并开始发送 LLM 请求，这个请求应该在 SGLang 侧停下来。
- `pause_generation` 是否支持基于 request id 暂停？如果不支持，为什么？
- `in_place` 模式和 `retract` 有什么区别？
- `in_place` 模式能不能用于 RL 权重更新？
- `in_place` 的好处是否是少了一次 prefill，而 `retract` 虽然连接不断，
  但恢复时需要额外 prefill/recompute？

代码里的实际名称是 `retract`，不是 `retrect`。

## API 形式

`pause_generation` 是 engine 级别的控制接口：

```bash
curl -X POST http://host:port/pause_generation \
  -H 'Content-Type: application/json' \
  -d '{"mode":"retract"}'

curl -X POST http://host:port/continue_generation \
  -H 'Content-Type: application/json' \
  -d '{}'
```

当前请求结构只支持：

```python
mode: Literal["abort", "retract", "in_place"] = "abort"
```

`PauseGenerationReqInput` 里没有 `rid`、`request_id` 或 selector，所以这个接口
不是 request-id 级别的暂停接口。

## 三种模式总览

| 模式 | 请求是否保留 | 连接是否可保持 | 是否保留 KV cache | 是否可 flush KV cache | 恢复后是否需要重新 prefill/recompute | 典型用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `abort` | 否 | 否，会返回 abort/finish | 否 | 是 | 否 | 终止并丢弃当前请求 |
| `retract` | 是 | 是，前提是外部 timeout 不断开 | 否 | 是 | 是 | partial rollout、RL 权重更新、全局暂停 barrier |
| `in_place` | 是 | 是，前提是外部 timeout 不断开 | 是 | 否 | 否 | 短暂停顿，不做权重/cache reset |

## `abort`

`abort` 是终止模式。

通过 `pause_generation(mode="abort")` 使用时，它的语义基本等价于调用
abort endpoint 并设置 `abort_all=True`：当前 waiting queue 和 running queue
里的请求都会被 abort，客户端会收到 abort/finish 结果。

这个模式不会保留 partial rollout 的中间状态。它适合“取消请求”，不适合
“暂停后继续”。

## `retract`

`retract` 是最符合 partial rollout 语义的模式。

暂停发生时：

1. SGLang 在 tokenizer manager 侧设置全局 pause flag，新的 generation 请求会
   卡在入口，不会进入 scheduler。
2. 已经在 running batch 里的请求会被 retract 回 waiting queue。
3. 已经生成的 `output_ids` 会保留。
4. KV cache 可以释放，也可以在后续权重更新时 flush。
5. `continue_generation` 后，SGLang 会基于 `origin_input_ids + output_ids`
   重新构建 KV，然后继续 decode。

对于 black-box agent partial rollout，这正好匹配下面的需求：

- 请求不被 abort。
- streaming 连接可以保持。
- 暂停期间 SGLang 不继续返回新的 streaming token。
- 如果 agent 此刻正在执行工具，SGLang 无法打断工具执行，这是合理的。
- 但工具执行完成后，如果 agent 再发送 LLM 请求，这个请求会被 SGLang 的
  pause gate 挡住，直到恢复。

`retract` 的主要代价是恢复后需要额外 recompute。因为 KV cache 被释放了，
恢复时必须基于 `origin_input_ids + output_ids` 重新 prefill/recompute KV。

## `in_place`

`in_place` 是更轻量的暂停模式。

暂停发生时：

1. SGLang 暂停 inference scheduling。
2. 已经在 running batch 里的请求保持原地不动。
3. 这些请求的 KV cache 继续保留。
4. `continue_generation` 后，直接基于原有 KV cache 继续 decode。

它的核心好处就是避免 `retract` 恢复时需要的额外 prefill/recompute。

代价是：

- KV cache 一直占用显存。
- 不能 flush KV cache。
- 如果中间做了权重更新，会出现旧 KV cache 搭配新 weights 继续 decode 的情况。

如果 black-box agent rollout 只是为了同步暂停，不做权重更新，也不需要释放 KV，
那么 `in_place` 可能比 `retract` 更划算。

## RL 权重更新里的区别

`in_place` 和 `retract` 在 SGLang 的 RL 权重更新测试里都出现过，但语义不同。

更推荐的 RL partial rollout 权重更新路径是：

```text
pause_generation(mode="retract")
update_weights(..., flush_cache=True)
continue_generation()
```

这个路径更干净：KV cache 会被 flush，恢复后请求在新权重下重新计算 KV，然后继续
生成。

`in_place` 也可以这样使用：

```text
pause_generation(mode="in_place")
update_weights(..., flush_cache=False)
continue_generation()
```

它可以跑，但会保留旧 KV cache。也就是说，已经跑了一半的请求，前半段 KV 是旧权重
算出来的，后续 decode 用的是新权重。这避免了 recompute，但会带来 mixed-policy
或 mixed-cache 语义。如果 RL rollout 要求严格一致，一般不推荐这样做。

## 为什么不支持按 request id 暂停

`pause_generation` 现在设计成全局 generation barrier，主要服务于 RL rollout 控制
和权重更新。

它不是 request-id 级别暂停，主要原因是：

1. 权重更新前需要 SGLang 进入“没有活跃 inference 任务”的状态。只暂停某个 id
   不能保证这个全局条件。
2. KV cache 和调度都是 engine 级共享资源。部分请求暂停、部分请求继续，会让 cache
   生命周期和调度策略复杂很多。
3. 当前实现是对整个 running batch 做 retract，不是对单个 request id 做筛选。
4. 新请求也必须一起挡住，否则刚 retract 完又有新 generation 进来，无法形成稳定的
   rollout barrier。

对于“全部 agent 都暂停”的需求，这种全局设计是合适的。如果未来要支持只暂停部分
agent 或部分 request id，需要新增 request-level pause/retract API 和对应的 scheduler
策略。

## 选型建议

使用 `abort`：

```text
要终止请求，丢弃 partial work。
```

使用 `retract`：

```text
请求需要保留；
暂停期间希望释放/flush KV cache；
暂停期间要做 RL 训练和权重更新；
恢复后希望 KV 和新权重语义一致。
```

使用 `in_place`：

```text
只是短暂停一下；
不做权重更新；
不需要释放 KV cache；
希望恢复时避免额外 prefill/recompute。
```

对于 black-box agent partial rollout，如果是全部 agent 暂停：

```text
暂停期间不做权重更新：
  优先考虑 in_place，前提是显存压力可以接受。

暂停期间要做权重更新：
  优先考虑 retract + flush_cache=True。

需要硬取消：
  使用 abort。
```

最后有一个工程注意点：SGLang 自己会让请求保持 pending，但外层 HTTP client、
gateway 或 load balancer 仍然可能因为 streaming 连接空闲太久而断开。因此需要相应
调大 idle timeout，或者在外层做 keepalive。

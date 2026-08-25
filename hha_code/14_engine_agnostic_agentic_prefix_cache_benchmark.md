# 14. Agentic Prefix Cache 模拟测试实施方案

> 目标：不运行真实 agent，通过模拟多轮会话，测试开启/关闭 prefix cache 以及不同缓存配置下，cache hit rate 对 TTFT、E2E 和吞吐的影响。测试程序与 SGLang、vLLM 等具体推理引擎解耦。

## 1. 模拟目的

Agent 请求的主要特征是：下一轮会把上一轮完整历史重新作为 prefix 发送，只增加少量新的工具返回或 observation。

模拟器需要复现下面的数据形态：

```text
Turn 1:
  system prompt + initial task

Turn 2:
  Turn 1 input + Turn 1 model output + new observation

Turn 3:
  Turn 2 input + Turn 2 model output + new observation
```

其中：

- observation 提前生成，不运行真实工具。
- model output 使用推理服务的真实返回，并加入下一轮 history。
- 同一个 session 内逐轮串行，不同 session 之间并行。
- 所有请求使用 OpenAI-compatible `/v1/chat/completions` 接口。

这样既能真实形成逐轮增长的 KV prefix，又不依赖 agent framework 和任务正确性。

## 2. 可控条件

建议使用一个固定 workload 配置：

```yaml
seed: 42

num_sessions: 75
num_turns: 30
session_concurrency: 75

global_prefix_tokens: 20000
session_prefix_tokens: 10000
observation_tokens_per_turn: 2048
max_output_tokens: 900

inter_turn_delay_ms: 0
routing_policy: turn_round_robin

endpoints:
  - http://worker-a:port
  - http://worker-b:port
```

主要可控变量：

| 变量 | 含义 |
| --- | --- |
| `num_sessions` | 同时模拟多少条 agent trajectory |
| `num_turns` | 每条 trajectory 有多少次模型调用 |
| `global_prefix_tokens` | 所有 session 共享的 system/skills 长度 |
| `session_prefix_tokens` | 每个 session 独有的初始任务长度 |
| `observation_tokens_per_turn` | 每轮新增工具结果/环境输入长度 |
| `max_output_tokens` | 每轮模型最大输出长度 |
| `session_concurrency` | 并发 session 数量 |
| `inter_turn_delay_ms` | 模拟工具执行或 Mooncake 异步写入等待时间 |
| `routing_policy` | `sticky`、`round_robin` 或 `turn_round_robin` |

`turn_round_robin` 明确让同一 session 的相邻 turn 落到不同 endpoint：

```text
session 0: worker A -> worker B -> worker A -> worker B
session 1: worker B -> worker A -> worker B -> worker A
```

这比依赖推理引擎自己的 router 更容易确认是否真的发生跨实例访问。

## 3. 具体操作步骤

### 3.1 生成固定测试数据

vLLM main 的做法是：`benchmark_serving_multi_turn.py` 读取 `generate_multi_turn.json`，然后由 `bench_dataset.py` 使用指定 tokenizer 从文本语料中切出 common prefix、每个 conversation 的独立 prefix 和各轮消息。对应说明和实现：

- [vLLM multi-turn README](https://github.com/vllm-project/vllm/blob/main/benchmarks/multi_turn/README.md)
- [generate_multi_turn.json](https://github.com/vllm-project/vllm/blob/main/benchmarks/multi_turn/generate_multi_turn.json)
- [bench_dataset.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/multi_turn/bench_dataset.py)

这里提供一个独立于 vLLM/SGLang 的生成脚本：

[generate_agentic_prefix_workload.py](/mnt/shared-storage-user/huanghaian/code/slime_package/sglang/hha_code/py/generate_agentic_prefix_workload.py:1)

脚本使用指定模型的 tokenizer，提前生成并保存：

```text
一份所有 session 共享的 global prefix
每个 session 独立的 session prefix
每一轮固定的 observation
```

直接使用 vLLM 示例中的 Project Gutenberg 语料 `pg1184.txt`。下面按该语料能够满足本次规模执行；只需要把第一行改成推理服务实际使用的模型目录，并从 SGLang 仓库根目录运行：

```bash
MODEL_PATH=/path/to/model

mkdir -p hha_code/data

wget -O hha_code/data/pg1184.txt \
  https://www.gutenberg.org/ebooks/1184.txt.utf-8

python hha_code/py/generate_agentic_prefix_workload.py \
  --tokenizer "$MODEL_PATH" \
  --corpus hha_code/data/pg1184.txt \
  --output hha_code/data/agentic_workload.json \
  --num-sessions 75 \
  --num-turns 30 \
  --global-prefix-tokens 20000 \
  --session-prefix-tokens 10000 \
  --observation-tokens 2048 \
  --seed 42 \
  --trust-remote-code
```

成功后，后续所有 cache 配置实验都读取同一个 `hha_code/data/agentic_workload.json`，不要重新生成，以保证各组输入完全一致。

默认参数要求语料至少包含：

```text
20000 + 75 × 10000 + 30 × 2048
= 831440 tokens
```

所有 session 可以复用同一组 observation，因为每个 session 在 observation 之前已经有不同的 `session_prefix`，不会产生错误的跨 session 完整历史命中。

输出 JSON 结构：

```json
{
  "schema_version": 1,
  "metadata": {
    "num_sessions": 75,
    "num_turns": 30
  },
  "global_prefix": {
    "text": "...",
    "target_tokens": 20000,
    "actual_tokens": 20000
  },
  "sessions": [
    {
      "session_id": 0,
      "session_prefix": {
        "text": "...",
        "target_tokens": 10000,
        "actual_tokens": 10000
      }
    }
  ],
  "observations": [
    {
      "turn_id": 0,
      "text": "...",
      "target_tokens": 2048,
      "actual_tokens": 2048
    }
  ]
}
```

生成内容对应：

```text
global_prefix: 20K tokens
session_0_prefix: 10K tokens
session_1_prefix: 10K tokens
...
shared_observation_0..29: 每段 2048 tokens
```

脚本会同时记录 `target_tokens` 和 decode 后重新 tokenize 得到的 `actual_tokens`。后续统计应以实际 token 数为准。

数据只生成一次。不同缓存配置的 A/B 测试必须复用同一个 `agentic_workload.json`。

### 3.2 执行多轮 session

每个 session 独立执行：

```python
messages = [
    {"role": "system", "content": global_prefix},
    {
        "role": "user",
        "content": session_prefix + observations[0],
    },
]

for turn in range(num_turns):
    endpoint = select_endpoint(session_id, turn, routing_policy)

    response = await chat_completion(
        endpoint=endpoint,
        messages=messages,
        stream=True,
        temperature=0,
        max_tokens=max_output_tokens,
    )

    record_latency_and_cache_metrics(response)

    messages.append({
        "role": "assistant",
        "content": response.content,
    })

    if turn + 1 < num_turns:
        messages.append({
            "role": "user",
            "content": observations[turn + 1],
        })

    await sleep(inter_turn_delay_ms)
```

模型回答是否正确不影响测试；只要真实输出被原样加入下一轮，就能形成正确的 KV prefix 复用关系。

### 3.3 运行缓存配置 A/B

推荐先运行以下五组：

| ID | 服务配置 | 路由 | 目的 |
| --- | --- | --- | --- |
| A | prefix cache 关闭 | `turn_round_robin` | 全量 Prefill baseline |
| B | 仅本地 GPU prefix cache | `turn_round_robin` | 观察跨实例 miss |
| C | 仅本地 GPU prefix cache | `sticky` | 本地缓存性能上界 |
| D | GPU + Host cache | `turn_round_robin` | 验证 L2 是否只能帮助本实例 |
| E | GPU + Host + 分布式 L3 | `turn_round_robin` | 验证跨实例共享收益 |

对于 SGLang 的 Mooncake 配置，E 组应显式使用：

```bash
--enable-hierarchical-cache
--hicache-storage-backend mooncake
--hicache-write-policy write_through
```

每组实验执行前：

```text
1. 启动对应配置的推理服务
2. 使用与正式测试不同的 prompt 做模型 warmup
3. 清空所有 worker 的本地 prefix cache
4. 清空 Mooncake，或者切换到新的 cache namespace
5. 清零或记录一次服务 metrics 作为 baseline
6. 执行完全相同的 workload
7. 保存逐请求结果和整组汇总结果
```

每组至少运行三次，比较中位数。

### 3.4 单独扫描“命中率对 E2E”的关系

多轮 workload 的命中率由运行过程自然产生。如果需要精确得到“命中 25%、50%、75%、90% 时 E2E 如何变化”，再增加一组固定长度测试。

固定：

```text
总 prompt 长度 L = 32768 tokens
输出长度 O
并发度
模型和服务配置
```

只改变提前缓存的 prefix 长度 `K`：

```text
0%:  K = 0
25%: K = 8192
50%: K = 16384
75%: K = 24576
90%: K ≈ 29440
95%: K ≈ 31104
```

本地缓存测试：

```text
1. 在 worker A 上发送长度 K 的 seed prefix
2. 在 worker A 上发送相同 prefix + 长度 L-K 的新 suffix
3. 记录实际命中率和 E2E
```

分布式缓存测试：

```text
1. 在 worker A 上发送长度 K 的 seed prefix
2. 等待 KV 写入共享 L3
3. 在 worker B 上发送相同 prefix + 长度 L-K 的新 suffix
4. 记录实际 L3 命中率和 E2E
```

为了测异步写入窗口，可以分别等待：

```text
0 ms、100 ms、1 s、5 s
```

总 prompt 长度必须保持不变，否则无法区分 E2E 改善来自缓存命中还是输入本身变短。

## 4. 需要观察的指标

每个请求记录：

```text
session_id
turn_id
实际 endpoint/worker
prompt tokens
output tokens
actual cached tokens
device/host/storage cached tokens（服务支持时）
TTFT
request E2E latency
```

每个 session 记录：

```text
session E2E = 最后一轮完成时间 - 第一轮开始时间
session 总 prompt/output/cached tokens
完成的 turn 数
```

每组实验汇总：

```text
实际命中率 = Σ cached tokens / Σ prompt tokens
L3 命中率 = Σ storage cached tokens / Σ prompt tokens
P50/P90/P99 TTFT
P50/P90/P99 request E2E
P50/P90/P99 session E2E
整个 workload makespan
request throughput
output token throughput
```

其中必须区分：

```text
理论可复用率：history tokens / prompt tokens
实际命中率：服务真实返回或 metrics 统计的 cached tokens / prompt tokens
```

如果推理引擎没有统一的 cached-token 字段，只为该引擎增加一个很薄的 metrics adapter；workload 生成、请求调用、路由和 E2E 计时保持不变。

# Qwen3.5 Linear Attention Prefill Cache 源码分析

> 本文基于当前 SGLang 源码分析两件事：
>
> 1. 常规 full-attention 模型的 prefill cache / prefix cache 是怎么做的。
> 2. Qwen3.5 / Qwen3-Next 这类带 linear attention / GDN 模块的模型，为什么需要在普通 KV cache 之外做特殊处理。
>
> 为了方便新手阅读，前半部分先不讲 Qwen3.5，而是从“为什么需要 cache”开始讲。中间也会穿插 SWA / sliding window attention，帮助区分不同 cache component 的语义。

---

## 1. 先建立几个基础概念

### 1.1 Token 是什么

LLM 处理的不是原始字符串，而是 token id 序列。比如用户输入：

```text
你好，介绍一下 SGLang
```

tokenizer 会把它切成一串整数：

```text
[token_0, token_1, token_2, ...]
```

模型真正看到的是这些 token id。后面讲的 prefix cache，也是按 token id 序列做匹配，而不是按原始字符串做匹配。

### 1.2 一次生成请求分成 prefill 和 decode

普通自回归 LLM 的一次请求通常分两段：

```text
prompt: 用户输入的上下文
output: 模型逐步生成的新 token
```

执行上也分两段：

```text
prefill 阶段：
  一次性处理 prompt 中已有的 token。
  例如 prompt 有 2000 个 token，就把这 2000 个 token 跑过模型。

decode 阶段：
  每次生成 1 个或少量 token。
  生成出来的新 token 会追加到上下文里，再继续生成下一个 token。
```

用一个例子表示：

```text
prompt tokens:
  A B C D

prefill:
  一次处理 A B C D

decode step 1:
  基于 A B C D 生成 E

decode step 2:
  基于 A B C D E 生成 F

decode step 3:
  基于 A B C D E F 生成 G
```

### 1.3 为什么需要 KV cache

在 full attention Transformer 里，每一层 attention 都会为每个 token 产生 K/V：

```text
token_i -> K_i, V_i
```

decode 时，新 token 需要 attend 到所有历史 token。如果每生成一个 token 都重新计算整个历史，就会非常慢。

所以常规做法是：

```text
prefill 时：
  计算 prompt 每个 token 的 K/V，并存入 KV cache。

decode 时：
  只计算新 token 的 K/V。
  attention 直接读取历史 token 已经存好的 K/V。
```

这样 decode step 2 不需要重新计算 `A B C D E` 的全部 K/V，只要计算新 token `F` 的 K/V，然后读取历史 K/V 即可。

---

## 2. 常规模型的 prefix cache / radix cache 是什么

KV cache 解决的是“同一个请求内部，decode 不重复算历史”。Prefix cache 解决的是“不同请求之间，如果前缀相同，也不要重复算”。

例如第一个请求：

```text
A B C D -> 生成 E F G
```

第二个请求：

```text
A B C D X Y
```

第二个请求的前缀 `A B C D` 已经在第一个请求里算过。如果 KV cache 还在，就可以直接复用 `A B C D` 的 K/V，只 prefill 后面的 `X Y`。

所以常规 full attention 的 prefix cache 可以理解为：

```text
token prefix -> KV cache pages
```

这里的 `token prefix` 是 token id 序列，`KV cache pages` 是这些 token 在 KV memory pool 里的位置。

SGLang 中这个结构主要叫 radix cache / radix tree：

- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`

名字里 radix 的意思是“压缩前缀树”。它适合存很多共享前缀的 token 序列。

### 2.1 为什么用 radix tree

假设缓存里有两条序列：

```text
请求 1: A B C D E
请求 2: A B C X Y
```

它们共享 `A B C`。Radix tree 不会为两条序列各自完整存一遍，而是共享公共前缀：

```text
A B C
  -> D E
  -> X Y
```

这样做有两个好处：

1. prefix match 快：给一个新 token 序列，可以沿着树找最长已缓存前缀。
2. 公共前缀天然共享：多个请求命中同一段 KV cache 时，只需要维护引用计数和生命周期。

### 2.2 常规 prefix cache 的命中流程

请求进入调度时，会走 `Req.init_next_round_input()`：

```python
match_result = tree_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(...),
        req=self,
        cow_mamba=cow_mamba,
    )
)
```

对常规模型来说，关键结果是：

```text
match_result.device_indices
```

它表示“命中的 prefix 对应哪些 KV slot”。

然后 request 会记录：

```text
req.prefix_indices = match_result.device_indices
```

后续 prefill 只需要处理未命中的 suffix：

```text
完整输入: A B C D X Y
已命中:   A B C D
需要算:           X Y
```

---

## 3. 常规模型的 cache 生命周期

这部分容易混淆：不是每个 KV 一产生就马上进入 radix tree。SGLang 里要区分两类状态：

```text
active request KV:
  当前请求正在运行，KV 由这个 request 持有。

radix cache KV:
  请求结束或 chunked prefill 中间保存后，KV 被 tree cache 接管，可以给后续请求复用。
```

### 3.1 prefill 时 KV 写到哪里

在 prefill/extend batch 中，SGLang 会为新 token 分配 KV slot，并把 request 的 token 位置映射写入 `req_to_token_pool`。

可以把它理解成：

```text
req_to_token[req_id, position] = kv_slot
```

比如：

```text
token:    A   B   C   D
slot:    10  11  12  13
```

此时这些 KV slot 先属于当前 request，不一定已经是 radix cache 的一部分。

### 3.2 一个 token 是否对应一个 KV slot

对常规 full-attention 模型，可以先近似理解为：

```text
一个 token 需要一个 KV slot
```

原因是每个 token 在每一层 attention 里都会产生自己的 K/V：

```text
token_i
  -> layer 0 的 K_i / V_i
  -> layer 1 的 K_i / V_i
  -> layer 2 的 K_i / V_i
  -> ...
```

SGLang 里通常用一个全局 slot index 表示“这个 token 的 KV 存储位置”。这个 slot 不是只存一层，而是作为索引，指向底层 KV cache 张量中这个 token 在各层的 K/V 存储位置。

所以逻辑上可以画成：

```text
token position 0 -> kv_slot 100
token position 1 -> kv_slot 101
token position 2 -> kv_slot 102
token position 3 -> kv_slot 103
```

`req_to_token_pool` 负责记录这个映射：

```text
req_to_token[req_id, token_position] = kv_slot
```

举个例子：

```text
请求 req_7 的 token:
  A B C D

分配到的 KV slot:
  A -> 100
  B -> 101
  C -> 102
  D -> 103

req_to_token[req_7]:
  [100, 101, 102, 103]
```

prefill 时，如果 prompt 有 2000 个未命中的 token，就需要为这 2000 个 token 分配 2000 个 KV slot。decode 时，每生成一个新 token，也需要再分配一个新的 KV slot。

不过这里有几个容易误解的细节：

1. **prefix cache 命中的 token 不重新分配 slot**
  如果 `A B C D` 已经命中 radix cache，那么当前请求会复用已有的 KV slot，不会再为空的 `A/B/C/D` 分配新 slot。只会给未命中的 suffix 分配新 slot。
2. **底层可能按 page/block 管理**
  如果 `page_size > 1`，allocator 可能按一页一页管理 KV 空间。  
   但从 request 的逻辑视角看，仍然可以理解为每个 token position 对应一个 KV slot index。
3. **speculative decoding 可能预分配更多**
  投机解码可能会提前分配一些 KV slot，最后没有被接受的 draft token 对应的 slot 会被释放。源码里用 `kv_committed_len` 和 `kv_allocated_len` 区分“真正提交的 KV”和“预分配的 KV”。
4. **特殊模型不完全一样**
  Sliding window attention、MLA/DSA、Qwen3.5 linear attention 等模型会有特殊 layout 或额外 state。  
   但对于普通 full-attention 模型，“一个 token 一个 KV slot”是最适合建立直觉的理解方式。

### 3.2.1 KV cache 在内存里到底长什么样

SGLang 源码里有一句很关键的注释：

```text
SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
```

对应文件：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/allocator/token.py`
- `python/sglang/srt/mem_cache/allocator/paged.py`

可以分成三层看：

```text
ReqToTokenPool:
  req_id + token_position -> kv_slot

TokenToKVPoolAllocator:
  管理哪些 kv_slot / page 空闲，哪些已经被占用

KVCache / TokenToKVPool:
  真正保存每一层 K/V tensor 的 GPU 内存
```

所以 `kv_slot` 不是一块单独的 Python 对象，而是一个整数索引。这个整数会被拿去索引每一层的 K/V buffer。

### 3.2.2 一个 KV slot 只存一个 token 吗

从逻辑上说：**是的，一个 KV slot 对应一个 token position 的 KV。**

但要注意，“一个 KV slot”不是只存一层，也不是只存一个标量。它表示：

```text
slot = 100

layer 0:
  K buffer[100] = token 在 layer 0 的 K
  V buffer[100] = token 在 layer 0 的 V

layer 1:
  K buffer[100] = token 在 layer 1 的 K
  V buffer[100] = token 在 layer 1 的 V

...

layer L-1:
  K buffer[100] = token 在 layer L-1 的 K
  V buffer[100] = token 在 layer L-1 的 V
```

也就是说，`slot=100` 是一个跨层共享的行号。每一层都有自己的 K/V tensor，但同一个 token 在这些 tensor 里使用同一个 slot id。slot=100 只是一个统一的逻辑行号。不同层有各自独立的 K/V tensor，所以实际内存不是同一块。

### 3.2.3 默认 MHA/GQA KV cache 的 shape

对常规 MHA/GQA attention，SGLang 里常见的 pool 是 `MHATokenToKVPool`。默认 NHD layout 下，每一层各有一个 K buffer 和一个 V buffer：

```python
self.k_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.head_dim), ...)
    for _ in range(self.layer_num)
]

self.v_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.v_head_dim), ...)
    for _ in range(self.layer_num)
]
```

可以简化成：

```text
k_buffer[layer_id].shape = [num_slots, num_kv_heads_local, head_dim]
v_buffer[layer_id].shape = [num_slots, num_kv_heads_local, v_head_dim]
```

其中：

- `num_slots` 近似是最多能同时容纳多少个 token 的 KV。
- `num_kv_heads_local` 是当前 rank 上的 KV head 数。TP 场景下它通常是切分后的本地 head 数，不一定是全模型 head 数。
- `head_dim` 是每个 head 的维度。
- 通常 `v_head_dim == head_dim`，但源码允许二者不同。
- `size + page_size` 里多出来的部分包含 padding / page 对齐用的槽位。

写入时，attention 层会产生：

```text
cache_k.shape = [num_new_tokens, num_kv_heads_local, head_dim]
cache_v.shape = [num_new_tokens, num_kv_heads_local, v_head_dim]
loc.shape     = [num_new_tokens]
```

然后执行类似：

```python
k_buffer[layer_id][loc] = cache_k
v_buffer[layer_id][loc] = cache_v
```

所以如果某个 token 的 `kv_slot=100`，那么它在第 `layer_id` 层的 K/V 大概就是：

```text
K = k_buffer[layer_id][100]  # [num_kv_heads_local, head_dim]
V = v_buffer[layer_id][100]  # [num_kv_heads_local, v_head_dim]
```

这个意义上，“一个 token 一个 KV slot”里的 slot 存的是“这一层的一行 K 和一行 V”；所有层合起来才是这个 token 的完整 KV cache。

### 3.2.4 KV block / page 是什么

`page_size` 大于 1 时，allocator 不再把每个 slot 完全独立管理，而是按 page / block 管理连续的一组 slot。

例如：

```text
page_size = 16

page 0 -> slot 0..15
page 1 -> slot 16..31
page 2 -> slot 32..47
```

源码里的 `PagedTokenToKVPoolAllocator` 分配 page 时，会返回 page 内所有 token slot：

```text
out_indices =
  page_id * page_size + [0, 1, ..., page_size - 1]
```

所以：

```text
KV slot:
  token 级地址，例如 100。

KV page / KV block:
  一组连续 KV slot，例如 96..111。
```

用户常说的 KV block，在这里可以先理解为 KV page。不同 kernel 或外部系统可能叫 block/page，但核心意思都是“按固定 token 数组织的一块 KV cache”。

### 3.2.5 Paged attention 和 radix tree 的关系

这两个名字经常一起出现，但它们解决的是不同层面的问题。

Paged attention 解决的是：

```text
attention kernel 如何高效读取离散存放的历史 KV。
```

一个请求的上下文 token 在逻辑上是连续的：

```text
position: 0   1   2   3   4   5
token:    A   B   C   D   E   F
```

但它们的 KV slot 在物理内存里不一定是一整段连续空间，尤其在多请求、释放、复用之后：

```text
position: 0   1   2   3   4   5
slot:     80  81  82  200 201 48
```

`req_to_token_pool` 保存的就是这张逻辑 position 到物理 slot 的表：

```text
req_to_token[req_id, position] = kv_slot
```

Paged attention kernel 读历史 KV 时，会通过这张表或由它派生出的 page table 找到每个 token/page 的物理位置，然后去 `k_buffer[layer][slot]`、`v_buffer[layer][slot]` 读取。

Radix tree 解决的是另一个问题：

```text
不同请求之间，如何根据 token 前缀找到可复用的 KV slot。
```

也就是说：

```text
paged attention:
  给定一个正在运行的请求，负责让 attention 找到历史 KV 在哪里。

radix tree:
  给定一个新请求的 token 序列，负责找最长已缓存前缀，并返回这段前缀对应的 KV slot/page indices。
```

它们会串起来使用：

```text
新请求 token: A B C D X Y

radix tree:
  发现 A B C D 已缓存
  返回 A/B/C/D 对应的 KV slot

req_to_token_pool:
  把这些 slot 填到当前请求的 position 0..3
  给 X/Y 新分配 slot

paged attention:
  forward 时按 req_to_token / page table 读取 A/B/C/D/X/Y 的 KV
```

### 3.2.6 page_size 为什么会影响 prefix cache

如果 `page_size=1`，每个 token slot 都可以独立插入 radix tree。

如果 `page_size=16`，radix cache 通常只能安全地缓存 16 的倍数长度的前缀：

```text
prefix 长度 100 -> cache 到 96
prefix 长度 1100 -> cache 到 1088
```

原因是 radix tree 存的是可复用的 KV page / slot 序列。为了让后续复用、释放、引用计数和 page allocator 对齐，插入前会做：

```python
RadixKey(...).page_aligned(self.page_size)
```

因此：

```text
token slot 是 token 级逻辑地址。
page/block 是 allocator 和 attention kernel 更喜欢的连续管理单位。
radix tree 的 cache 边界通常要和 page/block 对齐。
```

### 3.3 请求结束时如何进入 radix cache

常规 `RadixCache.cache_finished_req()` 会在请求结束时把 committed KV 插入 radix tree：

```python
kv_committed_len = req.pop_committed_kv_cache()
token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
kv_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, : len(token_ids)
]

radix_key = RadixKey(
    token_ids, req.extra_key, is_bigram=self.is_eagle
).page_aligned(self.page_size)

self.insert(InsertParams(key=radix_key, value=values, priority=priority))
```

对应文件：

- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`

注意这里的 `token_ids` 是：

```text
origin_input_ids + output_ids
```

也就是说，默认情况下不仅 prompt 会进入 cache，生成出来的 output token 也会进入 cache。

### 3.4 生成 1000 token 会全部变成 cache 吗

默认语义上，会尝试把 committed 的 `prompt + generated output` 都插入 radix cache。

假设：

```text
prompt 长度 = 100
生成长度 = 1000
总长度 = 1100
```

请求结束时，SGLang 会构造：

```text
token_ids = prompt_tokens + output_tokens
```

然后把前 `kv_committed_len` 个 token 对应的 KV 插入 radix tree。

但是要注意几个限制：

1. **page 对齐**
  `RadixKey(...).page_aligned(self.page_size)` 会把长度截到 page 对齐。  
   如果 `page_size=16`，长度 1100 会对齐到 1088，最后 12 个 token 不进 radix tree。
2. **内存淘汰**
  插入了不代表永久存在。KV cache 空间不足时，radix cache 会按策略淘汰部分节点。
3. **禁用或跳过插入**
  如果 radix cache 被禁用，或者释放时 `is_insert=False`，不会插入。
4. **特殊配置**
  例如 `strip_thinking_cache` 会让 `_cache_commit_len()` 只保留 prompt 部分，把 thinking/answer 生成部分回收。

所以更准确地说：

```text
生成 1000 token 后，请求结束时默认会尝试把 prompt + 1000 output token 的 committed KV 插入 prefix cache。
但实际进入 tree 的长度会受 page_size、配置、内存淘汰等影响。
```

### 3.5 运行中的 decode token 什么时候进 cache

decode 每生成一个 token 时，这个 token 的 KV 会写入 active request 的 KV slot。

但它通常不是“每生成一个 token 就立即插入 radix tree”。普通路径下，它会等到请求结束释放 KV 时，由 `cache_finished_req()` 一次性把 committed KV 交给 radix cache。

因此：

```text
生成过程中:
  output token 的 KV 在 active request 里。

请求结束时:
  committed KV 被插入 radix tree。
```

这点和 Qwen3.5 linear attention 的 extra buffer 也有联系：Qwen3.5 的 active recurrent state 会不断被覆盖，所以它不能简单等结束后直接拿 active state 当任意 prefix cache；后文会展开。

---

## 4. 分段：chunked prefill 和 radix tree 分段

你问“是否有分段的说法”，答案是有，但要分清两种不同含义。

### 4.1 数据结构上的分段：radix tree 节点

Radix tree 是压缩前缀树。它不是每个 token 一个节点，而是按公共前缀和分叉点组织。

例如：

```text
已缓存:
  A B C D E

新插入:
  A B C X Y
```

tree 可能变成：

```text
root
  -> A B C
       -> D E
       -> X Y
```

这个“分段”是为了共享公共前缀，不是固定按 256 或 1024 token 切。

### 4.2 调度上的分段：chunked prefill

长 prompt 可能不会一次 prefill 完。比如 prompt 有 10000 token，系统可能每轮只算 2048 个 token：

```text
第 1 轮 prefill: token 0..2047
第 2 轮 prefill: token 2048..4095
第 3 轮 prefill: token 4096..6143
...
```

这叫 chunked prefill。

SGLang 调度里，如果请求还没 prefill 完，会把它作为 `chunked_req` 暂存。中间结果会通过：

```python
maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
```

进入：

```python
tree_cache.cache_unfinished_req(req, chunked=True)
```

相关文件：

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`

`cache_unfinished_req()` 会把已经算完的 prefix 插入 radix cache，并更新当前 request 的 `prefix_indices`。下一轮继续 prefill 时，这段已经算过的 prefix 就可以当作命中的 cache，继续只算后面的 chunk。

简化时序：

```text
长 prompt: A B C D E F G H I J
chunk size = 4

第 1 轮:
  算 A B C D
  cache_unfinished_req: 把 A B C D 放入 radix cache

第 2 轮:
  prefix 命中 A B C D
  只算 E F G H
  cache_unfinished_req: 把 A B C D E F G H 放入 radix cache

第 3 轮:
  prefix 命中 A B C D E F G H
  只算 I J
```

### 4.3 page_size 对 cache 长度的影响

radix key 会做 page 对齐：

```python
RadixKey(...).page_aligned(self.page_size)
```

如果 `page_size=1`，基本每个 token 都可以作为 cache 边界。

如果 `page_size=16`，只有长度为 16 的倍数的前缀能完整进入 tree：

```text
长度 100 -> 只缓存到 96
长度 1100 -> 只缓存到 1088
```

未对齐尾部仍然可以在当前 request 中使用，但不会作为 radix tree 的可共享前缀保存下来。

---

## 5. 常规模型的完整时序

以普通 full attention 模型为例：

```text
请求 1:
  prompt = A B C D

调度:
  Req.init_next_round_input()
  -> tree_cache.match_prefix()
  -> 假设没有命中，prefix_indices = []

prefill:
  计算 A B C D 的 K/V
  req_to_token 记录 token position -> KV slot

decode:
  生成 E
  写入 E 的 K/V
  生成 F
  写入 F 的 K/V

请求结束:
  token_ids = A B C D E F
  cache_finished_req()
  -> 插入 radix cache

请求 2:
  prompt = A B C D X

调度:
  match_prefix() 命中 A B C D
  prefix_indices 指向已缓存的 KV slot

prefill:
  只计算 X
```

---

## 6. 先从算法层面理解 Qwen3.5 的 cache 区别

前面讲的是常规 full-attention 模型。进入 Qwen3.5 之前，最好先把几个概念分开，否则很容易把源码里的 `prefill cache`、`chunked prefill`、`KV cache`、`mamba state` 混在一起。

### 6.1 KV cache 是“单个请求内部”的历史复用

KV cache 的核心问题是：

```text
同一个请求 decode 时，如何不重复计算历史 token？
```

例如：

```text
prompt = A B C D
```

prefill 后，模型已经算出了：

```text
A 的 K/V
B 的 K/V
C 的 K/V
D 的 K/V
```

decode 生成 `E` 时，不需要重新计算 `A B C D` 的 K/V，只需要：

```text
计算 E 的 K/V
读取 A/B/C/D 的历史 K/V
```

所以 KV cache 是模型推理的基础机制。没有 KV cache，decode 会非常慢。

### 6.2 Prefix cache / prefill cache 是“不同请求之间”的前缀复用

Prefix cache 解决的是另一个问题：

```text
不同请求之间，如果前缀相同，能不能跳过已经算过的 prefix？
```

例如：

```text
请求 1: A B C D E F
请求 2: A B C D X Y
```

请求 2 的 `A B C D` 和请求 1 前面一样。如果请求 1 的 `A B C D` KV 还在 radix cache 里，请求 2 就可以：

```text
复用 A B C D 的 KV
只 prefill X Y
```

因此，prefix cache / prefill cache 可以理解为：

```text
跨请求共享已经算过的前缀
```

它依赖 KV cache，但不是同一个概念：

```text
KV cache:
  让当前请求 decode 不重复算历史。

prefix cache:
  让后来的请求不重复算相同前缀。
```

### 6.3 Chunked prefill 是“长 prompt 分批计算”

Chunked prefill 又是第三个概念。

它解决的问题是：

```text
prompt 太长，不能或不想一次 prefill 完，怎么办？
```

比如 prompt 有 10000 token，系统可以每轮只算 2048 token：

```text
第 1 轮: 算 0..2047
第 2 轮: 算 2048..4095
第 3 轮: 算 4096..6143
...
```

这叫 chunked prefill。它是调度策略，不等价于 prefix cache。

但 chunked prefill 经常会和 prefix cache 配合：

```text
第 1 个 chunk 算完后，把 0..2047 放进 radix cache。
第 2 个 chunk 开始时，前 0..2047 就变成已命中 prefix，只需要继续算后面的 chunk。
```

所以三者关系可以这样理解：

```text
KV cache:
  单请求内部保存历史 K/V。

prefix cache / prefill cache:
  跨请求或跨 chunk 复用已算前缀。

chunked prefill:
  把长 prompt 切成多轮 prefill。
```

### 6.4 不同 cache 格式的共同点：先做 token prefix match

不管底层 cache 是 full KV、SWA window KV，还是 Qwen3.5 的 mamba recurrent state，第一步都是一样的：

```text
用 token ids 做 prefix match
```

也就是 radix tree 根据请求的 token 序列找最长共享前缀。

区别在第二步：**命中这个 token prefix 后，各类层需要恢复或读取的 cache 内容不同。**

可以抽象成：

```text
token prefix -> component-specific cached state
```

不同 component 的 state 不一样：

```text
Full attention:
  cache 内容是 prefix 内每个 token 的 K/V slots。

SWA / sliding window attention:
  cache 内容仍然是 K/V slots，
  但只需要命中点之前最近 sliding_window_size 个 token 的 KV。

Linear attention / Mamba / GDN:
  cache 内容不是每个历史 token 的 KV，
  而是命中 prefix 末尾的 conv state + recurrent / SSM state。
```

SGLang 的 unified radix cache 也是这个思路。同一个 radix node 可以挂多个 component：

```text
FULL component:
  保存 full-attention KV indices。

SWA component:
  保存 sliding-window attention 的 SWA KV indices。

MAMBA component:
  保存 linear-attention recurrent state slot。
```

所以判断一个 prefix 是否能安全跳过，不能只看 token 是否匹配，还要看相关 component 是否都满足这个模型继续计算所需的条件。

举例：

```text
token prefix 命中 A B C D E F
```

不同模型真正需要的东西不同：

```text
普通 full attention:
  需要 A..F 的 KV 都在。

SWA, window=3:
  SWA 层只需要 D E F 的 SWA KV 在。
  A B C 的 SWA KV 可以已经释放。

Qwen3.5 linear attention:
  linear 层需要 state_after_F 在。
```

因此：

```text
prefix match 是统一入口。
cache 复用语义由 component 决定。
```

### 6.5 常规 full attention 为什么好 cache

普通 full attention 的 prefix cache 是：

```text
token prefix -> KV cache pages
```

原因是 full attention 的历史信息就是一组 per-token K/V：

```text
A -> K_A, V_A
B -> K_B, V_B
C -> K_C, V_C
D -> K_D, V_D
```

如果后来请求命中 `A B C D`，只要拿回这些 token 对应的 KV slot，就能继续计算 suffix。

也就是说，full attention 的 cache 是“可逐 token 保存、可逐 token 复用”的。

### 6.6 SWA / sliding window attention 的 prefix cache 有什么不同

SWA 仍然是 attention，也仍然保存 K/V，但它不是所有历史 token 都有用。

如果：

```text
sliding_window_size = 4
prefix = A B C D E F G H
```

那么在 SWA 层里，从 `H` 后面继续算下一个 token 时，只需要最近窗口：

```text
E F G H
```

更早的：

```text
A B C D
```

对 SWA 层已经出窗口，可以释放 SWA KV。

所以 SWA 的 prefix cache 不是要求：

```text
整个 prefix 的 SWA KV 都存在
```

而是要求：

```text
命中点之前最近 sliding_window_size 个 token 的 SWA KV 连续存在
```

这也是为什么 SWA cache 里会看到 tombstone / `value=None` 这类概念。比如：

```text
A B C D:
  FULL 有 KV
  SWA tombstone / value=None

E F G H:
  FULL 有 KV
  SWA 有 KV
```

如果新请求只命中：

```text
A B C D X
```

并且 `window=4`，那么从 `D` 后算 `X` 时，SWA 层需要：

```text
A B C D
```

但这段 SWA KV 已经 tombstone，所以这个 prefix 对 hybrid SWA 模型不能作为完整有效命中。

如果新请求命中：

```text
A B C D E F G H X
```

从 `H` 后算 `X` 时，SWA 层只需要：

```text
E F G H
```

这段 SWA KV 还在，所以前面的 `A B C D` tombstone 不影响继续计算。

源码里 SWA component 的很多逻辑就是围绕这个“末尾窗口”做的：

```text
token prefix match:
  仍然从 root 往下、从前往后匹配 token。

SWA 有效性检查 / lock / load-back:
  从命中的 node 往 parent 回看，
  累计最近 sliding_window_size 个 token 的 SWA KV。
```

也就是说，SWA 的“从后往前”不是 token prefix 匹配从后往前，而是：

```text
命中 prefix 后，只关心这个 prefix 末尾窗口的 SWA KV 是否存在。
```

SWA 也有边界问题，但它的边界是 sliding window 的左边界：

```text
prefix 长度 = N
window = W

SWA 需要保留:
  token[N-W : N]

可以释放:
  token[0 : N-W]
```

源码里用 `swa_evicted_seqlen` 记录 SWA 左边已经释放到哪里。释放时还会考虑 `page_size`，所以实际边界通常是 page-aligned 的。

SWA 理论上也可以像 mamba 一样周期性保存多个历史窗口快照，但一个 SWA checkpoint 的代价是：

```text
O(sliding_window_size) 个 token 的 KV
```

而 mamba checkpoint 的代价是：

```text
O(recurrent_state_size)
```

所以 SGLang 更倾向于让 SWA 只保留末尾窗口，而不是每隔 N token 保存一份完整 window KV 快照。

### 6.7 Linear attention 为什么不能只 cache KV

Qwen3.5 的部分层不是 full attention，而是 linear attention / GatedDeltaNet。它的核心思路不是保存每个历史 token 的 K/V，然后让新 token attend 所有历史 K/V。

更接近下面这种递推形式：

```text
state_0 = 初始状态

读入 A:
  state_1 = update(state_0, A)

读入 B:
  state_2 = update(state_1, B)

读入 C:
  state_3 = update(state_2, C)

读入 D:
  state_4 = update(state_3, D)
```

到了 `D` 之后，历史 `A B C D` 的信息主要被压缩进一个 recurrent state：

```text
prefix A B C D -> state_4
```

下一步处理 `X` 时，不是读取 `A/B/C/D` 每个 token 的 K/V，而是从 `state_4` 继续递推：

```text
state_5 = update(state_4, X)
```

所以对于 linear attention 来说，cache 的对象不是每个 token 的 K/V，而是：

```text
某个 prefix 位置上的 recurrent state
```

这就是 Qwen3.5 的本质区别。

### 6.8 Qwen3.5 需要两套 cache 一起正确

Qwen3.5 是 hybrid 模型：有些层是 full attention，有些层是 linear attention。

因此一个 prefix 如果要被安全复用，必须同时满足两件事：

```text
full attention 层:
  命中这个 prefix 的 KV cache。

linear attention 层:
  命中这个 prefix 对应的 recurrent state。
```

所以 Qwen3.5 的 prefix cache 需要变成：

```text
token prefix -> full-attention KV pages + linear-attention recurrent state
```

其中 linear-attention recurrent state 至少包含：

- conv state：causal conv 的滑动窗口状态。
- SSM / temporal state：GatedDeltaNet / Mamba-like recurrent 状态。

这就是 Qwen3.5 需要特殊处理 prefill cache 的根本原因。

### 6.9 为什么还要关心 chunk 边界

full attention 的 KV 基本是每个 token 都有一个独立 K/V，所以直觉上每个 token 位置都可以作为 cache 边界。

linear attention 不一样。它的 prefill kernel 常常按 chunk 计算，并且中间 state 不一定在任意 token 位置都直接可取。源码里会选择一些可控边界去保存 state，例如：

```text
mamba_cache_chunk_size 边界
mamba_track_interval 边界
```

这也是为什么 Qwen3.5 的文档里会看到：

- `mamba_cache_chunk_size`
- `mamba_track_mask`
- `mamba_track_indices`
- `mamba_last_track_seqlen`
- ping-pong track buffer

这些不是普通 KV cache 的概念，而是为了让 linear attention 的 recurrent state 在正确的 prefix 边界上被保存和复用。

---

## 7. 直观对比：普通模型 vs Qwen3.5

### 7.1 普通 full-attention 模型

普通模型可以简单理解为：

```text
每个 token 都有自己的 KV。
prefix cache 保存 token prefix 对应的 KV slot 列表。
命中 prefix 后，suffix 直接读取这些历史 KV。
```

时序：

```text
请求 1:
  A B C D
  -> 计算并保存 KV_A/KV_B/KV_C/KV_D
  -> radix cache 记录 A B C D -> [slot_A, slot_B, slot_C, slot_D]

请求 2:
  A B C D X
  -> radix cache 命中 A B C D
  -> 只计算 X
```

### 7.2 Qwen3.5 hybrid 模型

Qwen3.5 需要同时保存两类东西：

```text
full attention 层:
  A B C D -> KV slots

linear attention 层:
  A B C D -> recurrent state slot
```

时序：

```text
请求 1:
  A B C D
  -> full attention 层保存 KV_A/KV_B/KV_C/KV_D
  -> linear attention 层计算 state_after_D
  -> radix cache 记录:
       A B C D -> KV slots + state_after_D

请求 2:
  A B C D X
  -> radix cache 命中 A B C D
  -> full attention 层复用 KV slots
  -> linear attention 层恢复 state_after_D
  -> 只计算 X
```

### 7.3 如果只命中 KV 不命中 recurrent state 会怎样

对于 Qwen3.5 来说，只命中 full attention KV 是不够的。

如果缺少 linear attention 的 recurrent state，模型没法直接从 `A B C D` 后面继续递推 `X`。它只能重新跑 `A B C D` 的 linear attention 部分，才能得到正确的 state。

所以 Qwen3.5 的 prefix match 需要 full component 和 mamba/linear component 都满足，才能算真正完整命中。

---

## 8. Qwen3.5 的 linear state 到底存在哪里

这一节专门回答一个很关键的问题：

```text
linear attention 的 state 是每个 token 都存一份吗？
还是只在某些边界存？
```

答案是：**只在某些可缓存边界存，不是每个 token 都存。**

### 8.1 普通 KV 可以逐 token 存

普通 full attention 很自然：

```text
A -> KV_A
B -> KV_B
C -> KV_C
D -> KV_D
E -> KV_E
```

如果 prefix 是 `A B C D E`，radix cache 可以保存：

```text
A B C D E -> [KV_A, KV_B, KV_C, KV_D, KV_E]
```

每个 token 都有独立 KV，所以从直觉上看，每个 token 位置都可以成为 cache 边界。

### 8.2 Linear attention 不会每个 token 存一个 state

linear attention 更像一个递推过程：

```text
state_0 = 初始状态

读入 A:
  state_1 = update(state_0, A)

读入 B:
  state_2 = update(state_1, B)

读入 C:
  state_3 = update(state_2, C)

读入 D:
  state_4 = update(state_3, D)
```

从数学上说，每个位置都可以有一个 `state_i`。但在服务系统里，如果每个 token 都保存一份完整 recurrent state，内存会非常大，而且 kernel 也不一定方便在任意位置暴露中间 state。

所以 SGLang 对 Qwen3.5 这类 linear attention 的 cache 更接近：

```text
只在某些边界保存 state
```

例如：

```text
token 0..255   -> state_at_256
token 0..511   -> state_at_512
token 0..767   -> state_at_768
```

这些边界通常和以下参数或机制有关：

- `mamba_cache_chunk_size`
- `mamba_track_interval`
- `page_size`
- chunked prefill 的切分点
- decode 过程中的 track boundary

### 8.3 radix cache 对 Qwen3.5 存的是“边界 state”

对 Qwen3.5 来说，radix cache 里 full attention 部分仍然是 KV：

```text
FULL.value = KV slots
```

但 linear attention 部分不是每个 token 的 state 列表，而是：

```text
MAMBA.value = 某个 prefix 边界上的 recurrent state slot
```

可以把一个 radix node 理解成：

```text
node key = 一段 token

FULL.value:
  这段 token 对应的 KV slots

MAMBA.value:
  从 root 到这个 node 结束位置的 recurrent state
```

注意 `MAMBA.value` 表示的是“整个 prefix 到这里为止”的 state，不是这个 node 里面每个 token 各自的 state。

### 8.4 radix tree 可能长什么样

假设可缓存边界是 256、512、768，并且之前确实在这些边界插入过 cache，那么可以直观画成：

```text
root
  -> token[0:256]
       FULL.value  = token[0:256] 的 KV slots
       MAMBA.value = state_at_256
       -> token[256:512]
            FULL.value  = token[256:512] 的 KV slots
            MAMBA.value = state_at_512
            -> token[512:768]
                 FULL.value  = token[512:768] 的 KV slots
                 MAMBA.value = state_at_768
```

但实际 radix tree 不一定天然按 256 切。Radix tree 是压缩前缀树，节点怎么切取决于实际插入历史和分叉情况。

如果历史上只插入过一个 768 token 的完整前缀，也可能更像：

```text
root
  -> token[0:768]
       FULL.value  = token[0:768] 的 KV slots
       MAMBA.value = state_at_768
```

如果后来插入了共享前缀但后面分叉的请求，tree 才可能 split 成更细的节点：

```text
root
  -> token[0:512]
       FULL.value  = token[0:512] 的 KV slots
       MAMBA.value = state_at_512
       -> branch A
       -> branch B
```

所以要区分两件事：

```text
可缓存边界:
  linear state 能安全保存和复用的位置。

radix tree 节点边界:
  tree 根据插入和分叉形成的压缩节点边界。
```

它们相关，但不一定完全一样。最终能否完整命中，要看 radix node 上对应 component 是否真的有可用的 `FULL.value` 和 `MAMBA.value`。

### 8.5 超过边界的部分怎么办

假设：

```text
mamba_cache_chunk_size = 256
page_size = 16
```

某个请求实际有 1000 token，但目前 linear state 最后只缓存到了 768：

```text
已缓存边界:
  state_at_256
  state_at_512
  state_at_768
```

如果新请求有相同的 1000 token 前缀，那么完整 cache 命中可能只能到 768：

```text
token 0..767:
  full attention 复用 KV
  linear attention 恢复 state_at_768

token 768..999:
  需要重新 prefill
  重新产生这段 suffix 的 full-attention KV
  从 state_at_768 继续递推 linear state
```

注意这里不是从头重算 0..999，而是：

```text
复用 0..767
重算 768..999
```

这就是 `prefix cache` 对 Qwen3.5 的直观语义：

```text
只能复用到 KV 和 recurrent state 都存在的最长边界。
超过这个边界的 suffix 需要继续 prefill。
```

### 8.6 为什么源码里有 mamba_last_track_seqlen

`mamba_last_track_seqlen` 可以理解为：

```text
当前 request 最后一次成功保存 linear state 的 prefix 长度
```

普通模型结束时可以尝试把完整 `token_ids_len` 插入 radix cache。

Qwen3.5 如果启用了 extra buffer，插入时不能简单用完整长度，而是要看：

```text
linear state 最后成功 track 到哪里
```

如果最后只 track 到 768，即使当前请求已经算到 1000，也只能把 linear attention 可复用部分安全地插到 768。否则 radix cache 里会出现一个问题：

```text
token key 声称缓存到了 1000
但 linear state 实际只对应 768 或者已经被后续 token 覆盖
```

这会导致后续请求从错误 state 继续算，结果就不对。

因此，Qwen3.5 的插入长度经常要受 `mamba_last_track_seqlen` 限制。

---

## 9. 接下来再看源码

理解上面的算法差别之后，再看源码就比较清楚了：

```text
普通模型:
  radix cache 只需要管理 KV slot。

Qwen3.5:
  radix cache 除了 KV slot，还要管理 linear attention state slot。
  forward 前要把 cached state 恢复到当前 request。
  forward 后还要在合适边界 track 新 state，供后续 cache insert。
```

下面开始进入具体源码路径。

---

## 10. 模型入口：Qwen3.5 的 linear attention 层

相关文件：

- `python/sglang/srt/configs/qwen3_5.py`
- `python/sglang/srt/configs/qwen3_next.py`
- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/layers/radix_linear_attention.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`

`Qwen3_5TextConfig` 继承自 `Qwen3NextConfig`。`Qwen3NextConfig` 里定义了 hybrid layer 布局：

```python
@property
def layers_block_type(self):
    layer_type_list = []
    for l in range(self.num_hidden_layers):
        if (l + 1) % self.full_attention_interval == 0:
            layer_type_list.append(HybridLayerType.full_attention.value)
        else:
            layer_type_list.append(HybridLayerType.linear_attention.value)
    return layer_type_list
```

也就是说，不是所有层都是 full attention。`linear_layer_ids` 会筛出 linear attention 层，并用于后续 mamba / linear state cache 的初始化。

`Qwen3_5GatedDeltaNet` 是 Qwen3.5 的 linear attention 模块。它做几件事：

1. 输入 hidden states。
2. 投影出 `q/k/v/z` 和 GDN 参数 `a/b`。
3. 调用 `self.attn(...)`，其中 `self.attn` 是 `RadixLinearAttention`。
4. 对 core attention 输出做 gated norm 和 output projection。

简化链路：

```text
Qwen3_5GatedDeltaNet.forward()
  -> in_proj_qkvz / in_proj_ba
  -> mixed_qkv, z, a, b
  -> RadixLinearAttention.forward()
  -> GDNAttnBackend.forward_extend / forward_decode
```

`RadixLinearAttention` 本身不实现 cache 逻辑，它只是把请求转给当前 attention backend：

```python
return get_attn_backend().forward(
    layer=self,
    forward_batch=forward_batch,
    mixed_qkv=mixed_qkv,
    a=a,
    b=b,
)
```

真正处理 prefill cache / recurrent state 的地方在 `gdn_backend.py` 和 `hybrid_linear_attn_backend.py`。

---

## 11. 内存池：HybridReqToTokenPool + MambaPool

相关文件：

- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/mem_cache/memory_pool.py`

普通模型使用 `ReqToTokenPool`，它维护 request 到 KV slot 的映射。Qwen3.5 这类 hybrid SSM / linear attention 模型使用 `HybridReqToTokenPool`。

初始化入口在 `_init_pools()`：

```python
elif config := self.mambaish_config:
    self.req_to_token_pool = HybridReqToTokenPool(
        ...
        cache_params=config.mamba2_cache_params,
        mamba_layer_ids=[...],
        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=self.server_args.enable_mamba_extra_buffer_lazy(),
        ...
    )
```

`HybridReqToTokenPool` 继承普通 `ReqToTokenPool`，并额外初始化：

- `MambaPool`
- `MambaSlotAllocator`
- `req_index_to_mamba_index_mapping`
- 可选的 `req_index_to_mamba_ping_pong_track_buffer_mapping`

`MambaPool` 里真正保存 linear state：

```python
self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
```

其中：

- `conv_state`：每个 linear layer 的 causal conv window。
- `temporal_state`：每个 linear layer 的 recurrent / SSM state。

所以一个 request 在 hybrid pool 中有两类状态：

```text
req_pool_idx
  -> req_to_token[req_pool_idx, :]      # full attention KV slot mapping
  -> mamba_pool_idx                    # linear attention active state slot
```

`HybridReqToTokenPool.alloc()` 里如果 request 还没有 `mamba_pool_idx`，会分配一个 active mamba slot，并标记 `req.mamba_needs_clear = True`。

---

## 12. Prefix match：不仅要命中 KV，还要命中 mamba state

相关文件：

- `python/sglang/srt/mem_cache/registry.py`
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`
- `python/sglang/srt/managers/schedule_batch.py`

当前源码中，hybrid SSM 默认可以走两类 radix cache：

1. legacy `MambaRadixCache`
2. unified radix cache + `MambaComponent`

如果启用 unified radix tree 或 HiCache，hybrid SSM 会使用 `UnifiedRadixCache`，并挂上 `ComponentType.MAMBA`：

```python
tree_components = [ComponentType.FULL]
if ctx.is_hybrid_ssm:
    tree_components.append(ComponentType.MAMBA)
```

prefix match 的入口在 `Req.init_next_round_input()`：

```python
match_result = tree_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(...),
        req=self,
        cow_mamba=cow_mamba,
    )
)
```

match 结果里除了普通 KV 的 `device_indices`，还有：

- `mamba_host_hit_length`
- `mamba_branching_seqlen`

`MambaComponent.finalize_match_result()` 会处理 mamba state：

```python
mamba_value = last_node.component_data[self.component_type].value
if cow_mamba and mamba_value is not None:
    if req.mamba_pool_idx is None:
        dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        req.mamba_pool_idx = dst_index[0]
    req.mamba_cow_src_index = mamba_value
    req.mamba_needs_clear = False
```

这里采用 deferred copy-on-write：

- radix node 上的 `mamba_value` 是 cached state slot。
- 当前请求需要自己的 active state slot。
- match 阶段只记录 `req.mamba_cow_src_index`。
- forward 前再在 forward stream 上 copy，避免和计算流发生竞态。

执行 deferred copy 的地方是 `MambaAttnBackendBase._execute_deferred_mamba_cow_and_clear()`：

```python
if forward_batch.mamba_cow_src_indices is not None:
    self.req_to_token_pool.mamba_pool.copy_from(
        forward_batch.mamba_cow_src_indices,
        forward_batch.mamba_cow_dst_indices,
    )
```

如果启用了 int8 mamba checkpoint pool，则会从 checkpoint pool dequantize 到 active mamba pool。

---

## 13. Extend / prefill forward：从 cached state 接着算 suffix

相关文件：

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`

`ForwardBatch` 会携带 mamba 相关字段：

```python
mamba_track_indices: Optional[torch.Tensor]
mamba_track_mask: Optional[torch.Tensor]
mamba_track_seqlens: Optional[torch.Tensor]
mamba_cow_src_indices: Optional[torch.Tensor]
mamba_cow_dst_indices: Optional[torch.Tensor]
mamba_clear_indices: Optional[torch.Tensor]
```

linear attention backend 初始化 metadata 时，先执行 deferred clear / copy-on-write，然后获取本 batch 的 active mamba slot：

```python
mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
    forward_batch.req_pool_indices
)
```

`GDNAttnBackend.forward_extend()` 是 prefill / extend 的核心：

```python
mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
conv_states = mamba_cache_params.conv[0]
ssm_states = mamba_cache_params.temporal

has_initial_states = forward_batch.extend_prefix_lens > 0
```

如果 prefix cache 命中，则 `extend_prefix_lens > 0`。这会让 causal conv 从已有 conv state 继续：

```python
mixed_qkv = causal_conv1d_fn(
    mixed_qkv,
    layer.conv_weights,
    layer.bias,
    activation=layer.activation,
    conv_states=conv_states,
    has_initial_state=has_initial_states,
    cache_indices=cache_indices,
    query_start_loc=query_start_loc,
    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
)
```

然后 GDN recurrent kernel 用 `ssm_states` 和 `cache_indices` 从已有 SSM state 继续递推：

```python
core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
    q=query,
    k=key,
    v=value,
    g=g,
    beta=beta,
    ssm_states=ssm_states,
    cache_indices=cache_indices,
    query_start_loc=query_start_loc,
)
```

因此 Qwen3.5 的 cache 命中不是“查已有 KV 并拼 attention mask”，而是：

```text
恢复 prefix 末尾的 conv state + SSM state
  -> 只对 suffix 做 linear attention 递推
  -> 得到和从完整 prompt 开始计算一致的输出
```

---

## 14. 为什么要按 chunk 边界 track

相关文件：

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

linear attention 的 recurrent state 不像 KV cache 那样天然每个 token 都有一个可直接保存的 K/V。prefill kernel 通常按 chunk 处理，并只在特定位置暴露可复用 state。

源码里使用 `mamba_cache_chunk_size` 决定缓存边界：

```python
@property
def mamba_cache_chunk_size(self) -> int:
    chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
    self._mamba_cache_chunk_size = max(chunk_size, self.page_size)
    return self._mamba_cache_chunk_size
```

`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend()` 会为每个请求决定本轮是否需要 track：

```python
mask = req.extend_range.length >= mamba_cache_chunk_size
mamba_track_seqlen = len(req.prefix_indices) + req.extend_range.length
mamba_track_seqlen_aligned = (
    len(req.prefix_indices)
    + (req.extend_range.length // mamba_cache_chunk_size)
    * mamba_cache_chunk_size
)
req.mamba_last_track_seqlen = mamba_track_seqlen_aligned
```

也就是说，真正写入 radix cache 的 linear state 对应的是 `mamba_last_track_seqlen`，通常是 chunk 对齐后的 prefix 长度，而不一定是请求的完整长度。

如果 prefix match 发现某个 branching point 可能需要补 mamba state，还会设置 `req.mamba_branching_seqlen`，并在当前 extend 覆盖到这个点时强制 track。

### 14.1 这些 mamba track 变量分别是什么意思

先给结论：

| 名字 | 是用户配置吗 | 作用 |
|------|--------------|------|
| `mamba_radix_cache_strategy` | 是 | 选择 mamba radix cache 策略：`auto` / `no_buffer` / `extra_buffer` / `extra_buffer_lazy` |
| `mamba_track_interval` | 是 | decode 阶段每隔多少 token 保存一次 mamba state |
| `mamba_cache_chunk_size` | 不是直接配置 | prefill/extend 阶段用于决定 mamba state 缓存边界，由模型 config 和 `page_size` 推导 |
| `mamba_track_mask` | 否 | 当前 batch 中哪些请求本轮需要 track mamba state |
| `mamba_track_indices` | 否 | 当前 batch 中每个请求要把 tracked state 写到哪个 mamba slot |
| `mamba_last_track_seqlen` | 否 | 当前 request 最近一次成功 track 的 prefix 长度 |
| ping-pong track buffer | 否 | extra buffer 策略下，每个 request 额外持有的 tracked-state slot 组 |

下面逐个解释。

#### 14.1.1 `mamba_radix_cache_strategy`

这是用户可配的 server arg：

```text
--mamba-radix-cache-strategy auto|no_buffer|extra_buffer|extra_buffer_lazy
```

旧参数名 `--mamba-scheduler-strategy` 仍作为 deprecated alias 存在，但当前源码实际写入的是：

```text
mamba_radix_cache_strategy
```

它决定 mamba radix cache 怎么保存 recurrent state：

```text
no_buffer:
  不使用额外 track buffer。
  约束更强，例如通常要求 page_size=1、关闭 overlap schedule。

extra_buffer:
  每个 request 除 active mamba slot 外，额外分配 track slot。
  track slot 用来保存可插入 radix cache 的稳定 state。

extra_buffer_lazy:
  和 extra_buffer 语义类似，但第二个 track slot 按需临时分配。
  内存占用更省，但如果边界分配失败，会跳过这次 radix cache insert。

auto:
  由 SGLang 根据模型、page_size、overlap schedule 等条件自动选择。
```

对 Qwen3.5 / Qwen3-Next 这类 hybrid linear attention 模型，理解文档后半部分时，重点看 `extra_buffer` / `extra_buffer_lazy` 路径。

#### 14.1.2 `mamba_cache_chunk_size`

这不是一个直接的用户 CLI 超参。当前源码里它是 `ServerArgs` 的 property：

```python
chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
self._mamba_cache_chunk_size = max(chunk_size, self.page_size)
```

含义是：

```text
prefill/extend 阶段，linear attention state 最好按多大的 token chunk 边界保存。
```

它主要由两部分决定：

1. 模型自己的 `hf_config.mamba_chunk_size`，如果没有则使用 FLA 默认 chunk size。
2. 服务参数 `page_size`。

最终取：

```text
mamba_cache_chunk_size = max(model_mamba_chunk_size, page_size)
```

源码还要求二者互相整除：

```python
max(chunk_size, page_size) % min(chunk_size, page_size) == 0
```

这是为了让 mamba state 的 track 边界和 KV page / radix cache 边界能对齐。

举个例子：

```text
模型 mamba_chunk_size = 256
page_size = 16

mamba_cache_chunk_size = 256
```

如果一次 extend 只算 100 token：

```text
100 < 256
本轮不 track mamba state
```

如果一次 extend 算 600 token，且 prefix 已命中 1024 token：

```text
prefix_len = 1024
extend_len = 600
mamba_cache_chunk_size = 256

可 track 到:
  1024 + floor(600 / 256) * 256
  = 1024 + 512
  = 1536

完整请求已算到:
  1024 + 600 = 1624

所以 mamba state 只安全缓存到 1536，1536..1623 不能作为 mamba prefix cache 复用。
```

#### 14.1.3 `mamba_track_interval`

这是用户可配的 server arg：

```text
--mamba-track-interval 256
```

默认值是 256。

它主要用于 decode 阶段。decode 每步生成一个 token，active mamba state 会不断被原地更新。如果每步都把 state 复制到 track buffer，开销太大；如果完全不复制，长 decode 产生的新前缀又没法被后续请求复用。

所以源码用 interval 做采样：

```python
self.mamba_track_mask = (
    self.seq_lens_cpu % mamba_track_interval == 0
)
```

含义是：

```text
当请求当前 committed sequence length 是 mamba_track_interval 的倍数时，
把 active mamba state 复制一份到 track buffer。
```

例如：

```text
mamba_track_interval = 256

decode 后长度到 256:
  track state_at_256

decode 后长度到 512:
  track state_at_512

decode 后长度到 768:
  track state_at_768
```

相关约束：

```text
如果 page_size 已设置:
  mamba_track_interval 必须能被 page_size 整除。

如果启用 speculative decoding:
  mamba_track_interval >= speculative_num_draft_tokens。
```

所以 `mamba_track_interval` 是用户可以调的；它越小，可复用边界越密，但 track/copy 开销和 mamba cache 占用可能更高。它越大，开销更低，但后续请求可复用的 prefix 边界更稀疏。

#### 14.1.4 `mamba_track_mask`

这不是用户配置，而是每个 forward batch 的运行时 tensor。

它的语义是：

```text
mamba_track_mask[i] = 当前 batch 第 i 个 request 本轮是否要保存 mamba state
```

在 prefill/extend 阶段，它大致来自：

```python
mask = req.extend_range.length >= mamba_cache_chunk_size
```

也就是说，如果这轮 extend 连一个完整 mamba cache chunk 都没覆盖到，就没必要 track。

在 decode 阶段，它来自：

```python
mamba_track_mask = (seq_lens_cpu % mamba_track_interval == 0)
```

也就是说，如果本步 decode 后的序列长度刚好到达 track interval 边界，就 track。

可以把它理解成：

```text
本轮哪些 request 需要拍一张 mamba state 快照？
```

#### 14.1.5 `mamba_track_indices`

这也不是用户配置，而是每个 forward batch 的运行时 tensor。

它的语义是：

```text
mamba_track_indices[i] = 当前 batch 第 i 个 request 的 tracked state 应该写到哪个 mamba slot
```

源码里会从每个 request 的 ping-pong track buffer 里取：

```python
req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
```

然后组成 batch tensor：

```python
batch.mamba_track_indices = ...
```

forward 里如果 `mamba_track_mask[i]` 为 true，就会把第 i 个请求的 active mamba state copy 到：

```text
mamba_track_indices[i]
```

所以：

```text
mamba_track_mask:
  控制要不要写。

mamba_track_indices:
  控制写到哪里。
```

#### 14.1.6 `mamba_last_track_seqlen`

这不是用户配置，而是 request 上的运行时字段。

它表示：

```text
当前 request 最近一次成功保存 mamba state 的 prefix 长度。
```

这个字段非常关键，因为 radix cache 插入时不能只看 token 已经算到哪里，还要看 mamba state 实际保存到哪里。

例如：

```text
请求已经算到 1624 token
但 mamba state 最近只 track 到 1536 token

req.mamba_last_track_seqlen = 1536
```

那么插入 radix cache 时，mamba component 会用：

```python
cache_len = req.mamba_last_track_seqlen
```

而不是完整 `token_ids_len = 1624`。

原因是：

```text
radix key 如果声称缓存到 1624，
但 mamba state 只对应 1536，
后续请求从 1624 继续算就会使用错误 state。
```

所以 `mamba_last_track_seqlen` 是 Qwen3.5 prefix cache 正确性的核心保护字段。

#### 14.1.7 ping-pong track buffer

这是 extra buffer 策略下的 per-request 运行时结构，不是用户直接配置的超参。

一个 request 至少有一个 active mamba slot：

```text
req.mamba_pool_idx
```

这个 slot 是 forward 真正使用的状态。prefill/decode 继续往后算时，它会不断被更新。

但是 radix cache 需要的是“某个 prefix 边界上的稳定 state”。如果直接把 active slot 插入 radix tree，后续 decode 继续覆盖 active slot，就会污染 radix cache。

所以 extra buffer 会给 request 额外准备 track slot：

```text
req.mamba_ping_pong_track_buffer[0]
req.mamba_ping_pong_track_buffer[1]
req.mamba_next_track_idx
```

直观时序：

```text
active_slot:
  当前 forward/decode 正在更新的状态。

track_buffer[0]:
  保存上一个可缓存边界的 state。

track_buffer[1]:
  保存下一个可缓存边界的 state。
```

为什么叫 ping-pong：

```text
第 1 次 track:
  写 track_buffer[0]

第 2 次 track:
  写 track_buffer[1]

第 3 次 track:
  再写 track_buffer[0]
```

这样可以避免当前正在被 radix cache 持有或准备插入的 state，被下一次 track 立刻覆盖。

`extra_buffer_lazy` 的区别是：

```text
普通 extra_buffer:
  一开始就分配完整 ping-pong track buffer。

extra_buffer_lazy:
  平时只持有一个 track slot。
  到 track boundary 时再临时分配另一个 slot。
  forward 后再释放临时 slot。
```

因此，ping-pong track buffer 是实现机制；用户真正能调的是：

```text
--mamba-radix-cache-strategy extra_buffer|extra_buffer_lazy
--mamba-track-interval N
--page-size N
```

而不是直接调 `mamba_track_mask`、`mamba_track_indices`、`mamba_last_track_seqlen` 或 ping-pong buffer 本身。

#### 14.1.8 extra_buffer 到底额外多占多少 mamba slot

先区分两类 slot：

```text
active mamba slot:
  request 当前 forward/decode 真正读写的 state。

track mamba slot:
  extra_buffer 为 prefix cache 保存稳定边界 state 的额外 slot。
```

每个运行中的 request 本来就需要：

```text
1 个 active mamba slot
```

开启 `extra_buffer` 后，会额外分配 track slot。源码里数量由：

```python
self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
```

决定。

所以常见情况是：

```text
extra_buffer + overlap schedule 开启:
  每个 request:
    active slot      = 1
    ping-pong track  = 2
    合计运行中占用   = 3 个 mamba slots

extra_buffer + --disable-overlap-schedule:
  每个 request:
    active slot      = 1
    track slot       = 1
    合计运行中占用   = 2 个 mamba slots

extra_buffer_lazy + overlap schedule:
  每个 request:
    active slot      = 1
    常态 track slot  = 1
    边界处可能临时再分配 1 个 track slot
    合计常态占用     = 2 个 mamba slots
    边界瞬时占用     = 3 个 mamba slots
```

例如有 32 条正在运行的请求，并且使用：

```text
mamba_radix_cache_strategy = extra_buffer
overlap schedule = enabled
```

那么运行中 request 至少占：

```text
active slots:
  32 * 1 = 32

extra ping-pong track slots:
  32 * 2 = 64

合计:
  32 + 64 = 96 个 mamba slots
```

这里的额外 64 个 slot 不是为了模型计算本身必须有，而是为了 prefix cache 的正确性：

```text
active slot 会不断被后续 forward/decode 覆盖。
radix cache 需要的是某个 prefix 边界上的稳定 state。
```

所以需要把 active state 在 cacheable boundary 复制到 track slot：

```text
边界 256:
  active state -> track slot 0

边界 512:
  active state -> track slot 1

边界 768:
  active state -> track slot 0
```

overlap schedule 下，forward、后处理、cache insert、下一轮调度可能流水重叠。某个 track slot 刚保存完边界 state，可能还在等待被 radix cache 插入或接管，此时下一轮又到了新的 track 边界，不能立刻覆盖同一个 slot。因此需要两个 track slot ping-pong。

还要注意：上面的 96 只是“32 个运行中 request 的基础 mamba slot 占用”。mamba pool 里还可能有：

```text
已经插入 radix cache、由 tree cache 持有的 cached mamba slots
等待释放或复用的 slots
lazy 模式边界临时 slots
```

所以总的 `max_mamba_cache_size` 通常要比 `running_requests * 每请求 slot 数` 更大。SGLang 源码里会用内部 ratio 给 mamba pool 预留容量：

```python
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY = 1
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1
```

直观理解：

```text
no_buffer:
  基础 ratio = 3

extra_buffer + overlap:
  ratio = 3 + 2 = 5

extra_buffer_lazy + overlap:
  ratio = 3 + 1 = 4

extra_buffer + no overlap:
  ratio = 3 + 1 = 4
```

这个 ratio 不是说每个 request 永远实际占这么多 slot，而是 SGLang 估算 mamba pool 总容量时，为 active state、radix cached state、track buffer 和并发调度预留空间的规则。

用户通常不需要直接配置“每个 request 几个 extra slot”。能影响这部分的主要参数是：

```text
--mamba-radix-cache-strategy
--disable-overlap-schedule
--max-mamba-cache-size
--mamba-full-memory-ratio
```

---

## 15. SSM state 的取法：aligned 和 unaligned 不一样

相关文件：

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

`_init_track_ssm_indices()` 的注释已经把这个问题解释得很清楚：

- kernel 输出 `h`：chunk 边界上的 intermediate recurrent state。
- kernel 输出 `last_recurrent_state`：本轮 prefill 末尾的 final state。

如果目标缓存长度刚好在 chunk 边界：

```text
cache state <- last_recurrent_state
```

如果目标缓存长度不是最后位置，而是中间某个 chunk 边界：

```text
cache state <- h[chunk_index]
```

源码里的判断：

```python
is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
track_ssm_final_dst = dst_masked[is_aligned]

track_ssm_h_src = offset_masked[not_aligned] + (
    lens_masked[not_aligned] // mamba_cache_chunk_size
)
track_ssm_h_dst = dst_masked[not_aligned]
```

最终 `_track_mamba_state_extend()` 负责写入：

```python
if forward_metadata.track_ssm_h_src.numel() > 0:
    ssm_states[forward_metadata.track_ssm_h_dst] = h[
        forward_metadata.track_ssm_h_src
    ]

if forward_metadata.track_ssm_final_src.numel() > 0:
    ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
        forward_metadata.track_ssm_final_src
    ]
```

这个设计避免了一个错误：如果目标缓存点是 chunk 边界，但本轮 prefill 实际继续算了更多 token，不能直接拿最后的 recurrent state，否则 cached prefix 会包含多算的 suffix 信息。

---

## 16. Conv state 的 track

相关文件：

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/layers/attention/linear/gdn_backend.py`

除了 SSM state，linear attention 前面还有 causal conv，所以 cache 还要保存 prefix 末尾的 conv window。

`_init_track_conv_indices()` 会计算需要保存的最后 `conv_state_len` 个位置：

```python
conv_state_len = self.conv_states_shape[-1]
lens_to_track = forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
```

在 `GDNAttnBackend.forward_extend()` 里，如果需要 track，会把这些 token 对应的 mixed_qkv 写入 track slot 的 conv state：

```python
if forward_metadata.has_mamba_track_mask:
    mixed_qkv_to_track = mixed_qkv[
        :, forward_metadata.track_conv_indices
    ].transpose(0, 1)
    conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track
```

因此，一个可复用的 mamba cache slot 同时包含：

```text
conv_states[slot]  # prefix 末尾 conv window
ssm_states[slot]   # prefix 对应 recurrent state
```

---

## 17. Extra buffer / ping-pong buffer

相关文件：

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

当前源码里，extra buffer 由 `mamba_radix_cache_strategy` 控制：

```python
def enable_mamba_extra_buffer(self) -> bool:
    return (
        self.disable_radix_cache is False
        and self.mamba_radix_cache_strategy in ("extra_buffer", "extra_buffer_lazy")
    )
```

开启 extra buffer 后，请求除了 active mamba slot 外，还会有 ping-pong track buffer：

```text
req.mamba_pool_idx                    # 当前请求实际 forward 用的 active state
req.mamba_ping_pong_track_buffer[0/1] # 用于保存 prefix cache state 的额外 slot
req.mamba_next_track_idx              # 下一次写入哪个 track slot
```

这样做的原因是：active state 会随着请求继续 decode 不断变化。如果直接把 active slot 插入 radix cache，后续 forward 可能继续覆盖它，导致 radix cache 中的 state 被污染。

extra buffer 的策略是：

1. active slot 继续服务当前请求。
2. track slot 保存某个可缓存边界上的 state。
3. 插入 radix cache 时，把 track slot 作为 `mamba_value`。
4. 请求继续运行时切换到另一个 track slot。

`extra_buffer_lazy` 则只在边界临时分配第二个 slot，减少常驻 mamba slot 占用。如果边界临时分配失败，源码会设置 `req.mamba_lazy_is_insert = False`，结束时跳过 radix cache insert，避免插入错误状态。

---

## 18. 插入 radix cache

相关文件：

- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/mem_cache/unified_cache_components/mamba_component.py`

请求结束或 chunked prefill 中间缓存 unfinished req 时，`UnifiedRadixCache` 会让每个 component 准备自己的 insert data：

```python
for comp in self._components_tuple:
    cl = comp.prepare_for_caching_req(
        req=req,
        insert_params=insert_params,
        token_ids_len=len(token_ids),
        is_finished=True/False,
    )
    if cl is not None:
        effective_cache_len = min(effective_cache_len, cl)
```

`MambaComponent.prepare_for_caching_req()` 会决定 mamba component 能缓存到哪里：

```python
cache_len = (
    req.mamba_last_track_seqlen
    if self.enable_mamba_extra_buffer
    else token_ids_len
)
```

如果启用 extra buffer，插入的长度不是完整 `token_ids_len`，而是最后一次 track 成功的 `mamba_last_track_seqlen`。

然后把 mamba state slot 填入：

```python
insert_params.mamba_value = mamba_value
```

最终 radix node 同时拥有：

```text
ComponentType.FULL.value   -> full attention KV pages
ComponentType.MAMBA.value  -> linear attention state slot
```

后续 prefix match 时，只有 FULL 和 MAMBA component 都满足要求，这个 prefix 才能作为完整的 device match。

---

## 19. Decode 阶段也会持续维护 cacheable state

相关文件：

- `python/sglang/srt/layers/attention/linear/gdn_backend.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

decode 每步只处理一个 token。`GDNAttnBackend.forward_decode()` 会：

1. 从 `mamba_cache_indices` 找到 active mamba slot。
2. `causal_conv1d_update()` 原地更新 conv state。
3. GDN decode kernel 原地更新 SSM state。
4. 如果本步达到 track 边界，复制 active state 到 track slot。

decode 准备阶段：

```python
if get_global_server_args().enable_mamba_extra_buffer():
    mamba_track_interval = get_global_server_args().mamba_track_interval
    self.mamba_track_mask = (
        (self.seq_lens_cpu % mamba_track_interval == 0)
        .pin_memory()
        .to(device=self.device, non_blocking=True)
    )
```

forward 里：

```python
self._track_mamba_state_decode(
    forward_batch, conv_states, ssm_states, cache_indices
)
```

`_track_mamba_state_decode()` 会在 `mamba_track_mask` 为 true 的请求上，把 active slot 的 conv/SSM state copy 到 track slot。

batch result 处理阶段 `_mamba_prefix_cache_update()` 会更新：

- `req.mamba_last_track_seqlen`
- `req.mamba_next_track_idx`
- lazy 模式下的临时 slot 回收

因此，不只是 prefill 会生成可复用 state，长 decode 请求在经过 `mamba_track_interval` 边界时也会留下新的 prefix cache state。

---

## 20. 和普通 KV prefix cache 的差异总结


| 维度          | 普通 full attention               | Qwen3.5 linear attention                 |
| ----------- | ------------------------------- | ---------------------------------------- |
| 缓存内容        | 每个 token 的 K/V                  | conv window + SSM recurrent state        |
| 缓存粒度        | token/page 粒度                   | chunk / track interval 边界                |
| prefix 命中后  | 直接复用 KV，suffix attend prefix KV | 恢复 recurrent state，只对 suffix 继续递推        |
| radix value | KV page indices                 | KV page indices + mamba state slot       |
| 状态更新        | append KV                       | 原地更新 active state                        |
| 状态写回        | KV pages 插入 radix               | track slot 插入 radix                      |
| 风险点         | KV 生命周期管理                       | active state 被继续覆盖，需要 extra buffer / COW |


---

## 21. 一条完整时序

```text
请求进入
  -> Req.init_next_round_input()
  -> tree_cache.match_prefix()
  -> FULL component 返回 prefix_indices
  -> MAMBA component 返回/准备 mamba state
  -> req.mamba_cow_src_index = cached_mamba_slot

构建 prefill batch
  -> HybridReqToTokenPool.alloc()
  -> req.mamba_pool_idx = active_slot
  -> ScheduleBatch 准备 mamba_track_mask / indices / seqlens
  -> ForwardBatch 携带 mamba_cow_src/dst 和 mamba_track_*

forward 前
  -> _execute_deferred_mamba_cow_and_clear()
  -> cached_mamba_slot copy 到 active_slot

GDN forward_extend
  -> causal_conv1d_fn(... has_initial_state = extend_prefix_lens > 0 ...)
  -> GDN recurrent kernel 从 ssm_states[active_slot] 继续算
  -> 如果需要 track，写 conv state 和 SSM state 到 track_slot

cache insert
  -> MambaComponent.prepare_for_caching_req()
  -> effective_cache_len = req.mamba_last_track_seqlen
  -> insert_params.mamba_value = track_slot
  -> UnifiedRadixCache.insert()

后续请求
  -> prefix match 同时命中 KV 和 mamba state
  -> 重复上述流程，只计算 suffix
```

---

## 22. 结论

Qwen3.5 的 linear attention prefill cache 是专门实现的，不是普通 KV cache 的自然副产物。

它的核心设计是：

1. full attention 层继续使用普通 KV prefix cache。
2. linear attention 层额外维护 per-request recurrent state。
3. radix node 同时保存 KV pages 和 mamba state slot。
4. prefix 命中后通过 COW 恢复 mamba state。
5. prefill / decode 到特定 chunk 或 interval 边界时，把 conv state + SSM state track 到额外 slot。
6. 插入 radix cache 时使用 track slot，避免 active slot 被后续 decode 覆盖。

因此，对于 Qwen3.5 这类 hybrid linear attention 模型，prefix cache 的正确性依赖两件事同时成立：

- KV cache 命中的是 full attention 层所需的历史 K/V。
- mamba / linear state 命中的是 linear attention 层在同一 prefix 边界上的 conv + recurrent 状态。

只命中 KV 而没有对应 mamba state，不能安全地跳过 prefix。

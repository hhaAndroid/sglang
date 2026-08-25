# SGLang KV Cache Eviction 源码分析

本文基于当前工作区源码分析 SGLang 的 KV cache 淘汰逻辑，重点回答三个问题：

1. KV cache 什么时候触发淘汰？
2. 现在有哪些 eviction policy？
3. 淘汰是搬到 CPU，还是直接删掉？

## 总结

结论先放前面：

- 默认普通 `RadixCache` 的淘汰不是搬到 CPU，而是把 radix tree 中可淘汰叶子节点对应的 device KV index 释放回 GPU KV pool，并从 radix tree 删除节点。
- 只有启用 `--enable-hierarchical-cache` 的 HiCache 路径，淘汰才可能变成 GPU -> CPU demotion。也就是说 device KV 被释放，但节点仍留在树上，并带着 `host_value`，后续命中时可以 load back 到 GPU。
- HiCache 下如果节点没有 CPU backup：
  - `write_back` 策略会先写到 host，再从 GPU evict。
  - 非 `write_back` 路径下，没有 backup 的叶子会直接从树中删除。
- CPU/host 侧也有二级淘汰：当 host pool 不够时，会把已经只在 host 上的叶子彻底释放，并从树中删除。
- `--disable-radix-cache` 后使用 `ChunkCache`，它没有 prefix cache eviction policy；请求结束时直接释放本请求占用的 KV。
- CLI 公开的 `--radix-eviction-policy` 目前是 `lru`, `lfu`, `slru`, `priority`。底层 `evict_policy.py` 还实现了 `fifo`, `mru`, `filo`，但默认 server args choices 没暴露它们。

## 触发点

KV cache 分配前会尝试从 tree cache 里淘汰可回收 prefix cache。

入口在 `python/sglang/srt/mem_cache/common.py`：

- `alloc_token_slots()` 先调用 `evict_from_tree_cache(tree_cache, num_tokens)`，再 `allocator.alloc(num_tokens)`。
- `evict_from_tree_cache()` 对普通 allocator 检查 `allocator.available_size() < num_tokens`，不足才调用 `tree_cache.evict(EvictParams(num_tokens=num_tokens))`。
- 对 SWA hybrid allocator，会分别看 full pool 和 SWA pool，不够多少分别传 `num_tokens` 和 `swa_num_tokens`。

对应源码：

- `common.py:272-284`
- `common.py:300-323`

decode 阶段还有一层保护：`ScheduleBatch.check_decode_mem()` 也会先尝试 tree cache eviction；如果仍然不够，scheduler 才 retract 正在 decode 的请求。这个 retraction 不是普通 prefix cache eviction，而是抢占活跃请求来腾内存。

## 普通 RadixCache 怎么淘汰

普通路径是 `python/sglang/srt/mem_cache/radix_cache.py` 的 `RadixCache.evict()`。

核心流程：

1. `evictable_leaves` 里只放可淘汰叶子。
2. 每个候选节点按当前 `eviction_strategy.get_priority(node)` 放入 heap。
3. 不断 pop 最低优先级节点。
4. 对节点 `x` 调用 `self.token_to_kv_pool_allocator.free(x.value)`。
5. `num_evicted += len(x.value)`。
6. `_delete_leaf(x)` 从 radix tree 中删除该叶子。
7. 如果父节点因此也变成无锁叶子，把父节点继续放回 heap。

关键源码是 `radix_cache.py:568-595`。

这里的 `free(x.value)` 只是把 token index 归还给 allocator。以普通 `TokenToKVPoolAllocator` 为例，`free()` 把 index concat 回 `free_pages` 或 `release_pages`，并不复制到 CPU。对应 `allocator/token.py:55-69`。

所以普通 `RadixCache` 的 eviction 语义是：释放 GPU KV slot + 删除 radix 节点。没有 host backup，也没有 CPU load back。

## 哪些节点可淘汰

普通 `RadixCache` 通过 lock ref 保护正在被请求使用的 prefix。

- `inc_lock_ref(node)` 从命中节点一路到 root 增加 `lock_ref`。如果节点之前 `lock_ref == 0`，它会从 evictable 转为 protected。
- `dec_lock_ref(node)` 反向释放引用。如果节点从 `lock_ref == 1` 降到 0，就重新计入 `evictable_size_`。
- `_update_leaf_status(node)` 只有在节点未 evicted、`lock_ref == 0`、并且没有未 evicted child 时，才把它加入 `evictable_leaves`。

这意味着 SGLang 普通 radix tree 只淘汰无活跃引用的叶子，再逐步向上合并删除。

相关源码：

- `radix_cache.py:597-636`
- `radix_cache.py:760-778`

## Eviction policy

策略定义在 `python/sglang/srt/mem_cache/evict_policy.py`，底层支持：

- `lru`: priority 是 `node.last_access_time`，越老越先淘汰。
- `lfu`: priority 是 `(node.hit_count, node.last_access_time)`，命中少的先淘汰，同命中数下 LRU。
- `fifo`: priority 是 `node.creation_time`，越早创建越先淘汰。
- `mru`: priority 是 `-node.last_access_time`，最近访问的先淘汰。
- `filo`: priority 是 `-node.creation_time`，最近创建的先淘汰。
- `priority`: priority 是 `(node.priority, node.last_access_time)`，低 priority 先淘汰，同 priority 下 LRU。
- `slru`: hit_count 小于阈值 2 的节点属于 probationary segment，先淘汰；hit_count >= 2 的 protected segment 后淘汰；segment 内按 LRU。

策略工厂在 `python/sglang/srt/mem_cache/utils.py` 的 `_EVICTION_POLICY_FACTORIES`。

但是 CLI 暴露的 choices 在 `python/sglang/srt/server_args.py`：

```python
RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]
```

也就是说，正常命令行可直接选的是 `lru`, `lfu`, `slru`, `priority`。`fifo/mru/filo` 是底层实现有，但默认 CLI choices 没列出来。

相关源码：

- `evict_policy.py:16-65`
- `utils.py:31-39`
- `server_args.py:281`
- `server_args.py:761-772`

## hit_count 和 priority 从哪里来

普通 radix cache 插入时会更新节点元信息：

- `_insert_helper()` 命中已有节点时调用 `_inc_hit_count()`。
- 新节点创建时也会 `_inc_hit_count(new_node, chunked)`。
- chunked prefill 传 `chunked=True` 时不增加 hit_count，避免一个 chunked request 自己反复命中自己制造出来的前缀导致 hit_count 膨胀。
- `priority` 来自 request 上的 `req.priority`，插入时用 `max(node.priority, priority)` 沿路径传播。

相关源码：

- `radix_cache.py:687-753`
- `radix_cache.py:668-674`
- `radix_cache.py:468-472`
- `radix_cache.py:509-515`

## 请求结束时发生什么

普通 `RadixCache.cache_finished_req()` 的语义：

- 如果 radix cache 被禁用，直接释放请求的 KV indices。
- 如果启用 radix cache：
  - 把 page-aligned 的 prompt + output KV indices 插入 radix tree。
  - 对已经存在于 tree 中的重复 prefix，释放本请求重复持有的 KV indices。
  - 对未对齐尾巴直接释放。
  - 对请求之前 lock 的 last_node 调 `dec_lock_ref()`，使 prefix 重新可淘汰。

这说明请求结束不是立刻删除全部 KV；启用 radix cache 时，会把可复用 prefix 留在 tree cache 中，直到未来内存紧张时被 eviction policy 淘汰。

相关源码：`radix_cache.py:442-491`。

## ChunkCache: 关闭 radix cache 时

当 `--disable-radix-cache` 且 chunked prefill 场景需要一个 cache 对象时，默认 factory 会创建 `ChunkCache`。

`ChunkCache` 的行为很简单：

- `match_prefix()` 永远返回空命中。
- `insert()` 是 no-op。
- `evict()` 是 no-op。
- `cache_finished_req()` 直接释放这个请求已提交的 KV indices。

所以这一路没有 prefix cache eviction policy，也不会搬 CPU。它只按请求生命周期释放。

相关源码：

- `registry.py:75-82`
- `chunk_cache.py:67-95`

## HiCache: 什么时候搬到 CPU

开启 `--enable-hierarchical-cache` 后，普通 MHA/MLA 会走 `HiRadixCache`；hybrid SWA/SSM 或 unified radix tree 会走 `UnifiedRadixCache.init_hicache()`。

server args 里 HiCache 相关关键参数：

- `enable_hierarchical_cache`
- `hicache_ratio`
- `hicache_size`
- `hicache_write_policy`: `write_back`, `write_through`, `write_through_selective`
- `hicache_io_backend`: `direct`, `kernel`, `kernel_ascend`
- `hicache_mem_layout`
- `hicache_storage_backend`: file/mooncake/hf3fs/nixl/aibrix 等 L3 storage

相关源码：`server_args.py:1861-1900`。

### HiRadixCache 的 device eviction

`HiRadixCache.evict()` 与普通 `RadixCache.evict()` 很像，也是按 eviction policy 对 `evictable_leaves` 建 heap。但处理节点时分三种情况：

1. 节点没有 backup 且 write policy 是 `write_back`：
  - 先 `write_backup(x, write_back=True)` 写 host。
  - 写完后 `_evict_backuped(node)`。
2. 节点没有 backup 且不是 write_back：
  - `_evict_regular(x)`，直接释放 device KV，并删除 tree leaf。
3. 节点已经 backuped：
  - `_evict_backuped(x)`，释放 device KV，但保留树节点和 `host_value`。

关键源码：

- `hiradix_cache.py:1035-1081`
- `hiradix_cache.py:1083-1096`
- `hiradix_cache.py:1098-1106`

`_evict_backuped()` 里的语义非常明确：

- `cache_controller.evict_device(node.value)` 释放 GPU/device KV。
- `node.value = None` 标记 device KV 不在了。
- 节点不从树里删除，host metadata 仍在。

所以 HiCache 下，已备份节点的 device eviction 是 GPU -> CPU demotion，不是彻底删除。

### HiRadixCache 的 host eviction

Host pool 也可能满。`HiRadixCache.evict_host()` 会从 `evictable_host_leaves` 里按同样 eviction policy 选 host-only leaf：

- 只处理 `x.evicted == True` 的节点，也就是 GPU 已经没有了。
- 跳过 `host_ref_counter > 0` 的节点。
- 调 `cache_controller.evict_host(x.host_value)` 释放 host KV。
- 从 parent children 删除该节点。

这才是 CPU/host 侧的彻底删除。

相关源码：`hiradix_cache.py:1108-1142`。

### HiCache load back

命中 host-only 节点时，`load_back()` 会沿着 evicted 祖先链收集 `host_value`，如果满足阈值和 quota，就调用 `cache_controller.load()` 分配 device KV 并从 host 读回。

关键点：

- 小于 `load_back_threshold` 默认 10 token 会跳过 load back。
- 如果 GPU 内存不够，会先 `self.evict(EvictParams(num_tokens=len(host_indices)))` 尝试腾 device KV。
- 成功后给这些 node 重新设置 `node.value = device_indices[...]`。

相关源码：`hiradix_cache.py:1143-1210`。

## UnifiedRadixCache 和 SWA 的特殊情况

### UnifiedRadixCache

Unified radix tree 把 FULL/SWA/MAMBA 等 component 放在同一棵树中。其 `evict()` 会遍历每个 component，让 component 自己 `drive_eviction()`。

相关源码：`unified_radix_cache.py:604-624`。

HiCache 语义和 `HiRadixCache` 类似：

- `_evict_to_host()` 明确是 GPU -> CPU demotion：释放 device resources，节点留在树里。
- `_evict_device_leaf()`：
  - backuped: `_evict_to_host()`
  - not backuped + write_back: 先 `write_backup()`，再 `_evict_to_host()`
  - not backuped + write_through: cascade evict all components，并删除 leaf

相关源码：`unified_radix_cache.py:1464-1518`。

### SWARadixCache

SWA hybrid cache 有 full attention KV pool 和 SWA KV pool 两类资源：

- full KV eviction 主要删 leaf，释放 full KV，并且如果 SWA 还没 tombstone，也连带释放 SWA。
- SWA eviction 可以只释放 SWA pool，把节点标记为 `swa_tombstone`，full KV 仍可能保留。这是 sliding window attention 的特殊内存回收，不等价于普通 prefix cache 的整块删除。
- SWA 版本用专门的 `LRUList` 维护 full 和 SWA 两套 LRU，没有走 `evict_policy.py` 那套 `lru/lfu/slru/priority` heap 策略。

相关源码：

- `swa_radix_cache.py:563-667`
- `swa_radix_cache.py:740-818`

## 直接回答

“淘汰是全部移到 CPU 还是真的删掉？”

准确说分情况：

1. 普通 `RadixCache`：真的删掉 tree entry，并释放 GPU KV slot；不移到 CPU。
2. `ChunkCache`/禁用 radix cache：没有 prefix cache eviction；请求结束直接释放 KV。
3. HiCache 已 backup 的节点：先从 GPU/device evict，保留 CPU/host backup；这是移到 CPU 后的 device demotion。
4. HiCache 未 backup 的节点：
  - `write_back`: eviction 时先写 CPU，再释放 GPU。
  - 非 `write_back`: 直接删除。
5. HiCache host pool 满时：host-only 节点会从 CPU/host 彻底删除，树节点也删掉。
6. SWA/hybrid：可能只释放 SWA pool 并 tombstone，full KV 还在；也可能 full+SWA 一起释放。

所以不能笼统说 SGLang eviction 都会 offload 到 CPU。默认不是；只有 HiCache 分层缓存路径才会有 GPU -> CPU -> storage 这种多层语义。

## 补充：开 prefix cache 不等于开 offload

这里有一个容易混淆的点：**prefix cache 和 offload 是两件事**。

默认不开 `--disable-radix-cache` 时，SGLang 会启用 radix prefix cache。它的含义是：

1. 请求运行时产生的 KV 在 GPU KV pool 里。
2. 请求结束，或者 chunked prefill 中间阶段，SGLang 会把 page-aligned、未来可能复用的 prefix KV 插入 radix tree。
3. 这些 prefix cache 继续占用 GPU KV slot。
4. 后续请求如果有相同 prefix，可以直接复用 GPU 上的 KV，减少 prefill 计算。
5. 如果 GPU KV pool 不够，scheduler 会从 radix tree 中选择无活跃引用的 prefix cache 淘汰。
6. 默认淘汰语义是释放 GPU KV slot，并删除 radix tree 节点；不会自动搬到 CPU。

所以默认 radix cache 更准确地说是 **GPU-resident prefix cache**。它会尽量保留可能复用的 KV，但保留范围受 GPU KV pool 容量限制。一旦显存紧张，它会牺牲未来 prefix cache 命中率，换取当前请求继续运行。

三种模式可以这样区分：

- `--disable-radix-cache`：不保存 prefix cache。请求结束后 KV 基本按生命周期释放。
- 默认 radix cache：保存 prefix cache 在 GPU。显存不够时，按 eviction policy 直接删掉无活跃引用的 prefix cache。
- `--enable-hierarchical-cache`：引入 CPU/host 层。已 backup 的节点在 device eviction 时可以 GPU -> CPU demotion，后续命中时再 load back。

因此，“开了 prefix cache 就一定会保存可能复用的 cache”只在资源允许时成立。默认实现没有无限保留，也不会在 GPU 不够时自动 offload。要获得 offload 语义，需要显式启用 HiCache。

## Agent workflow metadata 为什么有价值

从这个角度看，agent-aware KV cache 的价值就比较清楚了。

默认 radix prefix cache 已经会缓存可能复用的 KV，但它在显存紧张时必须决定删谁。现有策略主要依赖比较通用的局部信号：

- 最近访问时间：LRU/MRU。
- 命中次数：LFU/SLRU。
- 请求 priority：priority policy。
- 节点是否仍有活跃引用：lock ref。

这些信号不知道一个请求属于哪个 agent workflow，也不知道 planner/tool/worker step 之间的父子关系、后续步骤是否很快会复用同一段 prefix、tool call 后是否会回到同一上下文、某段 prefix 是否有更强的短期保留价值。

所以 agent workflow metadata 的作用不是“让 prefix cache 才开始存在”，也不是“默认把 cache offload 到 CPU”。更准确地说，它可以给服务端更完整的未来复用信号，让 eviction policy 在 GPU KV pool 不够时更聪明地决定：

- 哪些 prefix cache 应该优先保留。
- 哪些 cache 虽然最近没访问，但马上可能被 child step 或 tool-return step 复用。
- 哪些 workflow 已经接近结束，相关 cache 可以更早释放。
- 哪些 shared prefix 属于多个 step/agent，应避免被普通 LRU 过早淘汰。

因此，在 agentic workload 下，agent-aware 信息主要提升的是 **cache retention decision**：显存紧张时保谁、删谁。它和 HiCache/offload 是正交能力；即使不启用 offload，服务端也可以用这些 metadata 提高 GPU prefix cache 的命中率。如果同时启用 HiCache，这些信息还可能进一步影响哪些节点值得 demote 到 host、哪些值得从 host load back。
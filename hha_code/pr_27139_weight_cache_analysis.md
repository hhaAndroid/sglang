# SGLang PR #27139：Weight Cache Daemon 从使用到实现

PR：[https://github.com/sgl-project/sglang/pull/27139](https://github.com/sgl-project/sglang/pull/27139)

## 0. 后续实现需要持续关注哪些 RFC / Roadmap

截至 2026-08-15，这条技术路线最值得持续关注的是下面几个入口：

| 链接 | 当前定位 | 应该重点关注什么 |
|---|---|---|
| [#27052 Fast Recovery RFC](https://github.com/sgl-project/sglang/issues/27052) | 最初的总目标，目前 issue 已关闭并标记 inactive | 为什么需要快速恢复；权重缓存、CUDA Graph 恢复、Python/NCCL 初始化优化等完整启动路径 |
| [#27139 Weight Cache PR](https://github.com/sgl-project/sglang/pull/27139) | 已合入的基础实现，也是本文主要分析对象 | daemon 长期持有 post-quantized 权重，Engine 通过传统 CUDA IPC 快速映射 |
| [#27310 GMS / VMM RFC](https://github.com/sgl-project/sglang/issues/27310) | 当前仍为 open RFC | VMM transport、稳定 VA、daemon 故障隔离、GMS shared owner、RL 权重更新协议 |
| [#33522 Weight Cache Roadmap](https://github.com/sgl-project/sglang/issues/33522) | 当前仍为 open，最适合持续跟踪落地进度 | VMM FD、weight update、共卡/非共卡 RL、多实例、跨 GPU、模型与量化覆盖、运维可靠性 |
| [#33279 Weight Daemon abstraction PR](https://github.com/sgl-project/sglang/pull/33279) | 当前仍为 open 的相关实现 PR | Weight Cache transport/daemon 抽象；Roadmap 将其列为 `vmm_fd` backend 的相关实现入口 |

如果只收藏一个后续 tracker，优先看 **#33522**；如果要理解设计原因和最终架构，则重点读 **#27310**。

### 0.1 特别注意：这里存在两套“Phase 1 / Phase 2”编号

这是最容易看错的地方。

#### 第一套：Fast Recovery 总路线的 phase

#33522 将 #27139 称为已经落地的 **Phase 1**：

```text
Fast Recovery Phase 1（已经合入 #27139）
  -> daemon 持有 GPU 权重
  -> traditional CUDA IPC transport
  -> Engine 重启时跳过 checkpoint load
```

#33522 后面还把 CUDA Graph serialization 称为 Fast Recovery Phase 2，把 kernel warmup、server/distributed init 等称为后续 phase。这套编号描述的是“整个 Engine 启动路径如何逐步缩短”。

#### 第二套：#27310 自己提出的两阶段演进

#27310 是在 #27139 **已经合入以后**重写的。它把 Weight Cache 从传统 CUDA IPC 演进到 GMS 分成另外两个阶段：

```text
当前 baseline（已经合入 #27139）
  owner  = daemon
  writer = daemon
  transport = traditional CUDA IPC

        │
        ▼

#27310 Phase 1：只替换 transport（尚未合入）
  owner  = daemon
  writer = daemon
  transport = VMM shareable FD
  不引入 GMS，不改变 daemon 加载模型的架构

        │
        ▼

#27310 Phase 2：shared ownership / GMS（尚未合入）
  owner  = GMS
  writer = daemon 或 Engine
  transport = VMM shareable FD
```

所以针对“目前合入的是第一阶段，GMS 是第二阶段吗”这个问题，准确答案是：

- **按 Fast Recovery Roadmap #33522 的叫法：是。** 已合入的 #27139 被称为 Phase 1。
- **按 GMS RFC #27310 自己的叫法：不是。** #27139 只是 baseline；它的 Phase 1 是尚未合入的 VMM FD transport swap，GMS 才是它的 Phase 2。

也就是说，当前实际落地进度可以记成：

```text
传统 CUDA IPC Weight Cache：已合入
VMM FD transport：          RFC/实现推进中，尚未合入
GMS shared owner：          RFC 第二阶段，尚未合入
```

这里的“尚未合入”是截至 2026-08-15 的状态，后续应以 #33522、#27310 和关联 PR 的最新状态为准。

## 1. 先看怎么用

这个功能需要两类进程：

1. 一个长期运行的 Weight Cache Daemon；
2. 一个可以反复启动、退出和重启的 SGLang engine。

最重要的部署关系是：

```text
进程组 A：Weight Cache Daemon launcher
    ├─ GPU 0 daemon
    ├─ GPU 1 daemon
    ├─ GPU 2 daemon
    └─ GPU 3 daemon

进程组 B：SGLang Engine
    ├─ GPU 0 scheduler
    ├─ GPU 1 scheduler
    ├─ GPU 2 scheduler
    └─ GPU 3 scheduler
```

Daemon launcher 必须独立于 engine 运行。engine 重启时，只重启进程组 B，不能把进程组 A 一起杀掉。

### 1.1 第一步：启动 standalone daemon

以 TP=4 为例：

```bash
python -m sglang.srt.weight_cache.daemon \
    --model-path /path/to/model \
    --tp-size 4 \
    --load-format auto \
    --dtype auto
```

如果模型使用 PR 当前支持的 block-wise FP8，可以增加：

```bash
--quantization fp8
```

这个命令会：

1. 启动 4 个 daemon，每张 GPU 一个；
2. 让 4 个 daemon 建立 distributed/model-parallel group；
3. 每个 daemon 从磁盘加载自己的 TP shard；
4. 完成量化和权重后处理；
5. 在 GPU 中一直持有权重；
6. 等待 engine 连接。

launcher 不会在加载完成后退出，而是继续运行并监控所有 daemon。

### 1.2 第二步：用 client 模式启动 engine

等待 daemon 全部 ready 后，启动 SGLang：

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp-size 4 \
    --weight-cache-mode client
```

这个 engine 不再从磁盘加载权重，而是连接前面 4 个 daemon：

```text
Engine rank 0 -> daemon rank 0 -> GPU 0 的 weight shard
Engine rank 1 -> daemon rank 1 -> GPU 1 的 weight shard
Engine rank 2 -> daemon rank 2 -> GPU 2 的 weight shard
Engine rank 3 -> daemon rank 3 -> GPU 3 的 weight shard
```



### 1.3 第三步：重启 engine

只终止 SGLang engine，不终止 daemon launcher。

然后再次执行相同的 client 命令：

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --tp-size 4 \
    --weight-cache-mode client
```

新 engine 会重新映射 daemon 中仍然存在的 GPU 权重，所以不需要重新读取 checkpoint。

### 1.4 不要混淆 `daemon` 模式

SGLang 还提供：

```text
--weight-cache-mode daemon
```

这个模式是让 engine 主进程自动创建并管理 daemon。随后 scheduler 仍然通过 CUDA IPC 映射 daemon 持有的权重，但是这些 daemon 是 engine 的 child process，与 engine 同生共死：

```text
启动 engine
  -> engine 创建 daemon
  -> daemon 从磁盘加载权重
  -> scheduler 通过 IPC 映射权重

退出 engine
  -> daemon 一起退出
  -> GPU 权重被释放
```

所以，`daemon` 模式本身不能加速下一次 engine 启动。第一次启动不仅仍要完整加载权重，还多了一次 daemon 与 scheduler 之间的 IPC 建连和映射。

它当前更接近一个**一体化启动和验证模式**，主要价值是：

- 不用手动运行 standalone daemon，一条命令即可启动完整的 weight-cache 链路；
- 方便开发、调试和 E2E 测试 CUDA IPC 权重共享；
- 为未来可能支持的“保留 engine 主进程、只重建 scheduler worker”提供基础设施。

因此可以近似记成：


| 模式       | 当前主要用途                          |
| -------- | ------------------------------- |
| `daemon` | 一体化启动、开发调试、功能验证；没有跨 engine 重启收益 |
| `client` | 连接独立且长期运行的 daemon，用于生产快速恢复      |


因此真正的快速重启方式是：

```text
standalone daemon + --weight-cache-mode client
```

而不是单独使用 `--weight-cache-mode daemon`。

## 2. 再看一次完整流程

可以把整个功能理解成：**一次冷加载，换取后续多次复用。**

### 2.1 第一次启动 daemon：仍然需要正常加载权重

```text
Checkpoint
    │
    ├─ 磁盘读取
    ├─ TP/PP sharding
    ├─ 搬运到 GPU
    ├─ quantization
    └─ process_weights_after_loading
            │
            ▼
Weight Cache Daemon 持有最终 GPU 权重
```

第一次启动 daemon 不会凭空变快。它仍然需要支付一次完整的磁盘加载和量化成本。

区别是这些权重加载到 daemon，而不是加载到一个随时可能退出的 engine 中。

### 2.2 第一次启动 engine：映射 daemon 权重

```text
Engine 启动
    │
    ├─ 连接对应 rank 的 daemon
    ├─ 检查双方模型和并行配置是否一致
    ├─ 创建没有真实权重的 meta model
    ├─ 获取 CUDA IPC handles
    ├─ 把 Parameter/Buffer 指向 daemon GPU memory
    └─ 继续初始化 KV cache、CUDA Graph 等
```



### 2.3 Engine 退出

engine 退出时释放的是：

- engine 自己的 CUDA context；
- KV cache；
- CUDA Graph 和运行时 buffer；
- engine 进程中的 IPC mapping。

权重的 physical GPU memory 仍由 standalone daemon 持有，所以不会释放。

### 2.4 新 Engine 重启

```text
旧 Engine 退出
    │
    │ daemon 仍然持有 GPU 权重
    ▼
新 Engine 启动
    │
    ├─ 重新连接 daemon
    ├─ 重新建立 IPC mapping
    └─ 跳过磁盘权重加载
```

这就是这个 PR 所说的 fast recovery。

## 3. Daemon 和 Engine 各自负责什么


| 组件                  | 负责的事情                                                     |
| ------------------- | --------------------------------------------------------- |
| Weight Cache Daemon | 从磁盘加载权重、完成 TP/PP sharding、完成量化后处理、持有 GPU 权重、导出 IPC handle |
| Engine              | 构建推理模型结构、映射 daemon 权重、分配 KV cache、capture CUDA Graph、处理请求 |


最关键的职责边界是：

```text
Daemon 负责 weight storage 的生命周期
Engine 负责 serving runtime 的生命周期
```

所以 engine 可以重启，而 weight storage 不需要跟着重建。

### 3.1 会不会额外多一份 GPU 权重

正常情况下不会。

```text
Daemon
    └─ 100 GB physical GPU weights
                 ▲
                 │ CUDA IPC 零拷贝映射
                 │
Engine A ─────────┤
Engine B ─────────┘
```

多个 engine tensor 指向 daemon 的同一份 physical GPU memory，而不是每个 engine 再复制 100 GB。

但每个 engine 的以下显存仍然是独立的：

- KV cache；
- CUDA Graph pool；
- 推理 workspace；
- CUDA context。

daemon 自己也有 CUDA context、distributed communicator 和少量辅助 buffer 开销。因此是“权重约 1×”，不是“多个完整 engine 的总显存只有 1×”。

## 4. 从入口看代码调用链

先看一条主线，再进入实现细节。

### 4.1 Daemon 启动链路

```text
python -m sglang.srt.weight_cache.daemon
    │
    ├─ launch_weight_cache_daemons()
    │      └─ 为本节点的每个 PP×TP rank 启动一个子进程
    │
    └─ run_weight_cache_daemon()
           │
           ├─ WeightCacheDaemon.load()
           │      ├─ 初始化 distributed/model parallel
           │      ├─ 调用原 SGLang model loader
           │      └─ _export_state()
           │
           └─ WeightCacheDaemon.serve()
                  └─ 监听 Unix socket，等待 engine
```

相关文件：

```text
python/sglang/srt/weight_cache/daemon.py
python/sglang/srt/weight_cache/protocol.py
```



### 4.2 Engine 加载链路

```text
ModelRunner.load_model()
    │
    ├─ build_load_config()
    │
    ├─ maybe_enable_ipc_weight_cache()
    │      ├─ load_format 改为 IPC_CACHE
    │      └─ 计算当前 rank 的 daemon socket
    │
    ├─ get_model_loader()
    │      └─ 返回 IpcModelLoader
    │
    └─ IpcModelLoader.load_model()
           ├─ _fetch_from_cache()
           ├─ _load_zero_copy_mode()
           ├─ _rebuild_stale_views()
           └─ 启动 daemon liveness watchdog
```

相关文件：

```text
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
python/sglang/srt/model_loader/loader.py
python/sglang/srt/weight_cache/ipc_loader.py
```

到这里已经可以概括整个 PR：daemon 正常加载一次权重，engine 的 loader 从“读 checkpoint”切换成“创建 meta model 并映射 CUDA IPC tensor”。

## 5. Daemon 内部实现细节



### 5.1 复用原来的模型加载器

`WeightCacheDaemon.load()` 没有重新实现 checkpoint loader，而是调用原来的 `get_model_loader()`：

```python
loader = get_model_loader(
    load_config=load_config,
    model_config=model_config,
)

self.model = loader.load_model(
    model_config=model_config,
    device_config=DeviceConfig(...),
)
```

因此 daemon 保存的是完成下列处理后的最终权重：

- checkpoint load；
- TP/PP shard；
- quantization load；
- `process_weights_after_loading()`；
- post-quant 参数和权重 layout。

这意味着 engine 不需要重新执行这些步骤。

### 5.2 导出模型状态

`_export_state()` 首先遍历：

```python
self.model.state_dict()
```

导出 Parameter 和 persistent buffer。

然后遍历：

```python
self.model.named_buffers()
```

补充导出不在 `state_dict()` 中的 non-persistent buffer，例如部分 rotary cache。

### 5.3 生成 IPC handle

每个 tensor 通过：

```python
MultiprocessingSerializer.serialize(tensor.data, output_str=True)
```

生成 CUDA IPC handle，并记录：

```python
{
    "handle": ipc_handle,
    "shape": list(tensor.shape),
    "dtype": ...,
    "is_param": ...,
}
```

socket 传输的是这些 handle 和 metadata，不是完整权重数据。

### 5.4 每个 rank 的 socket

rank 计算公式是：

```python
global_rank = tp_size * pp_rank + tp_rank
```

默认 socket 是：

```text
/tmp/sglang_weight_cache_rank{global_rank}.sock
```

不同 engine rank 连接各自对应的 daemon。

## 6. Engine 内部实现细节



### 6.1 为什么使用 meta model

如果 engine 正常初始化模型，它会先分配一份完整权重，然后才能替换成 IPC tensor，这就失去了节省显存和时间的意义。

所以 `IpcModelLoader` 使用：

```python
with torch.device("meta"):
    model = _initialize_model(...)
```

meta model 只有 module hierarchy、shape 和 dtype，没有真实 weight storage。

### 6.2 打开 daemon 的 tensor

engine 对每个 entry 调用：

```python
imported_tensor = MultiprocessingSerializer.deserialize(entry["handle"])
```

得到的 tensor 映射 daemon 的 CUDA allocation，不会复制 tensor 内容。

### 6.3 替换 Parameter 和 Buffer

loader 根据完整 tensor name 找到对应 module，然后替换整个 Parameter：

```python
new_param = nn.Parameter(imported_tensor, requires_grad=False)
setattr(module, parameter_name, new_param)
```

Buffer 通过 `register_buffer()` 替换。

如果量化后处理产生了 meta model 中不存在的新参数，例如某些 `weight_scale`，loader 会把它注册成新 Parameter/Buffer。

### 6.4 完整性校验

映射过程中会检查同名 tensor 的 shape 和 dtype。

全部映射完成后，再扫描模型是否还有 Parameter/Buffer 留在 meta device。如果存在，就说明 daemon 没有导出完整状态，engine 直接启动失败。

它不会用未初始化的 `torch.empty()` 填充缺失权重，避免服务看似正常启动但输出错误。

### 6.5 为什么跳过 post-process

daemon 已经执行过 `process_weights_after_loading()`，因此 client 不再执行 `_post_load_weights()`。

否则已经量化或重排的权重会被处理第二次。

### 6.6 为什么需要 `_rebuild_stale_views()`

部分 module 在初始化时保存了 Parameter 的普通 tensor view：

```python
self.conv_weights = self.conv1d.weight.view(...)
```

IPC loader 替换 `conv1d.weight` 后，这个普通属性仍然指向旧 meta tensor。因此代码会重新构建已知的 `RadixLinearAttention.conv_weights` 和 bias 引用。

### 6.7 保持 mapping 存活

imported tensors 被保存在：

```python
model._ipc_imported_tensors
```

防止 Python GC 回收 tensor 并解除 IPC mapping。

## 7. 配置校验、支持范围和失败行为

这些限制是理解实现正确性的最后一部分。

### 7.1 Daemon 与 engine 必须是同一套权重配置

双方通过 `CacheConfig` 比较：

```text
model_path / model_arch / revision
tp_size / tp_rank
pp_size / pp_rank
dp_size / ep_size
quant_method / quant_config_hash
dtype
GPU compute capability
PyTorch version
```

任何字段不一致，daemon 都拒绝返回权重。

### 7.2 当前只支持未量化和 block-wise FP8

CUDA IPC 只共享 tensor storage，不会自动共享 quant method 的 Python metadata 或 layout 语义。

所以 PR 使用 allowlist：

```python
IPC_QUANT_ALLOWLIST = {
    "": ...,       # 未量化
    "fp8": ...,    # 只允许 block-wise FP8
}
```

其它量化方式直接报错，避免 IPC tensor 能映射但数值含义不正确。

### 7.3 Daemon 不能在 engine 运行期间退出

engine 的 Parameter 指向 daemon 的 CUDA allocation。daemon 退出后，这些 pointer 不再安全。

`IpcModelLoader` 会每 5 秒检查 daemon PID。daemon 死亡时，watchdog 会终止 engine，避免继续读取悬空 GPU memory。

### 7.4 不能修改共享权重

所有 client 共享同一份 physical weights。任意一个 client 原地修改权重，其他 client 也会看到修改。

因此 IPC 模式禁止：

- 在线更新权重；
- release/resume weights；
- weights CPU backup。



### 7.5 Client fallback 行为


| 情况                | 行为                    |
| ----------------- | --------------------- |
| daemon socket 不存在 | 使用原 load format 从磁盘加载 |
| socket 存在但连接失败    | 报错                    |
| CacheConfig 不匹配   | 报错                    |
| IPC tensor 导入失败   | 报错                    |


daemon 模式不允许 disk fallback，因为 daemon 已经在同一 GPU 上持有权重，再加载一份可能 OOM。

## 8. 最终效果

PR 给出的 Qwen3-235B FP8、TP=4 数据是：


| 指标           | 普通启动        | IPC client 启动 |
| ------------ | ----------- | ------------- |
| 权重加载         | 约 306–327 秒 | 约 0.63–1.3 秒  |
| 完整 server 启动 | 约 390 秒     | 约 80 秒        |


它消除了最大的权重加载瓶颈，但没有消除：

- tokenizer/server init；
- NCCL/distributed init；
- DeepGEMM JIT warmup；
- CUDA Graph capture；
- server warmup。

最准确的一句话总结是：

> standalone daemon 先正常加载并长期持有每个 rank 的最终 GPU 权重；可重启的 engine 使用 meta model 建立空结构，再通过 CUDA IPC 把 Parameter/Buffer 指向 daemon 的同一份 physical memory，从而跳过重复的 checkpoint 读取、TP/PP sharding 和量化后处理。



## 9. 与 RL 权重同步的关键区别

RL 训练中的权重同步也可能使用 CUDA IPC，并且发送的同样只是 handle。但“发送 handle 时没有复制”不代表整个权重更新过程没有复制。

需要区分两种 zero-copy：

1. **传输层 zero-copy**：进程间只传 CUDA IPC handle，不经 CPU 搬运完整 tensor；
2. **最终存储 zero-copy**：接收方的模型参数长期直接指向发送方的 GPU allocation。



### 9.1 普通 RL 权重同步：IPC 映射后再复制

典型流程是：

```text
训练侧参数
    -> gather / flatten / dtype 转换
    -> 生成临时同步 buffer
    -> 通过 CUDA IPC 发送 handle

推理侧
    -> 打开 handle，得到映射训练侧 buffer 的 tensor
    -> param.data.copy_(ipc_tensor)
    -> 推理参数现在持有自己的权重内容
    -> 释放 IPC tensor

训练侧
    -> 收到同步完成信号
    -> 释放临时同步 buffer
```

因此，RL 同步的 IPC 通信阶段是 zero-copy，但最终仍然存在一次 GPU-to-GPU copy。SGLang 的标准 `load_weights()` 路径最终通常会进入类似下面的 weight loader：

```python
param.data.copy_(loaded_weight)
```

这里 `loaded_weight` 可以是通过 CUDA IPC 打开的训练侧 tensor，而 `param` 是推理 engine 原本就持有的参数。

之所以必须复制，是因为训练侧通常只在同步期间保留 gather/flatten 后的临时 buffer。同步完成后它需要释放或复用这块显存，而且下一轮训练还会继续修改权重。推理 engine 不能长期依赖这个临时 allocation。

### 9.2 Weight Cache：直接使用 IPC 映射，不复制

这个 PR 的流程不同：

```text
daemon 长期持有 GPU 权重
    -> 发送 CUDA IPC handle

engine
    -> 打开 handle
    -> Parameter/Buffer 直接指向 IPC tensor
    -> 不执行 param.data.copy_(ipc_tensor)
```

这里 daemon 的 GPU allocation 本身就是 engine 最终使用的权重存储。engine 不需要再建立一份内容相同的权重，因此是最终存储层面的 zero-copy。

代价是 daemon 必须一直存活，并且共享权重必须保持只读。否则 daemon 释放或修改 allocation，会直接影响所有 engine。

### 9.3 放在一起比较


| 对比项                     | RL 权重同步              | Weight Cache Daemon           |
| ----------------------- | -------------------- | ----------------------------- |
| IPC 中传递的内容              | handle               | handle                        |
| IPC 打开后是否映射源 allocation | 是                    | 是                             |
| 是否再写入推理侧 Parameter      | 是，通常执行 `copy_()`     | 否，直接替换 Parameter/Buffer 的底层存储 |
| 完整流程是否 zero-copy        | 否，只有传输层 zero-copy    | 是，最终权重也共享同一份 physical memory  |
| 源进程/源 buffer 能否在完成后释放   | 可以；复制完成后推理侧不再依赖它     | 不可以；daemon 必须与 engine 同时存活    |
| 权重能否继续被修改               | 训练侧可以继续训练，推理侧使用自己的快照 | 共享权重应保持只读                     |
| 主要目标                    | 把新版本权重更新到一个已运行的推理模型  | 让重启后的 engine 复用已经加载好的权重       |


一句话区分：

> RL 权重同步是“通过 CUDA IPC 零拷贝地拿到源 tensor，再复制到推理参数”；Weight Cache 是“推理参数长期直接使用 CUDA IPC tensor”。



## 10. RFC 延伸：共卡 RL 如何让最新权重跨 Engine 重启

本章分析的不是 PR #27139 已经实现的功能，而是后续 RFC 对**训练和 rollout 共卡**场景的设想。这里先不看 GMS、VMM、lease 和 commit 等术语，只看它想解决什么问题以及整体方案怎么运转。

### 10.1 先从问题和整体方案看起

普通共卡 RL 的流程是：

```text
SGLang Engine 使用 W0 rollout
        -> Trainer 训练得到 W1
        -> W1 同步到 SGLang Engine
        -> Engine 使用 W1 rollout
```

问题在于：rollout 权重显存通常跟着 SGLang Engine 进程。即使已经成功同步到 W1，只要 Engine 崩溃，这份 rollout 权重显存也会随进程消失；新 Engine 只能重新加载或重新同步 W1。

RFC 的高层思路是：**把 rollout 权重的生命周期从 Engine 中拿出来，交给一个长期运行的 GPU 显存管理服务。**

```text
以前：Engine 既使用权重，也拥有权重显存

以后：独立服务长期保管权重显存
      Engine 连接它、使用它，但不决定这块显存活多久
```

这样整个 RL 循环变成：

```text
第一次启动
  -> Engine 把 W0 加载进长期存在的权重显存
  -> Engine 使用 W0 rollout

训练更新
  -> Trainer 产生 W1
  -> Engine 把 W1 写入这块长期存在的权重显存
  -> Engine 使用 W1 rollout

Engine 崩溃
  -> Engine 自己的运行时资源消失
  -> 长期显存服务仍保留 W1
  -> 新 Engine 重新连接这份 W1
  -> 不重新从磁盘加载权重
```

这个长期运行的 GPU 显存管理服务，就是 RFC 中的 **GPU Memory Service（GMS）**。引入 GMS 不是为了替代 Trainer 或 SGLang，也不是为了执行推理；它只解决一个核心问题：**让最新 rollout 权重的显存比 Engine 进程活得更久。**

### 10.2 三个角色分别负责什么


| 角色            | 通俗理解        | 负责什么                                          |
| ------------- | ----------- | --------------------------------------------- |
| Trainer       | 新权重的生产者     | 根据 rollout 数据训练出 W1、W2 等新版本                   |
| SGLang Engine | 懂模型的使用者和写入者 | 接收训练权重，调用模型 weight loader 写入 rollout 权重，并执行推理 |
| GMS           | 长期运行的显存保管者  | 分配并保留权重显存，让权重不随 Engine 退出而消失                  |


三者的关系是：

```text
Trainer 产生 W1
        │
        ▼
SGLang Engine 负责按模型规则写入 W1
        │
        ▼
GMS 负责让写好的 W1 显存长期存在
        │
        ▼
当前或重启后的 SGLang Engine 使用 W1 rollout
```

GMS 不知道什么是 Transformer、TP shard 或 FP8，也不执行 `load_weights()`。这些模型逻辑都留在 SGLang Engine 中。GMS 只负责显存的分配、保存和重新共享。

本章后面的具体设计来自 [Weight Cache Roadmap #33522](https://github.com/sgl-project/sglang/issues/33522) 和 [GPU Memory Service RFC #27310](https://github.com/sgl-project/sglang/issues/27310)。下面再进入物理显存、映射和更新细节。

### 10.3 GMS 和前面的 Weight Cache Daemon 有什么区别

两者的核心思想相同：都把权重显存放到 Engine 之外，使权重不跟随 Engine 进程退出。

真正的区别是：

> Weight Cache Daemon 是懂模型的权重管理进程；GMS 是不懂模型的纯显存管理进程。

当前 PR #27139 的 daemon 同时负责：

```text
Checkpoint
    ↓
Weight Cache Daemon
    ├─ 创建 SGLang 模型
    ├─ 从磁盘加载权重
    ├─ 执行 TP/PP sharding
    ├─ 执行量化和权重后处理
    ├─ 持有最终 physical GPU memory
    └─ 导出 CUDA IPC handle
              ↓
           Engine 读取
```

所以 daemon 同时是：

```text
Owner  = daemon：持有 physical memory
Writer = daemon：负责加载并写入权重
Reader = Engine：映射并使用权重
```

它必须理解模型结构、参数名、并行切分、量化方式和 SGLang weight loader。

GMS 只负责：

```text
cuMemCreate() 分配 physical GPU memory
保存 allocation handle
向客户端导出 shareable FD
管理谁可以读、谁可以写
保存 layout 和 ready 状态
```

GMS 不创建模型，不读取 checkpoint，不调用 `load_weights()`，也不知道 allocation 中保存的是 Transformer 权重还是其它 tensor。在共卡 RL 中，角色变成：

```text
Owner  = GMS：持有 physical memory
Writer = Engine：接收 Trainer 权重并调用 model.load_weights()
Reader = Engine：使用已写好的权重 rollout
```

放在一起比较：


| 对比项                    | Weight Cache Daemon                | GMS                                         |
| ---------------------- | ---------------------------------- | ------------------------------------------- |
| 是否独立于 Engine 持有显存      | 是                                  | 是                                           |
| 是否理解 SGLang 模型         | 是                                  | 否                                           |
| 是否创建模型并加载 checkpoint   | 是                                  | 否                                           |
| 谁执行 `load_weights()`   | daemon                             | Engine 或其它 writer client                    |
| 是否处理 TP shard、量化和模型后处理 | 是                                  | 否                                           |
| 显存机制                   | PyTorch/CUDA allocation + CUDA IPC | CUDA VMM physical allocation + shareable FD |
| 是否管理读写权限和发布状态          | 当前主要提供只读共享                         | 提供 RW/RO 和 commit 状态机                       |
| 定位                     | SGLang 专用的权重仓库                     | 通用的 GPU 显存 owner                            |


理论上也可以继续扩展 daemon，让 Trainer 把新权重发给 daemon，再由 daemon 更新共享权重。但共卡 RL 中，Engine 已经具有在线权重同步、模型 weight loader、并行切分和更新后清理逻辑。让 Engine 直接写 GMS 的路径更短：

```text
经过 daemon：Trainer -> daemon -> shared weights -> Engine

共卡 GMS： Trainer -> Engine(writer) -> GMS physical memory
```

GMS 和 daemon 也不一定互相替代。RFC 将 owner 和 writer 拆开后，可以组成不同方案：

```text
当前 PR #27139：
Owner = daemon，Writer = daemon，Reader = Engine

非共卡/解耦场景的 RFC 方向：
Owner = GMS，Writer = daemon，Reader = Engine

共卡 RL 的 RFC 方向：
Owner = GMS，Writer = Engine，Reader = Engine
```

因此，GMS 本质上是把 daemon 原来承担的“物理显存所有权”抽成一个独立的通用层；模型加载和权重更新仍可以根据场景由 daemon 或 Engine 完成。

### 10.4 再看底层：权重物理显存由 GMS 持有

共卡方案最重要的事实只有一个：

> GMS 调用 CUDA VMM 的 `cuMemCreate` 分配并长期持有权重的 physical GPU memory；Engine 只把这块显存映射成自己的 Parameter。

GMS 和 Engine 持有的东西不同：

```text
GMS Server
  └─ physical allocation handle
       └─ 真正保存 W0/W1 数据的 GPU memory

SGLang Engine
  └─ virtual-address mapping
       └─ Parameter 通过这个地址访问 GMS 的 physical memory
```

不过，这两个独立进程并不是凭空看到同一块显存。**VMM 没有消灭 IPC。** GMS 仍然需要把一个可共享的 handle 通过进程间通信交给 Engine。

完整链路是：

```text
GMS Server
  │
  ├─ cuMemCreate()
  │    └─ 创建并持有 physical GPU memory
  │
  ├─ cuMemExportToShareableHandle()
  │    └─ 为这块 allocation 导出一个 shareable FD
  │
  │ Unix Socket + SCM_RIGHTS 传递 FD 和 metadata
  ▼
SGLang Engine
  │
  ├─ cuMemImportFromShareableHandle()
  │    └─ 导入同一个 physical allocation
  │
  ├─ cuMemAddressReserve()
  │    └─ 预留 Engine 想使用的 virtual address
  │
  ├─ cuMemMap()
  │    └─ 将 GMS allocation 映射到这个地址
  │
  └─ cuMemSetAccess()
       └─ 按当前角色设置 RW 或 RO 权限
```

因此，如果把“CUDA IPC”泛指所有跨进程 GPU memory sharing，那么 VMM FD 仍然是一种 IPC。如果特指 `cudaIpcGetMemHandle/cudaIpcOpenMemHandle` 这套传统 API，那么 GMS 使用的是另一套方案：`cuMemExportToShareableHandle`、OS FD 和 `cuMemMap`。

两条路径的对比是：


|                 | 传统 CUDA IPC               | VMM shareable FD                   |
| --------------- | ------------------------- | ---------------------------------- |
| 创建显存            | 普通 PyTorch/CUDA allocator | `cuMemCreate()`                    |
| 导出              | `cudaIpcGetMemHandle()`   | `cuMemExportToShareableHandle()`   |
| 进程间传递           | CUDA IPC handle           | Unix Socket 传 OS FD                |
| 导入              | `cudaIpcOpenMemHandle()`  | `cuMemImportFromShareableHandle()` |
| VA 由谁决定         | CUDA driver 决定            | Client reserve VA 后主动 `cuMemMap()` |
| 能否保留 VA 再 remap | 不能保证                      | 可以                                 |
| 是否复制权重数据        | 否                         | 否                                  |


两种方案传递的都只是 handle/FD 和 metadata，而不是完整权重。初始 W0 或训练后的 W1 仍然需要由 writer Engine 通过 `load_weights()` 写入最终 physical allocation。

Engine Parameter 与 GMS 之间也不是两份权重：

```text
Engine Parameter
       │ virtual address
       ▼
Engine 的 VMM mapping
       │
       ▼
GMS 持有的唯一一份 physical weights
```

Engine 退出时，只会销毁自己的 Parameter、虚拟地址映射和本地导入 handle。GMS 保存的 physical allocation 仍然存在，所以新 Engine 可以再次取得 FD，重新导入并映射同一份权重。

这就是它能跨 Engine 崩溃恢复的根本原因。后面的 RW/RO、commit 和 manifest，都只是围绕这块由 GMS 持有的物理显存进行管理。

#### 为什么有 VMM 还要保留传统 CUDA IPC

VMM 的地址和生命周期控制更强，但使用成本也更高：

- 传统 CUDA IPC 可以直接导出已经由 PyTorch allocator 创建的 CUDA tensor；
- VMM 要求从分配阶段就通过专门 allocator 将 weight allocation 路由到 GMS；
- VMM 还需要管理 alignment、VA reservation、FD、mapping、layout 和访问权限；
- 当前只读 Weight Cache 不要求更新前后保持同一个 VA，传统 CUDA IPC 已经能解决主要问题；
- VMM FD 主要面向支持相应 CUDA Driver API 的 NVIDIA 环境，`torch_ipc` 仍需要作为其它平台和配置的 fallback。

所以 RFC 的方向不是让 VMM 在所有场景替代传统 CUDA IPC，而是提供两种 transport：简单只读共享优先使用已有 IPC 路径，需要稳定 VA、权重更新或 sleep/wake 时使用 VMM FD。

共卡方案的数据路径最终仍是：

```text
Trainer -> SGLang Engine -> GMS 持有的 physical weights
```

这里没有 Weight Cache Daemon。GMS 也不理解模型结构或调用 `load_weights()`；模型相关的加载和更新仍由 SGLang Engine 完成。

### 10.5 一张 GPU 上有哪些进程

对于一个共卡 RL job，典型拓扑是：

```text
GPU 0
  ├─ GMS Server 0：长期持有 rollout 权重的 physical memory
  ├─ Trainer rank 0：训练并产生 W1
  └─ SGLang Engine rank 0：使用权重 rollout，并负责接收更新
```

一个 GMS Server 只管理一张物理 GPU。TP=4 时，每张 GPU 各有一个 GMS Server，分别持有对应 TP rank 的权重：

```text
GPU 0：GMS 0 持有 TP rank 0 权重
GPU 1：GMS 1 持有 TP rank 1 权重
GPU 2：GMS 2 持有 TP rank 2 权重
GPU 3：GMS 3 持有 TP rank 3 权重
```

GMS 长期运行，SGLang Engine 可以退出和重新创建。通常一张 GPU 同时只有一个工作的 SGLang Engine rank；GMS 支持多个 reader 是额外能力，不是本章共卡 RL 流程的前提。

### 10.6 第一次启动：W0 怎样真正进入 GMS 显存

第一次启动仍然要从 checkpoint 正常加载一次 W0。区别是权重的 physical allocation 由 GMS 创建，而不是由 Engine 自己的普通 CUDA allocator 创建。

```text
1. Engine 向 GMS 申请一块权重显存
2. GMS 调用 cuMemCreate，持有 physical allocation
3. GMS 导出 shareable FD/handle 给 Engine
4. Engine 导入 handle，并以 RW 权限建立虚拟地址映射
5. Engine 创建 Parameter，使它指向这个映射地址
6. Engine 执行 model.load_weights(W0)
7. Engine 通过自己的 RW mapping 执行普通 `copy_()`，W0 直接写进 GMS 持有的 physical allocation
8. 全部写完后，Engine 调用 GMS commit()
9. Engine 改用 RO 映射，开始 rollout
```

这里7不是 Engine 把权重数据发送给 GMS Server，再由 GMS Server 执行一次 GPU copy。GMS Server 只负责分配显存和导出 FD；Engine 的 Parameter 已经指向这块映射，所以 `model.load_weights()` 中普通的 `param.data.copy_(loaded_weight)` 会直接写入 GMS-owned physical memory，数据面不经过 GMS Server。

第 7 步结束时，权重数据已经在 GMS physical memory 中。`commit()` 不会再复制权重，它只表示：

> W0 的所有 tensor 和恢复 metadata 已经写完，将这份 layout 标记为 ready，之后可以让 reader 映射。

可以把它直接读成 `mark_ready()`：

```python
gms.connect("RW")
model.load_weights(W0)   # 真正写权重
gms.commit()             # CUDA 同步并将当前 layout 标记为可读
gms.connect("RO")
gms.remap_all_vas()
start_rollout()
```

`commit()` 不是保存 checkpoint、不是将 W0 复制给 GMS，也不是创建一份新权重。GMS 从 `cuMemCreate` 开始就是 physical memory owner；commit 只是结束“尚未写完”的状态并发布当前 layout。

第一次启动结束后的关系是：

```text
GMS：持有 physical weights W0
                   ▲
                   │ RO mapping，同一份数据
                   │
Engine：Parameter ─┘
```



### 10.7 训练得到 W1 后，实际更新的是哪块显存

Trainer 产生 W1 后，目标不是把 W1 写进 Engine 私有显存，也不是让 Engine 再转发给 Weight Cache Daemon。目标就是把 W1 写进 GMS 持有的 rollout 权重 allocation。

RFC 预期流程是：

```text
1. Engine 暂停 rollout，等待正在执行的 forward 完成
2. Engine 释放 W0 的 RO 访问权限
3. Engine 从 GMS 获得独占 RW 访问权限
4. Trainer 通过现有权重同步路径把 W1 发给 Engine
5. Engine 调用 model.load_weights(W1)
6. param.data.copy_(trainer_weight) 将 W1 写入 GMS physical memory
7. 全部写完并完成必要后处理后，Engine 调用 commit()
8. Engine 重新以 RO 权限映射 committed W1
9. 复用现有 cache/version 收尾逻辑并恢复 rollout
```

数据流只有这一条：

```text
Trainer 中的 W1 tensor
        │ CUDA IPC / NCCL / 现有 update_weights 接口
        ▼
SGLang Engine 的 model.load_weights()
        │ 一次 GPU copy
        ▼
GMS 持有的最终 rollout weights W1
```

Engine 是执行模型 weight loader 的进程，但最终写入目标仍然是 GMS-owned physical memory。commit W1 的含义也只是“W1 已经全部写完，可以重新给 reader 使用”。

### 10.8 Engine 崩溃后为什么能恢复到 W1

假设 W1 已经写入并 commit。此时真正保存 W1 的仍是 GMS physical allocation：

```text
GMS：physical weights W1
                 ▲
                 │ Engine A 的 RO mapping
                 │
Engine A ─────────┘
```

Engine A 崩溃时，只会消失：

- Engine A 的进程和 PyTorch 对象；
- Engine A 的虚拟地址映射；
- Engine A 导入的本地 VMM handle；
- Engine A 的 KV Cache、CUDA Graph 和其他运行时资源。

不会消失的是 GMS 自己持有的 physical allocation，所以 W1 数据仍然在 GPU 上。

新 Engine B 的恢复流程是：

```text
1. Engine B 连接仍在运行的 GMS
2. 查询最后一次 committed layout 和模型恢复 metadata
3. 创建空的 meta model
4. 从 GMS 重新导出并导入 allocation handle
5. 将 Parameter 映射到 GMS 仍然持有的 W1
6. 恢复 rollout
```

这里没有从磁盘重新读取 W1，也没有从 GMS 再复制一份 W1 给 Engine B。Engine B 只是重新建立映射。

### 10.9 RW、RO 和 commit 放在一起理解

它们只是 GMS 对同一份 owner-managed memory 的访问状态：

```text
RO：权重已经 ready，Engine 只能读取并 rollout
RW：权重正在写，只允许一个 writer Engine 访问
commit：writer 宣布已经全部写完，将当前 layout 标记为 ready
```

最短状态流是：

```text
RO(W0) -> 暂停 rollout -> RW(写 W1) -> commit -> RO(W1)
```

commit 前不会把正在写的 layout 当作可读权重交给 reader。至于 writer 在更新中途崩溃后，是否保留上一版 W0、丢弃整个 active layout，还是从其它引用恢复，RFC 尚未最终确定；不能把它直接理解成已经具备数据库式版本回滚。

### 10.10 当前 SGLang 能复用什么，还缺什么

共卡方案不是重新发明整套 RL 权重同步。可以直接复用的部分包括：

- `update_weights_from_tensor/distributed/ipc`；
- 模型自己的 `load_weights()` 和 weight loader；
- rollout 暂停与权重更新锁；
- TP rank 协调；
- 更新后的 KV/Radix Cache 清理；
- weight version 管理。

RFC 真正需要新增的是：

- 把 weight allocation 路由到外部 VMM owner 的 allocator backend；
- RW/RO lease 和 epoch commit 协议；
- 保留虚拟地址并重新映射 backing；
- 将 manifest 与 committed 权重一起发布；
- 允许 Weight Cache 模式下的 `update_weights_*`，而不是像当前代码一样直接拒绝；
- 新 Engine 从 owner 的 manifest 恢复模型的入口。

所以它的重点不是“重新实现 cache 清理或 RL 权重同步”，而是给现有流程增加一个**跨 Engine 生命周期的权重存储和发布层**。

### 10.11 本章结论

共卡 RL 方案最核心的设计不是“让 Engine 把权重转发给 daemon”，而是把三个概念拆开：

```text
Owner 负责：权重显存活多久
Writer 负责：谁把本轮训练权重写进去
Reader 负责：谁使用已提交权重做 rollout
```

在共卡场景下，最短的数据路径是：

```text
Trainer -> Engine(writer) -> owner-managed memory
                            -> Engine(reader) rollout
```

Engine 从 reader 临时切换成 writer，直接复用现有 `load_weights()` 写入新权重；commit 后再切回 reader。Owner 让最后一次 committed 权重独立于 Engine 存活，从而同时实现在线 RL 更新和 Engine 快速恢复。

这仍然是 RFC/roadmap，而不是当前可用功能。尚未定型的部分主要是具体 API、更新中断时的失败语义、manifest 格式，以及量化模型的更新后处理方式。

## 11. 既然 CUDA IPC 已经能快速重启，为什么还要引入 VMM

先给结论：

> 如果目标只是“daemon 活着，Engine 挂掉后重新映射权重”，VMM 不是必需的，PR #27139 使用传统 CUDA IPC 已经能够做到。RFC 引入 VMM，是因为目标继续扩大到了权重 sleep/wake、在线更新时保持指针、CUDA Graph 恢复，以及更通用的外部显存 owner。

但这句话只覆盖了 Engine 崩溃。RFC #27310 还强调了传统 CUDA IPC 的另一个实际故障：**exporter daemon 一旦崩溃，Engine 中导入的 mapping 就不再安全，当前实现只能让 watchdog 连带杀死 Engine。** VMM FD 可以解除这层生死绑定。

### 11.1 先用通俗方式理解 VMM

先暂时忘掉 GMS、RL 和 CUDA Graph。VMM（Virtual Memory Management）只做一件事：

> 把“程序使用的地址”和“真正保存数据的物理显存”拆开管理。

可以把它们理解成：

```text
Virtual Address（VA）：程序使用的门牌号
Physical Memory：      真正存放数据的房间
Mapping：              门牌号和房间的对应关系
```

普通 `torch.empty(..., device="cuda")` 可以粗略理解为一次完成三件事：

```text
1. 找到 physical memory P0
2. 分配 virtual address A
3. 建立 A -> P0 的 mapping
```

PyTorch Tensor 最终只保存并使用地址 A：

```text
Tensor.data_ptr() = A
        │
        ▼
physical memory P0
```

VMM 将这三步拆开：

```text
cuMemAddressReserve()：只保留 VA A，背后暂时没有显存
cuMemCreate()：        只创建 physical memory P0，暂时没有地址
cuMemMap(A, P0)：      建立 A -> P0
```

最关键的是，VMM 可以解除 mapping 但保留 VA：

```text
开始：A -> P0

unmap：
A -> 空
P0 可以继续由 owner 持有

remap：
A -> P1
```

所以程序始终看到同一个地址 A，但背后的 physical backing 可以变化：

```text
更新前：Parameter VA A -> physical P0(W0)
更新后：Parameter VA A -> physical P1(W1)
```

这就是“稳定虚拟地址”的含义。需要特别记住：

- VMM 不会自动加载权重；
- VMM 不会自动复制 W0/W1；
- VMM 不理解 Parameter 或模型；
- 它只管理 VA、physical memory 以及两者之间的 mapping。

在 GMS 方案中，GMS 持有 P0/P1，Engine 持有 VA A 和 mapping：

```text
GMS：   持有 physical backing P
Engine：持有 VA A，并建立 A -> P
```

有了这个基础，再看后面的权重更新、sleep/wake 和 CUDA Graph，就只是在讨论“什么时候解除 A -> P0，什么时候建立 A -> P1，以及为什么希望 A 不变”。

### 11.2 RFC 到底为什么要把 CUDA IPC 换成 VMM FD

[RFC #27310 的第 2 节](https://github.com/sgl-project/sglang/issues/27310)给出的理由非常明确：传统 CUDA IPC 有两个结构性限制。

#### 限制一：importer 无法控制映射地址

传统 CUDA IPC 的过程是：

```text
daemon 中已有 PyTorch CUDA tensor
    -> cudaIpcGetMemHandle
    -> 将 CUDA IPC handle 交给 Engine
    -> cudaIpcOpenMemHandle
    -> CUDA driver 为 Engine 选择映射地址
```

Engine 不能指定“请映射到原来的 VA A”，也不能在保留 A 的前提下解除并更换 physical backing。因此：

- sleep/wake 后不保证 Parameter 仍使用原地址；
- W0 切换为 W1 时，原有 tensor/view 指针可能需要重建；
- CUDA Graph 中已经 capture 的 weight pointer 无法自然保持有效。

VMM shareable FD 的路径则是：

```text
创建 physical allocation P
    -> 导出 OS FD
    -> Engine 导入 FD
    -> Engine 自己 reserve VA A
    -> cuMemMap(A, P)
```

Engine 可以保留 VA A，先 unmap P0，再把 P0 或 P1 映射回 A。这才是 RFC 选择 VMM transport 的首要技术原因。

#### 限制二：传统 CUDA IPC mapping 的生命周期绑在 exporter 上

在 PR #27139 中，daemon 是 CUDA allocation 的 exporter：

```text
daemon 活着
  -> Engine 的 CUDA IPC mapping 可用

daemon 意外退出
  -> exporter allocation 被释放
  -> Engine 继续 forward 可能读到失效显存
  -> 可能发生 illegal address 或错误结果
```

所以当前 `ipc_loader.py` 使用 watchdog：发现 daemon PID 消失，就直接杀死 Engine。也就是说，传统 CUDA IPC 虽然能让 Engine 快速重启，却让 daemon 和 Engine 处于同一个故障域。

VMM 的 physical allocation 可以导出为 OS FD。FD 由内核引用计数；Engine 已经导入并建立的 mapping，不会仅因为最初导出 FD 的 daemon 退出就立刻失效。因此，**只做第一阶段的 VMM transport 替换，即使还没有 GMS，也能让正在运行的 Engine 脱离 daemon 的生命周期。**

这里必须区分：

```text
daemon 崩溃时，已经 attach 的 Engine 能否继续使用权重？
  -> VMM FD phase 1 可以解决

daemon 崩溃后，缓存能否继续接受新的 Engine attach？
  -> 仅 phase 1 不完整；需要 daemon 重载，或者由 phase 2 的 GMS 持有缓存
```

#### RFC 的两个阶段不是一回事

```text
当前 PR #27139
  owner = daemon，writer = daemon
  transport = traditional CUDA IPC

Phase 1：只换 transport
  owner = daemon，writer = daemon
  transport = VMM shareable FD
  没有新增 GMS，模型加载方式也不变

Phase 2：再换成 shared owner
  owner = GMS
  writer = daemon 或 Engine
  transport = VMM shareable FD
```

因此，RFC 不是因为“GMS 必须使用 VMM”才顺便替换 CUDA IPC；而是传统 CUDA IPC 自身缺少稳定 VA 和独立生命周期，先换 VMM transport 就已经有明确收益。GMS 是下一阶段对 physical ownership 的进一步拆分。

### 11.3 只做 Engine 权重快速恢复时，CUDA IPC 已经足够

PR #27139 的恢复流程是：

```text
daemon 一直持有 W0
        -> Engine A 通过 CUDA IPC 映射 W0
        -> Engine A 退出
        -> Engine B 重新创建模型
        -> Engine B 通过 CUDA IPC 重新映射 W0
```

Engine B 本来就会重新创建 Parameter、重新初始化运行时并重新 capture CUDA Graph，所以 W0 在新进程中映射到一个不同 virtual address 并不妨碍权重恢复。

因此，如果只评价“跳过磁盘加载权重”这一项：

```text
传统 CUDA IPC：够用
CUDA VMM：不是必要条件
```

### 11.4 为什么 RFC 不选择直接原地覆盖 W0

理论上可以暂停全部 reader，然后直接把共享 allocation P0 中的 W0 覆盖成 W1。但这不是 RFC 采用的更新协议，因为原地覆盖很难给出清晰的版本边界：

```text
copy 开始前：P0 全部是 W0
copy 进行中：P0 一部分是 W0，一部分是 W1
copy 完成后：P0 全部是 W1
```

只要暂停、同步或某个 TP rank 出错，reader 就可能面对半更新状态，而且旧的 committed W0 已被破坏，难以回退。

RFC 使用的是 RW/RO lease 和 epoch handoff：

```text
1. Engine 释放 W0 的 RO lease，并保留原 VA A
2. daemon/Engine writer 获得 RW lease
3. writer 写入新 epoch W1
4. 全部写完后 commit：RW -> RO
5. Engine 在原 VA A 上重新映射 committed W1
```

这里 `commit` 的作用是发布清晰的版本边界；VMM 的作用是让第 1、5 步 unmap/remap 后仍可使用 VA A。二者配合，才能让 Parameter 指针和已 capture 的 CUDA Graph 在更新后继续有效。

RFC 目前说明了 reader 不会看到写到一半的 epoch，但没有完整定义更新失败后是否保留 W0、如何回滚；这仍属于待设计的失败语义。

### 11.5 VMM 价值一：更换 physical backing，但保持 Parameter 地址

普通 CUDA IPC 重新打开另一块 allocation 时，新地址由 CUDA driver 决定：

```text
W0 handle -> VA A
W1 handle -> VA B
```

如果现有 Parameter、tensor view 或 kernel state 仍然保存地址 A，就需要重建或修复它们。

VMM 可以保留 VA A，只更换它背后的 physical memory：

```text
更新前：Parameter VA A -> physical P0(W0)
暂停时：保留 VA A，解除 A -> P0
更新后：Parameter VA A -> physical P1(W1)
```

这里不要求 P1 与 P0 是同一块 physical allocation；真正保持不变的是 Engine 看见的 VA A。

### 11.6 VMM 价值二：权重 sleep/wake

共卡 RL 可能需要让 rollout Engine 暂时释放映射或显存使用状态，然后再恢复：

```text
Rollout：Parameter VA A -> physical weights P0

Sleep：保留 VA A，解除 mapping

Wake：重新将可用 backing 映射到 VA A
```

如果重新映射后仍然是 VA A，PyTorch Parameter 不需要因为一次 sleep/wake 而全部更换地址。

普通 CUDA IPC 可以关闭再打开 handle，但不能要求 CUDA driver 下次仍然返回 A，因此很难保证原有指针继续有效。

### 11.7 VMM 价值三：CUDA Graph 恢复

CUDA Graph capture 时会记住 kernel 使用的原始指针。假设 capture 时权重地址是 A：

```text
Captured CUDA Graph -> weight pointer A
```

如果恢复后权重映射到了 B，旧 Graph 仍然访问 A，不能直接复用。VMM 可以让新的或原来的 physical backing 继续映射到 A，从而为 CUDA Graph 恢复提供稳定指针基础。

这也是 VMM 最直观的收益，但不是它唯一的用途。

### 11.8 Phase 2 再用 GMS 把 physical owner 和模型进程拆开

传统 Weight Cache Daemon 使用 PyTorch/CUDA allocator 创建 tensor，再导出 CUDA IPC handle。daemon 同时拥有 CUDA allocation、模型对象和 loading 逻辑。

Phase 1 仍然由 daemon 持有 physical allocation；Phase 2 才引入 GMS，将它进一步拆开：

```text
GMS Server：cuMemCreate，持有 physical allocation
Writer Client：导入 FD、map 成 RW、写入内容
Reader Client：导入 FD、map 成 RO、执行推理
```

GMS Server 不需要运行模型或映射权重；Engine 退出只会销毁自己的 mapping，不会释放 GMS 持有的 physical allocation。这使 owner、writer 和 reader 可以由不同进程承担。

### 11.9 两种技术分别适合什么

| 能力 | 传统 CUDA IPC | CUDA VMM shareable FD |
|---|---:|---:|
| 零拷贝共享权重 | 可以 | 可以 |
| 新 Engine 重新映射权重 | 可以 | 可以 |
| 直接导出现有 PyTorch CUDA tensor | 方便 | 需要从 allocator 阶段接入 |
| 主动选择映射 VA | 不可以 | 可以 |
| unmap 后保留原 VA | 不可以 | 可以 |
| 更换 backing 但保持 Parameter 地址 | 不可以 | 可以 |
| exporter 退出后，已建立的 importer mapping 继续有效 | 不可以 | 可以，依赖内核引用计数的 FD |
| sleep/wake 后保持已有指针 | 很困难 | 可以 |
| 为 CUDA Graph 恢复提供稳定地址 | 很困难 | 可以 |
| 实现复杂度 | 较低 | 较高 |
| 平台和现有 PyTorch 路径兼容性 | 更广 | 主要面向支持相关 CUDA Driver API 的 NVIDIA 环境 |

所以两者不是简单的“旧方案”和“全面更好的新方案”：

```text
简单、只读、只要求重新加载权重快
    -> 传统 CUDA IPC 已经足够

需要稳定 VA、sleep/wake、在线切换 backing、恢复 CUDA Graph
    -> 使用 VMM FD
```

### 11.10 最后把几个目标彻底分开

这套 RFC 同时讨论几个容易混淆的目标：

```text
目标一：在当前 PR 中，Engine 崩溃后权重为什么没有消失？
答案：因为仍然存活的 daemon 持有 physical GPU memory。

目标二：daemon 崩溃后，已经 attach 的 Engine 为什么还能继续使用 mapping？
答案：VMM shareable FD 的 physical allocation/mapping 不与 exporter 进程同生共死。

目标三：daemon 崩溃后，缓存为什么还能服务后来启动的新 Engine？
答案：Phase 2 让 GMS 成为长期 physical owner，而不是 daemon。

目标四：解除/重新映射后，Parameter 和 CUDA Graph 为什么还能使用旧地址？
答案：因为 VMM 可以保留并复用 virtual address。
```

因此：

> 传统 CUDA IPC 足以实现“daemon 活着时快速重启 Engine”；VMM phase 1 进一步解决稳定 VA 和 exporter 生命周期绑定；GMS phase 2 再让 committed cache 本身独立于 daemon。三层能力不能混为一谈。

## 12. RFC 延伸：非共卡 RL 如何更新权重并快速恢复

本章讨论训练和 rollout 不在同一组 GPU 上的情况。它对应 [Weight Cache Roadmap #33522](https://github.com/sgl-project/sglang/issues/33522) 和 [GMS RFC #27310](https://github.com/sgl-project/sglang/issues/27310) 中的 disaggregated RL / daemon-published 方向，目前仍是 RFC，而不是已经可用的 SGLang 功能。

### 12.1 先从高层问题和整体方案看起

非共卡 RL 的基本拓扑是：

```text
训练 GPU/节点
  └─ Trainer 产生 W1

Rollout GPU/节点
  └─ SGLang Engine 使用 W0/W1 推理
```

Trainer 和 rollout Engine 不共享同一块物理显存，因此不能像同 GPU 进程那样只传一个 CUDA IPC handle，就把它当作 rollout GPU 上的本地权重。跨节点时 CUDA IPC 本身就不可用；即使 Trainer 与 rollout 位于同一节点的不同 GPU，rollout 直接访问 Trainer GPU allocation 也不等于在自己的 GPU 上获得了可长期保留的本地权重。新权重仍需通过 NCCL、RDMA 或其它传输路径真正搬到对应的 rollout GPU。

如果 Trainer 直接把权重推给某个 Engine，会遇到几个问题：

- Engine 可能正在重启，Trainer 找不到稳定的接收方；
- Engine 未启动时无法提前把最新权重预热到 rollout GPU；
- 权重接收和 Engine 生命周期再次绑在一起。

这里的 daemon **不会减少 rollout GPU 之间必需的权重副本**：每张 rollout GPU 仍需要持有属于自己 TP rank 的权重。它解决的是“该 GPU 上由谁稳定接收和保存这份权重”，不是让多张 GPU 共用同一份显存。

RFC 的高层方案是：在 rollout GPU 上保留一个稳定的 Weight Cache Daemon，让它负责接收 Trainer 推送；真正的 physical GPU memory 仍由 GMS 长期持有。

```text
Trainer
   │ NCCL / RDMA，真正传输 W1 数据
   ▼
Weight Cache Daemon
   │ 按模型规则写入 W1
   ▼
GMS 持有的 rollout physical weights
   │ VMM RO mapping
   ▼
SGLang Engine rollout
```

因此完整循环是：

```text
第一次启动
  -> daemon 将 W0 写进 GMS
  -> Engine 映射 W0 并 rollout

训练更新
  -> Trainer 产生 W1
  -> Trainer 将 W1 推送给 rollout 侧 daemon
  -> daemon 将 W1 写进 GMS 并标记 ready
  -> Engine 重新映射 W1 并 rollout

Engine 崩溃
  -> GMS 仍保留最后一次成功发布的 W1
  -> 新 Engine 重新映射 W1
```

### 12.2 四个角色分别负责什么

| 角色 | 负责什么 |
|---|---|
| Trainer | 在训练 GPU 上产生 W1，并通过 NCCL/RDMA 推送到 rollout 侧 |
| Weight Cache Daemon | 理解模型和 shard，接收 Trainer 权重并作为 writer 发布新版本 |
| GMS | 在 rollout GPU 上持有 physical memory，管理 RW/RO 和 committed layout |
| SGLang Engine | 作为 reader 映射 committed 权重并执行 rollout |

角色关系是：

```text
Owner  = GMS
Writer = Weight Cache Daemon
Reader = SGLang Engine
```

这和共卡方案的核心区别只是 writer 不同：

```text
共卡：   Writer = Engine
非共卡： Writer = daemon
```

### 12.3 为什么非共卡场景让 daemon 做 writer

共卡时，Trainer 本来就把权重发给同组 GPU 上的 Engine，Engine 也已经有完整的 `load_weights()` 和更新逻辑，所以可以直接做 writer。

非共卡时，daemon 更适合做 rollout 侧的稳定入口：

- daemon 可以在 Engine 尚未启动时接收和预热权重；
- Engine 崩溃或滚动升级不会中断 Trainer 的权重投递目标；
- 每个 daemon rank 可以作为对应 rollout GPU/TP rank 的稳定接收端；
- daemon 了解 SGLang tensor manifest、TP/PP shard、量化和模型后处理；
- daemon 可以加入 Trainer 建立的 NCCL/RDMA 权重传输链路。

因此 daemon 的定位从当前 PR 的“加载一次 checkpoint 并长期持有显存”，演进为“rollout 侧的模型感知权重发布者”。

### 12.4 一张 rollout GPU 上的进程关系

典型的一张 rollout GPU 上有：

```text
Rollout GPU 0
  ├─ GMS Server 0
  │    └─ 持有 TP rank 0 的 physical weights
  ├─ Weight Cache Daemon rank 0
  │    └─ 接收并发布 TP rank 0 权重
  └─ SGLang Engine rank 0
       └─ 映射 TP rank 0 权重并 rollout
```

TP=4 时，每张 rollout GPU 都有各自的 GMS、daemon rank 和 Engine rank：

```text
Trainer W1
  ├─ shard 0 -> daemon 0 -> GMS 0 -> Engine rank 0
  ├─ shard 1 -> daemon 1 -> GMS 1 -> Engine rank 1
  ├─ shard 2 -> daemon 2 -> GMS 2 -> Engine rank 2
  └─ shard 3 -> daemon 3 -> GMS 3 -> Engine rank 3
```

GMS Server 之间不直接协调模型版本。跨 TP rank 的“所有 shard 都写完后才能发布 W1”需要由 daemon/控制面进行 barrier 和版本协调。

### 12.5 第一次启动 W0

第一次启动时，daemon 是 writer：

```text
1. 每个 rollout GPU 的 GMS Server 启动
2. 每个 daemon rank 获取对应 GMS 的 RW 权限
3. daemon 从 checkpoint 加载 W0，或从 Trainer 接收初始 W0
4. daemon 执行对应 rank 的 sharding、量化和后处理
5. daemon 通过 RW mapping 将最终 tensor 写入 GMS allocation
6. 所有 rank 完成后，发布 W0/commit
7. Engine 以 RO 权限映射 W0
8. 开始 rollout
```

与共卡方案相比，差别不是 GMS，而是谁调用模型 loader：

```text
共卡首次发布：Engine -> GMS
非共卡首次发布：daemon -> GMS
```

### 12.6 Trainer 产生 W1 后的完整更新流程

假设 Engine 正在使用 committed W0 rollout：

```text
第一步：控制面要求 Engine 暂停新的 rollout
第二步：等待正在执行的 forward 完成并同步 CUDA stream
第三步：Engine 解除 W0 的 RO mapping，保留需要复用的 VA
第四步：所有 reader 释放 RO 后，daemon 获得 GMS RW 权限
第五步：Trainer 通过 NCCL/RDMA 将 W1 推送给各 daemon rank
第六步：daemon 按模型规则将 W1 写入 GMS-managed allocation
第七步：执行必要的量化、scale 生成、融合和模型后处理
第八步：所有 daemon rank 确认 W1 shard 完整且一致
第九步：daemon 发布/commit W1，RW 阶段结束
第十步：Engine 重新获得 RO，并在保留的 VA 上 remap W1
第十一步：复用已有 cache/version 收尾逻辑并恢复 rollout
```

VMM 在这里仍然不负责接收或加载 W1。它只负责让 Engine 能够解除 W0 mapping，并尽量在相同 VA 上重新映射 daemon 发布的 W1。

### 12.7 权重数据到底经过哪里

非共卡场景需要把权重数据从训练 GPU 搬到 rollout GPU，所以这一段不是 zero-copy：

```text
Trainer GPU 上的 W1
        │
        │ NCCL / RDMA / network transfer
        ▼
Rollout 侧 daemon 可访问的 source tensor/buffer
        │
        │ daemon model loader / copy / post-process
        ▼
GMS-owned physical weights W1
```

GMS Server 仍然不接收权重 tensor，也不执行 CUDA copy。daemon 通过自己的 RW VMM mapping 直接写入 GMS-owned allocation：

```text
错误理解：daemon -> 把 W1 数据 RPC 给 GMS Server -> GMS Server 写 GPU

实际路径：daemon 的 CUDA copy/kernel -> RW mapping -> GMS physical memory
```

W1 commit 后，daemon 也不再把完整权重复制给每个 Engine。Engine 只取得 VMM FD 和 metadata，重新映射 GMS 中已经存在的 W1。

所以：

```text
Trainer -> rollout GPU：必须传输真实权重数据
GMS -> Engine：只传 FD/metadata，Engine zero-copy 映射
```

### 12.8 Engine、daemon 分别崩溃会怎样

#### Engine 在 W1 commit 后崩溃

```text
GMS 仍持有 committed W1
daemon/manifest 仍能描述 W1
新 Engine 创建 meta model
新 Engine 重新映射 W1
```

这和共卡方案一样，可以跳过重新从训练侧或磁盘传输权重。

#### daemon 在 W1 commit 后崩溃

在 GMS-owned 方案中，physical W1 不属于 daemon。daemon 的 mapping 和进程退出后，GMS 仍持有 committed allocation。daemon 可以重启并重新连接 GMS；已经映射 W1 的 Engine 也不应因为 daemon writer 消失而自动丢失权重。

这与当前 PR #27139 不同：当前 daemon 同时是 physical owner，daemon 退出会让 Engine 的传统 CUDA IPC mapping 变得不安全。

#### daemon 在写 W1 中途崩溃

W1 尚未 commit，不能把这份半成品交给 Engine。上一版 W0 是否仍可恢复、active layout 如何回收以及各 TP rank 如何统一失败，是 RFC 实现阶段仍需明确的事务和失败语义。

#### GMS Server 崩溃

GMS 是 physical owner，它自身的故障恢复不等于 Engine/daemon 重启恢复。如何让 owner 的 allocation 和 metadata 跨 GMS 故障继续可发现，是更底层的可靠性问题，不应假设当前 RFC 已经完全解决。

### 12.9 共卡和非共卡放在一起比较

| 对比项 | 共卡 RL | 非共卡 RL |
|---|---|---|
| Trainer 与 rollout GPU | 同一组或本地共置 | 不同 GPU/节点 |
| Writer | SGLang Engine | Weight Cache Daemon |
| Physical owner | GMS | GMS |
| Reader | SGLang Engine | SGLang Engine |
| Trainer 权重发送目标 | Engine | daemon |
| Trainer 到 rollout 的传输 | 现有本地更新路径 | NCCL/RDMA/网络传输 |
| 是否经过 Weight Cache Daemon | 否 | 是 |
| 为什么这样选 writer | Engine 已有模型更新逻辑，路径最短 | daemon 是独立于 Engine 的稳定接收和发布端 |

两种方案使用相同的 owner/lease/remap 思路，只是在不同部署下选择离 Trainer 数据路径最合理的 writer。

### 12.10 与当前 PR #27139 的区别

当前 PR 的 daemon 是：

```text
Owner + Writer
从 checkpoint 加载一次
自己长期持有 CUDA allocation
通过传统 CUDA IPC 给 Engine 只读共享
```

非共卡 RFC 中的 daemon 是：

```text
Writer + Publisher
持续接收 Trainer 的 W0/W1/W2...
通过 RW mapping 写 GMS-owned allocation
发布新 epoch 后让 Engine 重新映射
自身不再决定 physical memory 生命周期
```

所以这不是简单地给现有 daemon 增加一个网络接收接口，而是将 physical owner 拆给 GMS，并让 daemon 专注于模型感知的权重接收、处理和发布。

### 12.11 当前还缺哪些实现

这一方案至少还需要落实：

- Trainer 到 daemon 的 NCCL/RDMA 更新协议；
- daemon writer 模式和 GMS allocator 集成；
- Engine reader 的 pause/unmap/remap 协议；
- TP/PP 多 rank 的统一版本和 commit barrier；
- 量化模型更新时的 scale、layout 和后处理；
- 更新失败、daemon 崩溃和部分 rank 成功时的回滚语义；
- `CacheConfig`/manifest 的版本与一致性检查；
- 解除当前 Weight Cache 模式对 `update_weights_*` 的拒绝。

因此它仍是清晰的架构方向，而不是少量参数即可开启的现成功能。

### 12.12 本章结论

非共卡 RL 的核心流程可以压缩成：

```text
Trainer 在训练集群产生 W1
        -> 通过 NCCL/RDMA 推给 rollout 侧 daemon
        -> daemon 作为 writer 将 W1 写入 GMS physical memory
        -> commit W1
        -> Engine 作为 reader 映射 W1 并 rollout
        -> Engine 崩溃后重新映射 GMS 中的 W1
```

一句话总结：

> 共卡时让 Engine 直接做 writer；非共卡时让 daemon 做稳定的远端权重接收和发布者；两者都让 GMS 做 physical owner，从而使最新 committed rollout 权重不依赖某个 Engine 或 daemon 的生命周期。

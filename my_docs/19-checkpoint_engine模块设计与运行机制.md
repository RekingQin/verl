# 19 · checkpoint_engine 模块设计与运行机制（Actor→Rollout 权重同步统一层）

> 返回索引：[README.md](README.md)
> 相关文档：[15-训练后端与优化器配置](15-训练后端与优化器配置.md)、[16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md)、[17-workers模块设计与运行机制](17-workers模块设计与运行机制.md)、[18-models模块设计与运行机制](18-models模块设计与运行机制.md)

`verl/checkpoint_engine/` 是 verl 里**"训练侧 → 推理侧权重同步的统一抽象层"**——它把"actor 训练完一步 → rollout 引擎（vLLM/SGLang/TRTLLM）拿到新权重"这条链路的**通信后端**做成可插拔的策略池，覆盖 NCCL/HCCL/NIXL/Mooncake/Kimi/Delta 六种传输方式，让 Trainer 只关心"发权重"，具体走哪种传输由 YAML 配置决定。

```
verl/checkpoint_engine/
├── README.md                       ← 官方说明 (含 benchmark 表)
├── __init__.py                     ← 可选后端 try/except import + 错误记录
├── base.py                         ← ⭐ 五大抽象: CheckpointEngine / Registry / Manager / Worker / TensorMeta
│
├── nccl_checkpoint_engine.py       ← ⭐ NCCL 后端 (all_gather + broadcast)
├── hccl_checkpoint_engine.py       ← 华为 HCCL 后端 (注册名也是 "nccl", 靠 device 派发)
├── nixl_checkpoint_engine.py       ← NVIDIA NIXL 后端 (ring p2p, 支持异构硬件+弹性)
├── mooncake_checkpoint_engine.py   ← 月之暗面 Mooncake Transfer Engine 后端
├── kimi_checkpoint_engine.py       ← Kimi 特化: Mooncake P2P + NCCL/HCCL broadcast
├── delta_checkpoint_engine.py     ← ⭐ Delta 后端: 只发变化的 (position, value) 对
│
└── delta_sync/                     ← Delta 侧的通用原语（不含通信）
    ├── __init__.py
    ├── encode.py                   ← DeltaParam / DeltaFlush / checksum 数据结构
    └── sparse_gather.py            ← shard_delta_indices + gather_slot_entries_to_rank0
```

## 19.1 定位：为什么需要 checkpoint_engine

在 RL 训练循环里，一次典型的 step 长这样：

```
Trainer: rollout → reward → advantage → update_actor → 【weight sync ← 本模块】→ next step
```

**"weight sync"** 这一步（把新的 actor 权重推给 rollout 服务器）看似简单，实际上要处理五种不同的场景：

| 场景 | 需要什么 |
|---|---|
| Actor + rollout **colocate 同进程**（V1 sync）| **零拷贝**，直接把生成器交给 rollout `update_weights_from_tensor` |
| Actor + rollout **分离但同集群**（V1 separate_async）| NCCL/HCCL 全 rank 参与的 all_gather+broadcast |
| **弹性 rollout**（rollout 可动态扩缩）| NIXL 的 ring p2p，避免固定 group |
| **异构硬件 rollout**（H100 训练 + A100 推理）| NIXL 或 Mooncake，跨硬件传输 |
| **省带宽的稳态同步**（模型参数每步只变一小部分）| Delta：只发变化的 `(index, value)` 对 |

`checkpoint_engine` 的目标：**六种后端一套接口 `send_weights` / `receive_weights`**，Trainer 一行 YAML `checkpoint_engine.backend=nccl/nixl/mooncake/kimi_ckpt_engine/delta_sharded` 切换。

## 19.2 全景架构：Actor / Rollout / CE 三方协作

`base.py:CheckpointEngineManager` docstring 里画的关系图（简化版）：

```
        ┌─────────────── Actor 侧 (训练 world) ───────────────┐          ┌────────────── Rollout 侧 (推理 world) ─────────────┐
        │                                                     │          │                                                    │
        │  ┌──── Ray Actor 进程 (ActorRolloutRefWorker) ────┐  │          │  ┌──── Ray Actor 进程 (CheckpointEngineWorker) ─┐  │
        │  │  ModelEngine (FSDP/Megatron/veomni/...)        │  │          │  │  CheckpointEngine (与 rollout worker 同进程)   │  │
        │  │    │ get_per_tensor_param()                     │  │          │  │    │ receive_weights → 生成 (name, tensor)     │  │
        │  │    v                                            │  │          │  │    v                                          │  │
        │  │  CheckpointEngine                               │  │          │  │  ServerAdapter (vLLM/SGLang/TRTLLM)           │  │
        │  │    │ send_weights(generator)                    │  │  一份卡  │  │    │ update_weights_from_tensor (CUDA IPC)     │  │
        │  │    │                                            │  │  各一份  │  │    v                                          │  │
        │  │    └──── NCCL group (across all ranks) ─────────┼──┼─────────>│  vLLM/SGLang WorkerProc 拿到新权重                │  │
        │  └────────────────────────────────────────────────┘  │          │  └───────────────────────────────────────────────┘  │
        │                                                       │          │                                                     │
        │  由 ActorRolloutRefWorker.update_weights 触发         │          │  由 CheckpointEngineWorker.update_weights 触发       │
        └──────────────────────────────────────────────────────┘          └────────────────────────────────────────────────────┘
                                                    ▲                                          ▲
                                                    │                                          │
                                                    └── 都由 Driver 的 CheckpointEngineManager.update_weights() 编排 ──┘
```

**关键概念澄清**（[17-workers模块设计与运行机制](17-workers模块设计与运行机制.md) 已提及）：
- **Actor 侧 CE** 与 ModelEngine **同进程**（在 `ActorRolloutRefWorker` 里由 `self.checkpoint_engine` 持有）
- **Rollout 侧 CE** 与 rollout worker **同进程**（`CheckpointEngineWorker` 与 SGLang WorkerProc colocate 在同一 GPU 上，通过 CUDA IPC 传递）
- **两侧的 CE 通过 NCCL/HCCL/NIXL/Mooncake 通信**——这条 wire 是**跨进程、跨节点**的

## 19.3 base.py：五大抽象类

`base.py` 提供整个模块的骨架，一共 5 个核心概念：

```
┌─────────────────────────────────────────────────────┐
│ TensorMeta (dataclass)                               │
│   name / shape / dtype / chunk_offset / chunk_size  │
│   bucket 内每片张量的元数据                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ CheckpointEngineRegistry                             │
│   _registry: {backend_str: EngineClass}              │
│   _import_errors: {module_name: ImportError}         │
│   @register("nccl") 装饰器 + get/new                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ CheckpointEngine (ABC) - 传输后端抽象                │
│   wire_format = "named_tensors" | "delta_flush"      │
│   prepare()                    → 分配 buffer + RDMA  │
│   build_topology(...)          → 分配 rank + master   │
│   init_process_group(...)      → NCCL/NIXL 组初始化   │
│   send_weights(generator)      → 发权重              │
│   receive_weights() → generator → 接权重              │
│   finalize()                   → 释放                │
└─────────────────────────────────────────────────────┘
    ├── ColocatedCheckpointEngine (naive, 零通信)
    ├── NCCLCheckpointEngine ("nccl")
    ├── HCCLCheckpointEngine ("nccl" on NPU)
    ├── NIXLCheckpointEngine ("nixl")
    ├── MooncakeCheckpointEngine ("mooncake")
    ├── KIMICheckpointEngine ("kimi_ckpt_engine")
    └── DeltaShardedCheckpointEngine ("delta_sharded", 继承 NCCL)

┌─────────────────────────────────────────────────────┐
│ CheckpointEngineWithCache (可选扩展 ABC)             │
│   多一个 get_weights() 从本地 cache（shm/disk）取     │
│   用于 partial rollout 场景（Laminar 论文）           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ CheckpointEngineWorker (Worker) - Ray Actor 外壳     │
│   持有: self.checkpoint_engine + self.server_adapter │
│   @register(ONE_TO_ALL) update_weights(global_steps) │
│      → CE.receive_weights → server_adapter.update    │
│   @register(DP_COMPUTE) execute_checkpoint_engine(   │
│      method, *args, **kwargs)  # 通用方法转发         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ CheckpointEngineManager - Driver 侧编排器             │
│   __init__(config, actor_wg, replicas)               │
│   build_process_group(rollout_wg)                    │
│   update_weights(global_steps) ★ 主流程 (下节详解)   │
│   sleep_replicas / wake_up_replicas                  │
│   abort_replicas / resume_generation_replicas        │
│   release_kv_cache / resume_kv_cache                 │
│   add_replicas / remove_replicas (弹性)              │
└─────────────────────────────────────────────────────┘
```

## 19.4 CheckpointEngineManager.update_weights：一次同步的 8 步编排

`base.py:CheckpointEngineManager.update_weights`（约 :450）是 Driver 触发一次权重同步的**总入口**，编排逻辑非常清晰：

```python
async def update_weights(self, global_steps: int = None):
    # 0. naive 后端（colocate 同进程）走捷径
    if self.backend == "naive":
        ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode="naive"))
        return {}
    
    # 1. abort 所有 in-flight 请求 (partial rollout: 中间态存回 TransferQueue)
    await self.abort_replicas()
    
    # 2. 把所有 replica 的 CheckpointEngineWorker 合并成一个临时 WorkerGroup
    workers = []
    for replica in self.replicas:
        workers.extend(replica.workers)
    rollout = RayWorkerGroup(worker_handles=workers, ...)
    
    # 3. 释放 rollout 侧的 kv_cache 显存 (weights 保留原地, sync 直接写回)
    await self.release_kv_cache_replicas()
    
    # 4. 建立通信拓扑 (下节详解 build_process_group)
    self.build_process_group(rollout)
    
    # 5. 双向并发 update_weights: actor 侧发 + rollout 侧收
    results = ray.get(
        actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
        + rollout.update_weights(global_steps=global_steps)
    )
    sync_metrics = {}
    for result in results[: actor_wg.world_size]:
        if isinstance(result, dict):
            sync_metrics.update(result)   # delta 后端会返回 nnz/wire_bytes 等指标
    
    # 6. 释放 CE 侧的 bucket 内存 + destroy process group
    ray.get(
        actor_wg.execute_checkpoint_engine(["finalize"] * actor_wg.world_size)
        + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
    )
    
    # 7. 恢复 kv_cache
    await self.resume_kv_cache_replicas()
    
    # 8. 恢复 rollout 生成 (处理之前 abort 的请求 partial resume)
    await self.resume_generation_replicas()
    
    return sync_metrics
```

**关键设计**：
- **abort + kv_cache 释放 + weight sync + kv_cache 恢复 + resume** 是标准的 5 段式舞蹈，保证 partial rollout（[13-异步训练方案对比](13-异步训练方案对比.md)）里的**未完成请求可以断点续跑**
- **步骤 5 用 `ray.get(actor_calls + rollout_calls)`** 并发触发两侧，NCCL group 是"阻塞式"的，两侧不同时发就会 hang
- **步骤 6 `finalize`** 释放 GPU bucket——如果不释放，vLLM/SGLang 后续做 kv_cache 分配可能看不到这些空间

## 19.5 build_process_group：通信拓扑动态建立

CE 的**核心难点**：actor 侧有 N 个 rank，rollout 侧有 M 个 rank，怎么把它们串成一个 NCCL/HCCL group？

`build_topology`（`CheckpointEngine.build_topology` 的抽象契约）就是把这个问题分成三步：

```
build_process_group(rollout):
    # 1. 所有 worker 各自 prepare (分配 buffer + 返回 metadata)
    metadata = ray.get(
        actor_wg.execute_checkpoint_engine(["prepare"] * actor_wg.world_size)
        + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
    )
    
    # 2. 每个 backend 自己算 topology
    actor_wg_kwargs, rollout_kwargs = self.backend_cls.build_topology(
        actor_wg.world_size, rollout.world_size, metadata
    )
    #   返回的 kwargs 长这样:
    #   actor_wg_kwargs = {
    #       "rank": [0, 1, ..., N-1],                      # 每个 actor rank 在 group 内的 rank
    #       "world_size": [N+M, N+M, ..., N+M],
    #       "master_metadata": [master_meta] * N,          # ZMQ / NIXL agent 元数据
    #       "num_senders": [num_senders] * N,              # NCCL 特有: 有几个 actor rank 参与广播
    #   }
    #   rollout_kwargs = 类似, rank 从 num_senders 开始
    
    # 3. 广播这些 kwargs, 各 rank 各自 init_process_group
    ray.get(
        actor_wg.execute_checkpoint_engine(method="init_process_group", **actor_wg_kwargs)
        + rollout.execute_checkpoint_engine(method="init_process_group", **rollout_kwargs)
    )
```

**巧妙之处**：
- `build_topology` 是 `@classmethod`，不需要实例——因为在 Driver 侧调用时，各个 worker 还没有 CE 实例
- **每种后端 override 自己的 `build_topology`**，例如：
  - **NCCL**：只让 actor rank 0（+ 每节点各 1 relay rank）参与 group，其他 rank 通过 `_relay_weights` 转发（省 NCCL 组内 rank 数）
  - **NIXL**：算出 ring 拓扑 (`actor_i → rollout_j`)，每对 rank 是 p2p
  - **Delta**：直接继承 NCCL 的 topology（复用 all_gather+broadcast）

## 19.6 六种后端设计对比

### 19.6.1 naive (`ColocatedCheckpointEngine`)

Actor 和 rollout **同进程**（V1 sync 模式 + hybrid rollout），根本不需要跨进程通信：

```python
def send_weights(self, weights, global_steps=None):
    self.weights = weights           # 就存着生成器
def receive_weights(self, ...):
    yield from self.weights          # 直接 yield 回来
    self.weights = None
```

上层 `ActorRolloutRefWorker.update_weights` 判断 `mode == "naive"` 时会走：
```
per_tensor_param = engine.get_per_tensor_param()
await self.rollout.update_weights(per_tensor_param)   # 直接调 vLLM ServerAdapter
```
——**完全绕过 CheckpointEngineManager**，最省事，只能 colocate。

### 19.6.2 NCCL (`NCCLCheckpointEngine`) — 标杆实现

**通信模式**：`all_gather + broadcast`
- **拓扑**：actor 侧只有 rank 0（或每节点 relay rank）真正参与 NCCL group
- **元数据流**：actor rank 0 用 ZMQ pub-sub 广播 `TensorMeta`（bucket 布局），rollout 各 rank 订阅
- **数据流**：actor 生成器每次填满一个 `bucket_size` bytes 的 buffer → `collective.broadcast(buf, src=0)` → 所有 rollout rank 收到

**核心代码**（`nccl_checkpoint_engine.py:315`）：
```python
async def send_weights(self, weights, global_steps=None):
    assert self.rank < self.num_senders
    if self.rank < 0:                             # 未参与 group 的 actor rank 也要遍历 generator
        for _n, _w in weights: pass
        return
    if self.rank > 0:                             # non-master sender: 转发到 rank 0
        await self._relay_weights(weights)
        return
    # rank 0: 打包成 bucket, 广播
    send_buf, recv_buf = self.send_buf, self.recv_buf
    broadcast_op = None
    async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
        ...
        collective.broadcast(send_buf, src=0, group_name=self.group_name)
        ...
```

**为什么 actor 侧不让所有 rank 都进 group？**
- FSDP world size 可能是 8，rollout world size 可能是 32，全放一个 NCCL group 里 = 40 rank 的巨大 group，广播效率反而下降
- 用 `num_senders`（通常 = actor world_size 除以某个因子）＋ `_relay_weights` 折叠到 rank 0，减少 group size

### 19.6.3 HCCL (`HCCLCheckpointEngine`)

**几乎与 NCCL 同构**（`hccl_checkpoint_engine.py`），最大区别：
```python
@CheckpointEngineRegistry.register("nccl")     # ★ 注册的名字也是 "nccl"!
class HCCLCheckpointEngine(CheckpointEngine):
```

`__init__.py` 的注释说明：**HCCL 在昇腾 NPU 环境下注册 `"nccl"` 名字**，与 NVIDIA NCCL 互斥（同一进程只会 import 到其中一个）。上层 YAML 只需要写 `backend=nccl`，实际用哪个由**硬件 + import 环境**决定——与 `EngineRegistry` 用 device 派发相同的哲学。

### 19.6.4 NIXL (`NIXLCheckpointEngine`)

**通信模式**：`all_gather + ring p2p`
- 底层用 NVIDIA NIXL (multi-transport 抽象)，支持 **UCX / UCCL / Mooncake** 多种传输
- 拓扑不是 group broadcast，而是 **actor rank i ↔ rollout rank ring(i)** 一对一 p2p
- 天然支持**弹性伸缩**：加/减 rollout replica 时只需更新 ring 邻居关系，不用重建 group

**关键类**（`nixl_checkpoint_engine.py`）：
- `NixlAgent`：封装 nixl `plugin_manager` + zmq 消息通道
- `ReadableOperation` / `ReadOperation`：readable = actor 发起注册（"我有这些数据可读"），read = rollout 主动 pull

**适用场景**：
- Rollout 需要弹性扩缩（例如推理压力大时临时加机器）
- 异构硬件（actor 在 H100，rollout 在 A100 / L40 / A10）
- Rollout 容错（一台掉线不影响其他）

### 19.6.5 Mooncake (`MooncakeCheckpointEngine`)

**通信模式**：Mooncake Transfer Engine (`mooncake.engine.TransferEngine`)
- 基于 Mooncake 项目的 KV Cache Transfer 库
- 用**注册的 CPU/GPU 内存段** + **p2p read**
- 每次同步：actor 端 register memory 段（含 metadata "session id"），rollout 端 read

**代码结构**（`mooncake_checkpoint_engine.py`）非常紧凑（<250 行）：`prepare` 分配 GPU bucket + `TransferEngine.register_memory`，`send_weights` 填 bucket + 通过 ZMQ 通知 rollout 拉取，`receive_weights` 调 `TransferEngine.transfer_sync_read` 完成传输。

### 19.6.6 Kimi (`KIMICheckpointEngine`)

**混合传输**：`Mooncake p2p + NCCL/HCCL broadcast`

流程（`kimi_checkpoint_engine.py:321`）：
```
Actor 每个 rank:
   1. 把自己那份权重 offload 到 CPU
   2. Mooncake TransferEngine 把 CPU 权重 p2p 发给指定 rollout replica 的 leader worker

Rollout leader worker:
   3. 收到 CPU 权重 → 上传回 GPU
   4. NCCL/HCCL broadcast 给同 replica 的其他 workers
```

**为什么这么绕？**
- Mooncake p2p **不受 NCCL group 约束**，可以跨异构硬件
- 但 Mooncake 一次 p2p 只能 1v1，效率低
- 于是**跨 world 用 p2p，同 replica 内用 broadcast**——鱼与熊掌兼得

Kimi 后端的另一大特色（README 里强调）：**每次同步自动落盘 checkpoint**，天然支持"训练崩了从最近 rollout 权重恢复"。

### 19.6.7 Delta (`DeltaShardedCheckpointEngine`)

**继承自 `NCCLCheckpointEngine`**，wire 还是 NCCL，但**载荷是稀疏 `(position, value)` 对**，不是完整张量。

**wire_format**：
```python
wire_format = "delta_flush"    # 与 named_tensors 相对
```

告诉 rollout 侧的 `server_adapter.update_weights(...)` 走 **SGLang 的 `custom-weight-loader` hook**（[verl.workers.rollout.sglang_rollout.delta_loader](/Users/chenruiqin/Documents/AICoding/verl/verl/workers/rollout/sglang_rollout/delta_loader.py)），不解全量、只做**稀疏 masked apply**。

**运行模式**（`delta_checkpoint_engine.py:578`）——一个 "seed → steady" 状态机：

```
send_weights(weights, global_steps):
    if first_call or force_full_resync:
        # SEED PATH: 全量同步 (values-only wire, 复用 backend 的 HF export)
        self._send_full_seed(weights, global_steps)
        # 之后调 engine.prime_delta_snapshots() 建 pinned CPU 快照
    else:
        # STEADY PATH: 只发变化的元素
        # 后端 iter (name, spec, delta_lidx, delta_lval) 序列
        # gather_slot_entries_to_rank0 收集到 rank 0
        # 打包成 DeltaFlush → NCCL broadcast
        self._steady_delta_send(weights, global_steps)
```

**为什么 wire_format 只在 sglang 上支持？**（`base.py:CheckpointEngineWorker.__init__` 里有 assert）
- vLLM/TRTLLM 的 `update_weights_from_tensor` 接口只接完整张量
- SGLang 提供 `--custom-weight-loader` hook，可以用户自己写 apply 逻辑
- 未来会加通用 sparse apply 接口

## 19.7 Delta 的核心机制（`delta_sync/`）

### 19.7.1 数据结构（`encode.py`）

三个原语：

**`DeltaParam`**：一个参数的**清单条目**
```python
@dataclass
class DeltaParam:
    name: str              # 参数名 (HF 命名)
    dtype: str             # 参数 dtype (str 便于跨进程)
    shape: list[int]       # 参数 shape (rollout 侧要用它 reshape)
    pos_start: int         # 在 __positions__ blob 里的字节偏移
    pos_end: int
    pos_width: int         # 2 或 4 (uint16/int32 position)
    val_start: int         # 在 __values__ tensor 里的元素偏移
    val_end: int
```

**`DeltaFlush`**：一次"就绪、可发"的 bucket
```python
@dataclass
class DeltaFlush:
    encoding: "indices"                    # 唯一编码方式
    params: list[DeltaParam]               # 清单 (走 zmq side-channel)
    positions_cpu: torch.Tensor            # 名字叫 cpu 实际在 GPU (省一次 D2H)
    values_gpu: torch.Tensor               # bucket 里所有变化元素的值
    checksum: int                          # 数据完整性校验
    
    @property
    def nnz(self) -> int: return self.values_gpu.numel()
    @property
    def wire_bytes(self) -> int: return positions.numel() + values.numel() * itemsize
```

**`checksum`**：用 `torch.hash_tensor` (XOR-reduce over uint64) 做一致性校验，一次 GPU 归约 + 一次 `.item()` 同步——sender 发前算，receiver 收后算，不一致就报错。

### 19.7.2 sparse gather（`sparse_gather.py`）

**核心问题**：FSDP2 下每个 rank 只持有参数的一片 shard，怎么把"每片 shard 各自 diff 后的稀疏索引"高效收到 rank 0？

**朴素方案**（默认 delta 路径）：所有 rank `full_tensor()` all-gather 完整参数 → rank 0 diff → broadcast。缺点：wire 是 100% 参数量。

**本模块方案**：每 rank 只 diff 自己的 shard → gather 稀疏 delta → wire 是 sparsity ratio（~1-3%）。

**`shard_delta_indices(local_new, local_snap, offset)`**：
```python
mask = local_new.view(int_dtype) != local_snap.view(int_dtype)   # bytewise diff, dtype 无关
local_idx = mask.nonzero(as_tuple=False).view(-1)
values = local_new[local_idx]
global_idx = local_idx.to(torch.int64) + offset                  # shard→全局参数的索引偏移
return global_idx, values
```

**`gather_slot_entries_to_rank0`**：批量稀疏 gather（关键优化）
- 一个 group 内 K 个参数，如果用 K 次 gather 会有 K 次 host sync，太慢
- 这个函数**一次 all_gather 交换 K×world 长度矩阵，两次 padded gather 移数据**
- 用 `max_round_bytes` 分子轮，保证单轮 padded blob 不超预算

### 19.7.3 sender 侧的 bucket 流水线（`delta_checkpoint_engine.py`）

数据流：
```
backend HF delta ENTRY (slots, dtype, counts, hf_idx, hf_val, group)   # 各 rank 各自 diff 出的稀疏 delta
    │
    v _GatherQueue                    # 攒够 batch_k 个参数再 gather (减少 collective 次数)
    │
    v per-SLOT delta on rank 0        # rank 0 拿到每个 slot 的 (idx, val)
    │
    v _bucket_* 装配成 _FlushPiece
    │
    v _FlushBucket                    # 攒够 cap bytes 一 flush
    │
    v DeltaFlush (params + positions + values + checksum)
    │
    v _publish_flush                  # NCCL broadcast + zmq 广播清单
    │
    v rollout 侧 delta_loader.py 稀疏 apply
```

**关键"一次前瞻"设计**（`_FlushBucket`）：
- 每次装满 bucket 后**先不 emit**，多攒一个 piece 再决定
- 因为 emit 时需要标 `is_last` 位（rollout 侧根据这个决定要不要 sync barrier）
- 一次前瞻确保 `is_last` 判断正确，避免"看似是最后一个但其实还有一个"

## 19.8 数据流：完整的一次权重同步时序

```
┌───────────────────────────────────────────────────────────────────┐
│  Driver (PPOTrainer._step_once 内)                                 │
│    checkpoint_manager.update_weights(global_steps)                 │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              v
   Manager.update_weights (base.py:CheckpointEngineManager)
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      v                       v                       v
   ┌──────────┐         ┌──────────┐         ┌──────────────┐
   │ 1. abort │         │ 3. rele- │         │ 5. actor+rol │
   │ replicas │         │ ase kv   │         │ update_weight│
   │ (partial │         │ cache    │         │ (并发)         │
   │ rollout) │         └──────────┘         └──────┬───────┘
   └──────────┘                                     │
                                                    v
                          ┌───── Actor 侧 ──────┐  ┌───── Rollout 侧 ─────┐
                          │ actor_wg.update_    │  │ rollout.update_weights│
                          │ weights(...) 内部:  │  │  (临时 wg)          │
                          │                     │  │                      │
                          │ if backend=="delta":│  │ CheckpointEngine     │
                          │   engine.get_per_   │  │ Worker.update_       │
                          │   tensor_param_     │  │ weights:             │
                          │   delta_shard()     │  │  weights = ce.       │
                          │ else:               │  │    receive_weights() │
                          │   engine.get_per_   │  │  await adapter.      │
                          │   tensor_param()    │  │    update_weights(   │
                          │                     │  │      weights,        │
                          │ ce.send_weights(    │  │      wire_format=...)│
                          │   generator)        │  │                      │
                          │                     │  │ SGLang/vLLM Worker:  │
                          │  ← NCCL/HCCL/NIXL/  │  │   CUDA IPC 拷进      │
                          │    Mooncake wire ──→│  │   模型参数           │
                          └─────────────────────┘  └──────────────────────┘
                              │                       │
                              └───────────┬───────────┘
                                          v
                          ┌────────────────────────────────┐
                          │ 6. finalize (释放 bucket)       │
                          │ 7. resume_kv_cache_replicas     │
                          │ 8. resume_generation_replicas   │
                          │    (partial rollout 断点续跑)   │
                          └────────────────────────────────┘
```

**关键点**：
- 步骤 5 里 `actor_wg.update_weights` 和 `rollout.update_weights` **必须并发**——NCCL group 里两侧不同时进就死锁
- **wire_format** 从 CE 传到 adapter：`getattr(self.checkpoint_engine, "wire_format", "named_tensors")` — delta 是 `"delta_flush"`，其他都是 `"named_tensors"`
- **谁调 `engine.get_per_tensor_param`？** 是 `ActorRolloutRefWorker.update_weights`（见 [17-workers模块](17-workers模块设计与运行机制.md#1726-update_weights-的模式分派)），不是 CE 内部——**backend 决定 shape，CE 决定 wire**

## 19.9 backend 选型指南

结合 README 里的 benchmark 表和场景表：

| 你的场景 | 推荐 backend | 理由 |
|---|---|---|
| Actor+rollout **colocate 同进程**（V1 sync） | `naive` | 零拷贝零通信 |
| Actor+rollout **分离但同硬件同集群**（V1 separate_async） | `nccl` (GPU) / `nccl` on NPU 自动 → HCCL | 8 GB/s 带宽，最成熟 |
| **弹性 rollout**（跑着跑着加机器） | `nixl` | ring p2p，动态调 topology |
| **异构硬件 rollout**（H100 训 + A10 推） | `nixl` 或 `mooncake` | 多 transport 支持 |
| **要求每步落盘 checkpoint**（Kimi 场景） | `kimi_ckpt_engine` | 自动持久化 |
| **稳态同步、参数变化少**（RLHF late stage） | `delta_sharded` | wire 缩到 1-3% |
| **只有 SGLang 且模型很大**（30B+） | `delta_sharded` | 目前只 sglang 支持 delta apply |

## 19.10 扩展点：添加新 backend

1. **实现 CheckpointEngine 子类**：
```python
@CheckpointEngineRegistry.register("my_backend")
class MyCheckpointEngine(CheckpointEngine):
    wire_format = "named_tensors"        # 或 "delta_flush"
    
    def prepare(self) -> dict[str, Any]:
        # 分配 GPU/CPU buffer, 返回 metadata (ip:port/agent-name/...)
        ...
    
    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        # 返回 (actor_kwargs, rollout_kwargs)
        ...
    
    def init_process_group(self, **kwargs):
        # 建自定义通信 group
        ...
    
    async def send_weights(self, weights_generator, global_steps=None):
        ...
    
    async def receive_weights(self, global_steps=None):
        ...
    
    def finalize(self):
        # 释放 buffer, 可选 destroy group
        ...
```

2. **在 `__init__.py` 里加 try/except import**：
```python
try:
    from .my_checkpoint_engine import MyCheckpointEngine
    __all__ += ["MyCheckpointEngine"]
except ImportError as e:
    CheckpointEngineRegistry.record_import_error("my_checkpoint_engine", e)
```

3. **YAML 配置切换**：
```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      backend: my_backend
      update_weights_bucket_megabytes: 512
      engine_kwargs:
        my_backend:
          custom_param: value
```

或者**动态从外部包注册**：设置 `checkpoint_engine.custom_backend_module: "my_pkg.my_module"`，`CheckpointEngineWorker.__init__` 会通过 `import_external_libs` 自动 import 使装饰器生效。

## 19.11 关键设计模式与最佳实践

| 模式 | 位置 | 价值 |
|---|---|---|
| **可选依赖 try/except** | `__init__.py` + `_import_errors` 记录 | cupy/nixl/torch_npu 装不上时也能跑；错误可查 |
| **注册表 + `@classmethod build_topology`** | `base.py:CheckpointEngine` | 拓扑计算不需要实例，Driver 侧可算 |
| **wire_format 字段** | `wire_format = "named_tensors" \| "delta_flush"` | rollout 侧 apply 逻辑分流，兼容稀疏与稠密 |
| **同名跨硬件注册** | HCCL 注册 `"nccl"`, YAML 无感切换 | 一份配置跑 GPU / NPU |
| **CE 与 ModelEngine 分离** | `ce.send_weights(engine.get_per_tensor_param())` | wire 换后端不用改训练；backend 换实现不用动 wire |
| **`abort → release_kv → sync → resume_kv → resume` 五段舞** | `Manager.update_weights` | partial rollout 断点续跑保障 |
| **seed / steady 状态机** | `DeltaShardedCheckpointEngine` | 冷启动全量、稳态增量，无缝切换 |
| **一次前瞻 bucket 装配** | `_FlushBucket` | 正确标注 `is_last` 位，配合 receiver sync barrier |
| **`torch.hash_tensor` 校验** | `delta_sync/encode.py:checksum` | 极低开销的 wire 完整性检查 |
| **CUDA IPC 同进程共享** | rollout CE 与 SGLang WorkerProc 同 GPU | 权重传输零拷贝 |

## 19.12 常见陷阱

| 陷阱 | 现象 | 规避 |
|---|---|---|
| Actor / rollout 不并发调 update_weights | NCCL group hang | Manager 里已用 `ray.get([...+...])` 并发触发 |
| 忘 `abort_replicas` 就 sync | in-flight 请求 kv_cache 被覆盖 → 输出乱码 | Manager 已内置 abort→sync→resume 序列 |
| Delta 首次 sync 报"no snapshot" | `prime_delta_snapshots` 没在 seed 之后调 | seed path 内部会自动 prime; 手动路径要显式调 |
| 用 delta 但 rollout 不是 sglang | `NotImplementedError` in `CheckpointEngineWorker.__init__` | 明确 assert；改用 nccl 或换 sglang |
| `custom_backend_module` 没生效 | 自定义 backend 显示 `not registered` | 检查 `import_external_libs` 是否被调用；确保模块 import 时会执行 `@CheckpointEngineRegistry.register` |
| bucket 太小 | 频繁广播，wire 效率低 | `update_weights_bucket_megabytes` 默认 512，大集群可调到 1024-2048 |
| bucket 太大 | GPU OOM（bucket 是 GPU 显存） | 减小 bucket size；启用 delta |
| HCCL 注册的名字是 "nccl" 意外覆盖 NCCL | 同一环境同时有 CUDA+NPU 支持 → import 冲突 | 两者互斥，运行时只装一个 |
| Mooncake / Kimi Ascend 白名单未配 | 通信直接失败 | 参考 README 中的 `HCCL_WHITELIST_*` / `HCCL_INTRA_ROCE_ENABLE=1` 环境变量 |

## 19.13 一图总结

```
                    ┌────────────────────────────────────────────────┐
                    │  verl/checkpoint_engine/                        │
                    │  "Actor→Rollout 权重同步统一层"                 │
                    └──────────────────┬─────────────────────────────┘
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                    │
     ┌───────v────────┐                              ┌────────────v───────────┐
     │   base.py       │                              │  具体后端 (7 种)         │
     │                 │                              │                         │
     │ 五大抽象:        │                              │ ┌── naive (colocate)  │
     │  TensorMeta     │                              │ ├── nccl (GPU 标杆)    │
     │  Registry       │                              │ ├── nccl (NPU=HCCL)   │
     │  CheckpointEng  │◀── 继承/注册                 │ ├── nixl (弹性/异构)   │
     │  Manager        │──── 编排                     │ ├── mooncake          │
     │  Worker (Ray)   │                              │ ├── kimi_ckpt_engine  │
     │                 │                              │ └── delta_sharded ────┼──┐
     │ 通用辅助:        │                              │        (继承 nccl)     │  │
     │  split_weight_  │                              └──────────────────────  │  │
     │    chunks       │                                                       │  │
     │  merge_weight_  │                              ┌────────────────────────v──┐
     │    chunks       │                              │  delta_sync/               │
     └─────────────────┘                              │                            │
                                                     │  encode.py:                │
              ┌──────────────────────────────────────┤    DeltaParam / Flush /    │
              │                                      │    checksum                │
              v                                      │                            │
      ┌───────────────────┐                          │  sparse_gather.py:         │
      │ Manager.update_   │                          │    shard_delta_indices     │
      │   weights (8 步)  │                          │    gather_slot_entries_to_ │
      │                   │                          │      rank0                 │
      │ 1. abort          │                          └────────────────────────────┘
      │ 2. temp wg        │
      │ 3. release_kv     │
      │ 4. build_topology │
      │ 5. actor+rollout  │◀── ce.send_weights (actor)
      │    (并发)          │    ce.receive_weights (rollout)
      │ 6. finalize       │    server_adapter.update_weights(wire_format)
      │ 7. resume_kv      │
      │ 8. resume_gen     │
      └───────────────────┘
```

**核心一句话**：`verl/checkpoint_engine/` 用**"CheckpointEngine 抽象 + Registry 派发 + Manager 编排"**三件套，把 6 种通信后端（NCCL/HCCL/NIXL/Mooncake/Kimi/Delta）统一到 `send_weights` / `receive_weights` / `prepare` / `finalize` 四方法接口下，让 Trainer **一行 YAML 切换传输方式**——从零拷贝的 naive、到经典的 NCCL、再到只发变化元素的 delta_sharded，覆盖从同进程到跨异构硬件的所有权重同步场景。

---

**深入相关模块**：
- [17-workers模块设计与运行机制](17-workers模块设计与运行机制.md#1726-update_weights-的模式分派) — `ActorRolloutRefWorker.update_weights` 如何调用 CE
- [18-models模块设计与运行机制](18-models模块设计与运行机制.md#1849-weight_converterpymcore--hf-权重转换rollout-同步核心) — mcore weight_converter 与 CE 的分工
- [16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md) — `release_kv_cache` / `resume_kv_cache` / `wake_up` 与 rollout 内存管理
- [13-异步训练方案对比](13-异步训练方案对比.md) — sync/colocate_async/separate_async 三种 mode 下的 CE backend 选择

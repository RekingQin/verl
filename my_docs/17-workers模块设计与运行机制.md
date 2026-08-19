# 17 · workers 模块设计与运行机制（engine_workers / engine / rollout / reward_manager / utils）

> 返回索引：[README.md](README.md)
> 相关文档：[10-Ray串联机制详解](10-Ray串联机制详解.md)、[12-decorator与dispatch机制详解](12-decorator与dispatch机制详解.md)、[15-训练后端与优化器配置](15-训练后端与优化器配置.md)、[16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md)

`verl/workers/` 是 verl 的**"执行层"**——所有真正吃 GPU 的计算都发生在这里。Driver（`TaskRunnerV1` + `PPOTrainer`）只是"编导"，Workers 才是"演员"。本文按由外到内的层次拆解：

```
verl/workers/
├── engine_workers.py        ← ⭐ Ray Actor 外壳：TrainingWorker / ActorRolloutRefWorker
├── engine/                  ← ⭐ 训练引擎抽象（FSDP/Megatron/veomni/torchtitan/...）
├── rollout/                 ← ⭐ 生成引擎抽象（vLLM/SGLang/TRTLLM/HF/Naive + Replica/LLMServerManager）
├── reward_manager/          ← ⭐ Reward 计算抽象（naive/batch/dapo/prime）
├── utils/                   ← ⭐ 训练算子：ppo_loss / value_loss / padding 转换
├── engine_workers_tinker.py ←   Tinker-style API 外壳（辅助）
└── config/                  ←   dataclass 配置定义（第 15/16 篇已覆盖）
```

## 17.1 全局分层视图

```
                              Driver (TaskRunnerV1 / PPOTrainer)
                                       │
                       @register 装饰器方法调用（Functor 代理）
                                       │
                        ┌──────────────┼──────────────┐
                        │              │              │
                        v              v              v
     ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │ ActorRolloutRefWorker│  │ TrainingWorker    │  │ Rollout Adapter   │
     │  (Ray Actor 外壳)    │  │  (纯训练 worker)  │  │ (vLLM/SGLang)     │
     │  role = actor+       │  │                   │  │                   │
     │         rollout+ref  │  │  self.engine ─┐   │  │  RolloutReplica  │
     │  子对象:            │  │               v   │  │  + LLMServerMgr   │
     │  ┌─ self.actor  ────┼──┤  BaseEngine       │  │                   │
     │  ├─ self.ref    ────┼──┤   (FSDP/Megatron) │  │                   │
     │  └─ self.rollout─────┼──┤─(BaseRollout: vllm/sglang/trtllm)──────┘
     │                      │
     │  self.checkpoint_engine (delta_sync / nccl / mooncake)
     └──────────────────────┘
                                       │
                                       v
                          verl/workers/reward_manager/*
                          (Driver 端调用，非 Ray Actor)
```

**四层抽象**：

| 层 | 定位 | 谁在调 | 谁在跑 |
|---|---|---|---|
| **engine_workers** | Ray Actor 外壳；对上暴露 `compute_log_prob/update_actor/update_weights` 等 register 方法 | Driver（`RayWorkerGroup` 代理） | Ray Actor 内 |
| **engine** | 训练后端抽象（前向/反向/优化器/权重导出）| engine_workers（`self.engine.train_batch(...)`）| Ray Actor 内（每卡 1 进程）|
| **rollout** | 生成后端抽象（vLLM/SGLang/TRTLLM）+ 权重同步 | engine_workers（`self.rollout.update_weights(...)`）+ agent_loop | 独立 Ray Actor（server 模式）|
| **reward_manager** | reward 计算（数据源分发 + timeout 保护）| PPOTrainer / RewardLoopManager | Driver 或 reward worker |

## 17.2 engine_workers.py（Ray Actor 外壳）

### 17.2.1 两大核心类

`engine_workers.py` 只定义了两个 Worker 类（都在 `verl/workers/engine_workers.py`）：

| 类 | 行数区间 | 角色 |
|---|---|---|
| `TrainingWorker(Worker, DistProfilerExtension)` | :76 | **纯训练 worker**：只做 train / infer / save / load，适用于 SFT/critic 等场景，也被 `ActorRolloutRefWorker` 用作内部对象 |
| `ActorRolloutRefWorker(Worker, DistProfilerExtension)` | :446 | **PPO Hybrid worker**：`actor+rollout+ref` 三合一，`role ∈ {actor, rollout, ref, actor_rollout, actor_rollout_ref}` 决定组合 |

两者都继承 `verl.single_controller.base.Worker`，因而带有 `MAGIC_ATTR` 属性可被 `RayWorkerGroup` 扫描并生成代理方法（详见 [12-decorator与dispatch机制详解](12-decorator与dispatch机制详解.md)）。

### 17.2.2 TrainingWorker：`self.engine` 一个字段拉起整个后端

关键构造代码（`engine_workers.py:83-149`）：

```
__init__(config: TrainingWorkerConfig):
  1. Worker.__init__() 建 torch.distributed 通信组
  2. initialize_global_process_group_ray(timeout_second=None)  # NCCL/HCCL init
  3. set_numa_affinity()                                       # 亲和性
  4. auto_select_engine_optim_fn?                              # 未指定时自动推断
  5. self.engine = EngineRegistry.new(                         # ⭐ 动态派发
       model_type=..., backend=engine_config.strategy,          #   派发到具体后端类
       model_config=..., engine_config=..., optimizer_config=..., 
       checkpoint_config=...)
  6. self._register_dispatch_collect_info(                     # ⭐ 声明 DP 拓扑
       mesh_name="train",
       dp_rank=self.engine.get_data_parallel_rank(),
       is_collect=self.engine.is_mp_src_rank_with_outputs())
  7. self.flops_counter = FlopsCounter(hf_config)              # MFU 计算
```

**要点**：
- 第 5 步是**"策略模式的运行时选择"**：一句 `EngineRegistry.new` 就把 `fsdp/fsdp2/megatron/veomni/torchtitan/automodel/mindspeed/fsdp_turbo` 中的一个具体 Engine 实例化并挂到 `self.engine`；上层永远只见 `BaseEngine` 接口。
- 第 6 步是**声明"这块 worker 属于哪个 DP mesh、我是否 collect"**。`make_nd_compute_dataproto_dispatch_fn(mesh_name="train")` 会读这个信息去切分/聚合数据。

### 17.2.3 训练接口三剑客：`train_mini_batch / train_batch / infer_batch`

所有对外方法都带同款 `register` 装饰器：

```
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"),
          blocking=False)
def train_mini_batch(self, data: TensorDict) -> TensorDict:
    ...
def train_batch(...):    # 类似
def infer_batch(...):    # forward-only
```

**统一执行范式**（以 `train_batch` 为例，:337）：

```
train_batch(data) 内部:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. maybe_fix_3d_position_ids(data)   # mRoPE 修正            │
  │ 2. self.engine.train_mode() 上下文  # load model+opt to GPU  │
  │ 3. self.engine.train_batch(data, self.loss_fn) # 真正训练    │
  │    -> engine.optimizer_zero_grad                             │
  │    -> engine.forward_backward_batch  (micro-batch 循环)       │
  │    -> engine.optimizer_step  (clip + step + inf/nan skip)    │
  │ 4. _postprocess_output(output, ...)  # loss all-reduce +      │
  │    metrics allgather + MFU 计算                              │
  │ 5. 退出 train_mode() -> offload 回 CPU                       │
  │ 6. return TensorDict (仅 metrics + model_output)             │
  └─────────────────────────────────────────────────────────────┘
```

`_postprocess_output`（:180）做四件事：
- **loss all_reduce**：AVG across dp group，得到全局平均 loss
- **metrics allgather**：dp 组内 dict 合并（`allgather_dict_into_dict`）
- **max_memory / cpu_memory / MFU**：性能指标补齐
- 返回 `TensorDict`（Driver 只需 metrics + 少量 model_output）

**关键设计**：`blocking=False` 让 Driver 可以并发发起多组 worker 的调用（如同时训 actor 和 critic），实际阻塞在 `ray.get()` 时刻决定；`dispatch_mode` 让 Driver 传的整批数据自动 chunk 到 DP rank。

### 17.2.4 ActorRolloutRefWorker：Hybrid 三合一的组装术

`ActorRolloutRefWorker.__init__`（:456）只是保存配置和 role 标志，**不做任何 GPU 侧初始化**；真正的初始化在 `init_model()`（:533），被 Trainer 通过 `@register(Dispatch.ONE_TO_ALL)` 集中触发：

```
init_model():   # 广播到每个 rank
  1. if "ref" in role:                                          
       - 合并 ref 配置字段（继承 actor 的 ppo_mini_batch_size 等）
       - self.ref = TrainingWorker(ref_training_config)           # ★ 嵌套
       - self.ref.reset()  -> engine.initialize()
       - self.set_dispatch_collect(mesh_name="ref", ...)          # 独立 mesh
  
  2. if "actor" in role:
       - self.loss_fn = partial(ppo_loss, config=actor_config)    # 或 distillation_ppo_loss
       - self.actor = TrainingWorker(actor_training_config)       # ★ 嵌套
       - self.actor.reset()
       - self.actor.set_loss_fn(self.loss_fn)
       - self.set_dispatch_collect(mesh_name="actor", ...)
  
  3. if "rollout" in role:
       - rollout_device_mesh = init_device_mesh(..., ["dp","infer_tp","infer_pp"])
       - rollout_cls = get_rollout_class(name, mode)              # ⭐ 派发
       - self.rollout = rollout_cls(config, model_config, device_mesh)
  
  4. if "actor" in role:
       - self.checkpoint_engine = CheckpointEngineRegistry.new(
           backend, bucket_size, ...)                              # 权重同步引擎
  
  5. aggressive_empty_cache(force_sync=True)   # 释放显存给 vLLM 用
```

**三个精妙点**：

- **嵌套复用**：`self.actor` 和 `self.ref` 其实是**在同一个 Ray Actor 进程内**创建的 `TrainingWorker` **对象**（不是新 Actor！）——共享同一个 CUDA context、同一个 torch.distributed 分组，但通过 `mesh_name` 让 dispatch 逻辑把它们当成不同的 DP mesh 看待。
- **三个 mesh，三种 dispatch**：`actor` / `ref` / `train` 各自登记，同一个 Ray Actor 可以用不同的 mesh 切数据。见 `compute_ref_log_prob` 用 `mesh_name="ref"`，`update_actor` 用 `mesh_name="actor"`，互不干扰。
- **`aggressive_empty_cache`**：这个动作是为了**让 colocated vLLM 通过 `cudaMemGetInfo` 看到最大可用显存**，随后才能安全 wake_up。这是"colocate mode 显存和平共处"的技术前提（配合 [16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md)）。

### 17.2.5 三种关键的 register 方法与其 dispatch 语义

| 方法 | dispatch_mode | 用途 | 数据流 |
|---|---|---|---|
| `init_model` / `reset` / `to` / `save_ckpt` / `load_ckpt` | `Dispatch.ONE_TO_ALL` | 广播控制指令 | Driver → 所有 rank |
| `compute_log_prob` / `compute_ref_log_prob` | `make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"/"ref")` | DP 切分并发推理 | Driver → chunk → dp rank → concat 回 Driver |
| `update_actor` | `make_nd_compute_dataproto_dispatch_fn(mesh_name="actor")` | DP 切分并发训练 | 同上，最后 collect metrics |
| `update_weights` | `Dispatch.ONE_TO_ALL, blocking=False` | 训练→rollout 权重同步 | 每个 rank 独立执行 |

**为什么 `update_weights` 是 `ONE_TO_ALL`？** 因为 rollout 权重同步本质上是**每个训练 rank 各自把自己那份 shard 交给对应的 rollout worker**——不需要 Driver 端切分数据，`blocking=False` 是为了让 Driver 可以在权重同步的同时做别的编排。

### 17.2.6 `update_weights` 的模式分派（`engine_workers.py:719+`）

```
update_weights(global_steps, mode="auto"):
  effective_mode = mode or config.rollout.checkpoint_engine.backend
  
  ┌─ effective_mode ≠ "naive"（async / disaggregated）
  │  ├── "delta_sharded": 直接把 self.actor.engine 传给
  │  │                    checkpoint_engine.send_weights，由 delta 引擎
  │  │                    自驱动 seed→snapshot→steady 状态机
  │  └── others:          engine.get_per_tensor_param()（全量）
  │                       → checkpoint_engine.send_weights
  │
  └─ effective_mode == "naive"（同步 colocate）
     1. resume_weights (rollout 侧睡眠时释放的 weight 内存重开)
     2. get_per_tensor_param(layered_summon, base_sync_done=True)
     3. self.rollout.update_weights(per_tensor_param, wire_format=...)
```

**要点**：
- **delta_sharded 特殊路径**：不用 `get_per_tensor_param` 走完整导出，而是把整个 `engine` 交给 checkpoint_engine，让它自己去调 `get_per_tensor_param_shard`（seed）或 `get_per_tensor_param_delta_shard`（steady）——这是让"只发变化元素"的核心开关。
- **sglang / vLLM 的 sleep_level 差异**：`is_sglang and sleep_level == 1` 时不 resume weights（SGLang adapter 模式没释放）；vLLM level-1 sleep 用 `CuMemAllocator.sleep(offload_tags=("weights",))` 会释放，必须 resume。

## 17.3 engine/ 子模块（训练后端抽象）

> 本节与 [15-训练后端与优化器配置](15-训练后端与优化器配置.md) 互补：15 篇讲配置，本节讲**代码结构与运行机制**。

### 17.3.1 目录结构

```
verl/workers/engine/
├── __init__.py           # 各后端 try/except 可选导入
├── base.py               # ⭐ BaseEngine / BaseEngineCtx / EngineRegistry
├── spec.py               # ⭐ ShardSpec / BlockPlacement / translate_flat_indices
├── utils.py              # prepare_micro_batches / postprocess / hf_delta_export
│
├── fsdp/                 # FSDP1/FSDP2（参考实现，最完整）
│   ├── transformer_impl.py    # FSDPEngine + WithLMHead + WithValueHead
│   ├── fsdp_turbo_impl.py     # FSDPTurboEngineWithLMHead
│   └── utils.py               # create_device_mesh / get_sharding_strategy / unfuse_moe_params
├── megatron/             # Megatron-Core (TP/PP/CP + Distributed Optimizer)
│   ├── transformer_impl.py
│   └── delta_export.py        # mcore weight converter + delta 导出
├── mindspeed/            # NPU 上复用 megatron backend 的特化
├── veomni/               # ByteDance VeOmni（多模态 + MoE + EP）
├── torchtitan/           # PyTorch TorchTitan（原生 DTensor）
└── automodel/            # NVIDIA NeMo AutoModel
```

### 17.3.2 BaseEngine 接口能力矩阵

`BaseEngine`（`base.py:30`）定义了 6 组能力（**接口即契约**）：

```
┌── 生命周期 ─────────────────────────────────────
│ initialize() → 构造 model+optimizer+scheduler
│ train_mode() / eval_mode() → 返回 ContextManager
│ to(device, model, optimizer, grad) → 手动 offload/load
├── 训练循环 ─────────────────────────────────────
│ train_batch(data, loss_fn)         → 完整训练步 (模板方法)
│ infer_batch(data, loss_fn=None)    → torch.no_grad 前向
│ forward_backward_batch()           → 核心 micro-batch 引擎
│ optimizer_zero_grad / optimizer_step / lr_scheduler_step
├── 分布式拓扑 ────────────────────────────────────
│ get_data_parallel_size / _rank / _group
│ is_mp_src_rank_with_outputs        → dispatch/collect 用
├── 权重导出（rollout 同步核心）────────────────────
│ get_per_tensor_param()             → 全量 (seed / first sync)
│ get_per_tensor_param_shard()       → 分片 (可控内存)
│ get_per_tensor_param_delta_shard() → ⭐ 只发变化元素 (~1%)
│ prime_delta_snapshots()            → 首次快照
├── Checkpoint ──────────────────────────────────
│ save_checkpoint / load_checkpoint  → 后端自己的 CheckpointManager
└── LoRA ────────────────────────────────────────
  disable_adapter() → ContextManager
```

**核心亮点：`train_batch` 是基类实现（不是 abstract！）**

```python
# base.py
def train_batch(self, data, loss_function):
    maybe_fix_3d_position_ids(data)
    self.optimizer_zero_grad()
    outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
    grad_norm = self.optimizer_step()
    if self.is_mp_src_rank_with_outputs():
        outputs["metrics"]["grad_norm"] = grad_norm
    return outputs
```

**这就是"模板方法模式"**：基类固定训练一步的骨架（zero_grad → forward+backward → step），子类只需实现 `forward_backward_batch` / `forward_step` / `optimizer_step` 等填空部分。

### 17.3.3 BaseEngineCtx：上下文管理器统一 offload 语义

```
class BaseEngineCtx (base.py:300):
    __enter__: engine.mode = mode; _context_switch(cuda)
        └── train 模式: to(cuda, model+optim+grad)
        └── eval  模式: to(cuda, model)
    __exit__:  _context_switch("cpu"); engine.mode = None

class EngineTrainModeCtx(BaseEngineCtx):   # 各后端自己再包一层
    __enter__: super().__enter__(); set_ulysses_sp_group(); module.train()
    __exit__:  set_ulysses_sp_group(prev); zero_grad_on_exit; super().__exit__()

class EngineEvalModeCtx(BaseEngineCtx):
    __enter__: super().__enter__(); set_ulysses_sp_group(); module.eval()
    __exit__:  set_ulysses_sp_group(prev); FSDP.reshard(); super().__exit__()
```

**关键设计**：**offload 逻辑写在基类**（BaseEngineCtx），**模块 train/eval 切换和 SP group 切换写在子类**——一份代码同时管住"参数搬迁 + 模块状态 + SP context"三件事。

### 17.3.4 EngineRegistry：4 维派发的策略池

```
_engines = {
    model_type: {
        backend: {
            (device, vendor): EngineClass  # 或 device 单键
        }
    }
}

get_engine_cls(model_type, backend):
    device = os.getenv("VERL_ENGINE_DEVICE") or get_device_name()
    vendor = os.getenv("VERL_ENGINE_VENDOR") or get_vendor()
    # 查找顺序: (device, vendor) → device → (cuda, nvidia) fallback
```

**注册实例**（扫描所有 `@EngineRegistry.register` 得）：

| model_type | backend | device | 实现类 |
|---|---|---|---|
| language_model | fsdp / fsdp2 | cuda / npu | `FSDPEngineWithLMHead` |
| value_model | fsdp / fsdp2 | cuda / npu | `FSDPEngineWithValueHead` |
| language_model | fsdp_turbo | cuda / npu | `FSDPTurboEngineWithLMHead` |
| language_model | megatron | cuda | `MegatronEngineWithLMHead` |
| language_model | megatron | **npu** | `MindspeedEngineWithLMHead`（NPU 特化）|
| value_model | megatron | cuda | `MegatronEngineWithValueHead` |
| value_model | megatron | **npu** | `MindspeedEngineWithValueHead` |
| language_model | veomni | cuda / npu | `VeOmniEngineWithLMHead` |
| value_model | veomni | cuda / npu | `VeOmniEngineWithValueHead` |
| language_model | torchtitan | cuda / npu | `TorchTitanEngineWithLMHead` |
| language_model | automodel | cuda | `AutomodelEngineWithLMHead` |

**巧妙之处**：Megatron 和 Mindspeed **共用 backend 名 "megatron"**，只用 device 区分（cuda→Megatron，npu→Mindspeed）——配置层写 `backend=megatron` 即可跨硬件运行。

### 17.3.5 FSDPEngine：初始化流程详解（参考实现）

```
__init__: 保存 config + _init_device_mesh + full_determinism 标志

initialize (transformer_impl.py:188):
  ┌── _build_model_optimizer:
  │   ├── _build_module (:235)
  │   │   ├── init_context(use_meta_tensor)          # 大模型省内存
  │   │   ├── AutoModel.from_pretrained(...)
  │   │   ├── strip _verl_strip_modules              # 去无用子模块
  │   │   ├── apply liger_kernel (可选)              # 融合算子
  │   │   ├── apply_monkey_patch (⭐关键)             # remove_padding + Ulysses SP
  │   │   ├── module.to(torch_dtype)
  │   │   └── gradient_checkpointing_enable
  │   ├── [可选] _build_lora_module                   # PEFT get_peft_model
  │   ├── _build_fsdp_module (:366)
  │   │   ├── strategy == "fsdp":  FSDP(module, ...)  # FSDP1 包装
  │   │   └── strategy == "fsdp2": apply_fsdp2(...)   # FSDP2 原地
  │   ├── _build_optimizer                           # AdamW / etc.
  │   └── _build_lr_scheduler                        # warmup+cosine
  ├── FSDPCheckpointManager 挂载
  └── to("cpu") 首次 offload
```

**为什么 `apply_monkey_patch` 是关键？**
它把 HF transformer forward 换成 verl 定制版本：
1. **remove_padding**：packed varlen forward，节省 40-60% flops
2. **Ulysses SP**：注入 all_to_all_single 实现序列并行
3. **fused kernels**：融合 linear + cross_entropy

### 17.3.6 forward_backward_batch：核心训练循环

```
forward_backward_batch(data, loss_fn, forward_only=False):
  1. batch_num_tokens = data["loss_mask"].sum(); all_reduce (DP)     # loss 归一化用
  2. micro_batches, indices = prepare_micro_batches(data, ...)       # ⭐ 动态 bsz 切分
  3. for i, micro_batch in enumerate(micro_batches):
       is_last = (i == len(micro_batches)-1)
       with _gradient_sync_context(is_last_micro_batch=is_last):     # ⭐ 关键: 非末批不同步梯度
         loss, meta = self.forward_step(micro_batch, loss_fn, forward_only)
         if not forward_only:
           (scaler.scale(loss) if scaler else loss).backward()
  4. postprocess_batch_func(output_lst, indices, data)                # ⭐ 恢复动态 batch 顺序
```

**两个性能优化**：
- **`_gradient_sync_context`**（:673）：梯度累积时只在最后一 micro-batch 触发 all_reduce（FSDP1 用 `no_sync()`，FSDP2 用 `set_requires_gradient_sync(False)`），把 N 次 reduce-scatter 降到 1 次。
- **动态 batch**：`rearrange_micro_batches` 按 seqlen² 均衡各 rank 工作量，`indices` 记录顺序以便 `postprocess_batch_func` 还原（详见 [05-性能优化设计](05-性能优化设计.md)）。

### 17.3.7 权重同步三件套（rollout weight sync 的核心）

```
        ┌────────────────────────┬───────────────────────┬─────────────────────────┐
        │ get_per_tensor_param() │ get_per_tensor_param_ │ get_per_tensor_param_   │
        │                        │       shard()         │       delta_shard()      │
        ├────────────────────────┼───────────────────────┼─────────────────────────┤
时机    │ seed / first sync      │ 内存受限的 seed       │ steady step             │
传输量  │ 100% (full tensor)     │ 100% (分片全 gather)  │ ~0.1%-5% (只发变化元素)  │
使用    │ naive / nccl backend   │ delta_sharded seed    │ delta_sharded steady    │
数据格式│ (name, full_tensor)    │ (name, local, spec)   │ (slots, idx, val, ...)  │
        └────────────────────────┴───────────────────────┴─────────────────────────┘
```

**delta 流水线**（`utils.py::hf_delta_export`）：

```
for name, local, spec in generator:
    snap = snapshots[name]                                # pinned CPU 快照
    if contributes:
        base = snap.to(gpu)
        lidx, lval = shard_delta_indices(local, base, 0)  # 稀疏 diff (只留变化的下标+值)
    else:
        lidx, lval = empty (但保持迭代节奏, 维持集合通信 lockstep)
    snap.copy_(local)                                     # 更新快照
    yield entry_fn(name, spec, place, lidx, lval), pg     # 后端注入 entry_fn (identity or EP)
```

**关键抽象 `ShardSpec`（`spec.py:59`）**：把"张量在集群里怎么分片"用 `mesh + placements + place + gather_group` 声明式描述出来，让 delta pipeline **完全后端无关**。

### 17.3.8 Delta 快照的选型：`delta_pin_snapshots`

```python
# BaseEngine.delta_pin_snapshots = True   (FSDP/veomni 默认)
# MegatronEngine.delta_pin_snapshots = False  (⭐ 覆盖!)
```

**为什么 Megatron 覆盖成 False？**
Megatron 自己就大量使用 pinned host buffer，如果再叠加一份 pinned 快照，会把节点的 cudaHostAlloc 池耗尽，surfaces 为无关处的 CUDA OOM（Bytedance 在 30B/235B 训练里踩过坑）。pageable 只是 H2D 慢一点，不会 OOM。

## 17.4 rollout/ 子模块（生成引擎抽象）

### 17.4.1 目录结构与三层抽象

```
verl/workers/rollout/
├── __init__.py
├── base.py              ← ⭐ BaseRollout (async 接口: resume/release/update_weights)
├── replica.py           ← ⭐ RolloutReplica (三种部署模式: hybrid/colocated/standalone)
├── llm_server.py        ← ⭐ LLMServerManager + LLMServerClient + GlobalRequestLoadBalancer
├── schemas.py           ←   请求/响应数据结构
├── tokenizer.py         ←   tokenizer 封装
├── utils.py             ←   通用工具
├── hf_rollout.py        ←   HF native rollout (调试用)
├── naive/               ←   naive rollout
├── vllm_rollout/        ← ⭐ vLLM ServerAdapter + async_server + bucketed_weight_transfer
├── sglang_rollout/      ← ⭐ SGLang ServerAdapter + async_server + delta_loader
└── trtllm_rollout/      ←   TensorRT-LLM ServerAdapter
```

**三层抽象**：

```
┌─────────────────────────────────────────────────────────────┐
│  BaseRollout (base.py:29) - 抽象接口                        │
│  ├── resume(tags: list[str])          # 恢复 weights/kv_cache│
│  ├── release()                        # 释放 GPU 内存        │
│  ├── update_weights(weights, wire_format)  # 权重同步        │
│  └── generate_sequences(prompts)      # 生成                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          v (get_rollout_class 派发)
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    vllm.ServerAdapter  sglang.ServerAdapter  trtllm.ServerAdapter
        │                                     │
        v                                     v
    RolloutReplica (replica.py:70)  - 部署单位
    ├── init_hybrid(worker_group)        # 与 trainer 同进程
    ├── init_hybrid_colocated(...)       # 同 PG 不同进程
    ├── init_colocated(resource_pool)    # colocated separate proc
    └── init_standalone()                # 独立资源池
        │
        v
    LLMServerManager (llm_server.py:464) - 多 replica 集中管理
    ├── _initialize_llm_servers()        # 拉起所有 replica
    ├── _init_global_load_balancer()     # 全局负载均衡
    └── get_client(client_cls) → LLMServerClient
```

### 17.4.2 BaseRollout 抽象：4 个 async 接口

`BaseRollout` 只强制要求 4 个方法（`base.py:29`）：

```
resume(tags)          → 恢复 GPU 内存（weights / kv_cache 分开控制）
update_weights(weights, wire_format)
                      → 从 trainer 侧接收权重
                      → wire_format ∈ {"named_tensors", "delta_flush"}
release()             → 释放全部 GPU 内存
generate_sequences(prompts)  → 生成（sync 模式）
```

**`wire_format` 的意义**：
- `"named_tensors"`：默认，(name, full_tensor) 生成器
- `"delta_flush"`：sglang 专用，接收 delta engine 的 sparse payload

派发注册表（`base.py:88`）：
```python
_ROLLOUT_REGISTRY = {
    ("vllm",   "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}
```

`get_rollout_class(name, mode)` 用 `importlib.import_module` 按需加载——**没装 sglang 也能跑 vllm**。

### 17.4.3 RolloutReplica：4 种部署模式（`replica.py`）

**核心概念**：一个 RolloutReplica = 一个"逻辑推理集群"（`tp × dp × pp`），对应命令行 `vllm serve --dp-size ... --tp-size ...` 或 `sglang.launch_server`。

```
RolloutReplica.__init__ 计算拓扑:
  world_size = tp * dp * pp
  gpus_per_replica_node = min(gpus_per_node, world_size)
  nnodes = world_size / gpus_per_replica_node   # 每个 replica 占的节点数
```

**4 种初始化模式**（`RolloutMode` 枚举）：

| 模式 | 谁调用 | 与训练关系 | 用于 |
|---|---|---|---|
| `init_hybrid(worker_group)` | Sync trainer | **同进程**，与 trainer engine 共用参数（`self.workers = worker_group.workers[...]`） | on-policy 同步训练 |
| `init_hybrid_colocated(wg, rp)` | Sync trainer | 同 PG 但**不同进程**，共享 GPU 靠 sleep/wake | 折中方案 |
| `init_colocated(resource_pool)` | Reward / Teacher | 复用已有 resource_pool，**独立进程** | GRM (LLM-as-Judge) / teacher policy |
| `init_standalone()` | Separate async | **独立创建 resource_pool** | off-policy 分离训练 |

对应到 V1 三种 trainer_mode（详见 [13-异步训练方案对比](13-异步训练方案对比.md)）：

```
sync           →  init_hybrid          (colocate + no partial rollout)
colocate_async →  init_hybrid_colocated (colocate + partial rollout, kimi-1.5 style)
separate_async →  init_standalone      (standalone + partial rollout, AReaL style)
```

### 17.4.4 RolloutReplica 生命周期：sleep/wake/abort

```
wake_up()              → 各 server 恢复 weights + kv_cache
sleep()                → 释放 GPU 内存
abort_all_requests()   → partial rollout: 中断并保存未完成请求
resume_generation()    → abort 之后重新调度
clear_kv_cache()       → reset kv cache
release_kv_cache()     → 只释放 kv cache 内存 (keep weights)
resume_kv_cache()      → 权重同步完成后恢复 kv cache
```

**权重同步与 kv_cache 的解耦**：`release_kv_cache / resume_kv_cache` 允许"权重同步过程中只暂时腾出 kv 空间"，不动 weights，节省 sync 时间。这是 partial rollout 稳态运行的关键。

### 17.4.5 LLMServerManager：多 Replica 编排

```
LLMServerManager (llm_server.py:464):
  ├── _initialize_llm_servers(start_rank):
  │     for i in range(num_replicas):
  │         replica = RolloutReplica(...)
  │         if standalone:  await replica.init_standalone()
  │         else:           await replica.init_hybrid(...)
  │         # 收集 server_address / server_handle
  │
  ├── _init_global_load_balancer():
  │     GlobalRequestLoadBalancer.acquire_server(request_id)
  │     基于 sticky routing + inflight count 均衡
  │
  └── get_client(client_cls) → LLMServerClient
        │
        ├── LLMServerClient:            用于 sync agent loop
        │     async generate() → ChatCompletion 结果
        │
        └── FullyAsyncLLMServerClient:   用于 async agent loop
              acquire_server 支持 partial rollout resume token
              超长 response 会 abort 并把中间态存回 TransferQueue
```

**GlobalRequestLoadBalancer 的 sticky routing**（`llm_server.py:69`）：
- 一次 agent 交互可能有多轮，用 `request_id` 做 sticky routing 保证多轮全在同一 server（KV cache 命中）
- 后端 in-flight 计数动态负载均衡
- `release_server` 显式解引用（避免 sticky cache 泄漏）

### 17.4.6 vLLM ServerAdapter：BaseRollout 实现（`vllm_rollout.py`）

```
ServerAdapter(BaseRollout):
  __init__: 记住 rollout_config + model_config + device_mesh, 不启动服务
  
  _ensure_server_handle() → 懒加载获取 server_handle
  
  async resume(tags):     → server.wake_up(tags)
  async release():        → server.sleep()
  async update_weights(weights, wire_format):
      1. 分块 (bucketed_weight_transfer.py) 减少 RPC 开销
      2. server.update_weights_from_tensor(bucket, wire_format)
  
  async _execute_method(method, *args, **kwargs):
      → server_handle 上执行任意方法（用于扩展）
  
  generate_sequences(prompts) → 兼容 sync 接口
```

**`bucketed_weight_transfer.py`**：把细粒度的权重张量按 MB 大小打包成"桶"，一次 RPC 传一个桶，避免每张量都跨进程 IPC（否则 100B 模型有几万个参数，RPC 开销爆炸）。

## 17.5 reward_manager/ 子模块

### 17.5.1 抽象接口

`AbstractRewardManager`（`abstract.py:27`）契约极简：

```
class AbstractRewardManager(ABC):
    def __init__(self, tokenizer, num_examine, compute_score, 
                 reward_fn_key="data_source", **kwargs): ...
    def __call__(self, data: DataProto, return_dict=False) -> Tensor | dict: ...
    def _extract_reward_from_rm_scores(self, data, return_dict) -> ... | None
```

**通用协议**：
- 输入：`DataProto`（含 `prompts / responses / attention_mask / non_tensor_batch.reward_model.ground_truth / data_source`）
- 输出：`reward_tensor: (bsz, response_length)` **只在有效 response 最后一个 token 位置写分数**
- `return_dict=True` 时返回 `{"reward_tensor": ..., "reward_extra_info": ...}`

### 17.5.2 注册表机制（`registry.py`）

```python
REWARD_MANAGER_REGISTRY = {}

@register("naive")
class NaiveRewardManager(AbstractRewardManager): ...

@register("batch")
class BatchRewardManager(AbstractRewardManager): ...

@register("dapo")
class DAPORewardManager(AbstractRewardManager): ...

@register("prime")
class PrimeRewardManager(AbstractRewardManager): ...
```

外部代码通过 `get_reward_manager_cls(name)` 按配置字符串取类。**与 `EngineRegistry` 完全同构的策略池模式**。

### 17.5.3 四种 RewardManager 对比

| 名称 | 计算方式 | 典型场景 | 特色 |
|---|---|---|---|
| `naive` | **逐样本串行**在 driver 端调 `compute_score(data_source, response, ground_truth)` | 规则式 reward（如 gsm8k / math） | ⭐ **SIGALRM 超时保护**（`_score_timeout`）：单样本 regex 灾难性回溯不会卡死训练 |
| `batch` | 整批一次性调 `verify(data)` | 批量向量化打分（正则匹配、判定器） | 一次 tokenize + 一次向量化处理，减少 python 调用开销 |
| `dapo` | DAPO 论文的算法逻辑 | DAPO 训练（组内 std=0 过滤） | 与 `algorithm.filter_groups.metric` 配合工作 |
| `prime` | PRIME 论文的 process reward | PRIME 训练 | 支持 process reward model |

### 17.5.4 SIGALRM 超时保护（`naive.py::_score_timeout`）

```python
@contextmanager
def _score_timeout(seconds):
    if not seconds or seconds <= 0:
        yield; return
    def _handler(signum, frame):
        raise TimeoutError(f"compute_score timed out after {seconds}s")
    try:
        old = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # 非主线程无法用 SIGALRM，降级到无 timeout
        yield; return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
```

**为什么需要**：一次 pathological 的 regex（如深度嵌套的 `.*` 灾难性回溯）能让 `compute_score` 挂几分钟，而 `NaiveRewardManager` 是**在 driver 主线程串行打分**——一个卡死等于整个训练卡死。SIGALRM 保证单样本超时 → 该样本 reward=0，继续训练。

### 17.5.5 与 V1 数据流的关系

在 V1 trainer 中（详见 [01-核心运行流程](01-核心运行流程.md)），reward 有两条路径：

```
路径 A: colocate reward loop（每步 Trainer._compute_reward_colocate）
    ├── Trainer._step_once (:_step_once)
    │   └── if reward_loop_manager.reward_loop_worker_handles is None:
    │       reward = compute_reward_colocate(batch)   # 用 RewardManager 直接算
    └── RewardManager 在 Trainer 进程内运行

路径 B: standalone reward worker（RewardLoopManager + reward pool）
    ├── AgentLoopWorker.postprocess:
    │   └── _compute_score(outputs)                    # 在 AgentLoopWorker 里调
    └── 结果直接写回 TransferQueue.extra_fields.reward_extra_info
```

**两条路径均使用同一个 `RewardManager` 抽象**，只是执行位置不同。DAPO 强制走路径 B（因为要在采样时就有 reward 用于 filter_groups）。

## 17.6 workers/utils/ 子模块（训练算子）

### 17.6.1 losses.py：三大损失函数

```
sft_loss(config, model_output, data, dp_group)
    → 监督微调（NLL over loss_mask）
    → left-shift loss_mask 1 位对齐 log_prob 位置

ppo_loss(config, model_output, data, dp_group)  ← ⭐ PPO 训练的核心
    1. no_padding_2_padding(log_probs)         # packed → padded
    2. 组装 global_batch_info (dp_size, batch_num_tokens, global_batch_size)
    3. select fields → to_padded_tensor:
       {response_mask, old_log_probs, advantages, [rollout_is_weights], [ref_log_prob]}
    4. policy_loss_fn = get_policy_loss_fn(loss_mode)   # vanilla / clip-higher / dual-clip / ...
    5. pg_loss, pg_metrics = policy_loss_fn(...)
    6. + entropy_loss * entropy_coeff
    7. + kl_loss * kl_loss_coef  (if use_kl_loss)
    8. return policy_loss, metrics

value_loss(config, model_output, data, dp_group)  ← ⭐ Critic 训练
    → compute_value_loss (clip range = cliprange_value)
    → 支持 loss_scale_factor / global_batch_size 全局归一化
```

**关键归一化设计**（`ppo_loss` 里）：

```
if dp_size>1 or batch_num_tokens is not None or global_batch_size is not None:
    metric_aggregation = AggregationType.SUM   # loss_fn 内部已 /global，metrics 侧再累加
else:
    metric_aggregation = AggregationType.MEAN
```

**为什么这样设计**：`policy_loss_fn` 在有全局信息时内部用 `dp_size / batch_num_tokens / global_batch_size` 做归一化，返回的每个 micro-batch 的 loss **已经是全局均值的一部分**。metrics 侧再 SUM 就得到全局均值。这保证 "grad 结果 与 micro-batch 切分方式无关"（否则 minibatch/microbatch 数量变了梯度就变了）。

### 17.6.2 padding.py：nested tensor ↔ padded tensor 互转

verl 内部数据流是 **packed varlen (`torch.nested.jagged`)**（零填充），但计算 loss 时又需要**规整的 padded tensor**（对齐 mask）。`padding.py` 提供 6 个工具：

```
left_right_2_no_padding(data)      → padded tensor 转 nested
no_padding_2_padding(tensor, data) → nested 转 padded (from response_mask)
embeds_padding_2_no_padding(data)  → embed 版本
response_from_nested(tensor, mask) → 从 nested 取 response 部分
response_to_nested(tensor, mask)   → 把 response 部分打包回 nested
build_attention_mask_from_nested(input_ids, max_seq_len)
```

**上下文**：Engine 内部走 packed 路线（详见 17.3），loss 计算走 padded 路线，两条路的边界就在 `no_padding_2_padding`。

## 17.7 数据流：完整训练一步中各模块如何协作

```
Driver (PPOTrainer._step_once)
  │
  │ (1) 采样  self.replay_buffer.sample(global_steps=..., partition_id="train")
  │      └─→ 阻塞直到本 step 全部 prompt 完成，返回 KVBatchMeta(keys, tags)
  │
  │ (2) reward (path A colocate)
  │      └─→ Trainer 侧 reward_loop_manager 里的 RewardManager 打分
  │          → tq.kv_batch_put(reward)
  │
  │ (3) 分发到 actor DP
  │      Driver call: self.actor_rollout_wg.compute_log_prob(batch)   # KVBatchMeta
  │        │
  │        │ @register 装饰的 Functor 代理:
  │        │   dispatch: mesh_name="actor" 按 dp_rank 切 keys 集
  │        │   execute:  各 Ray Actor 并发 .remote()
  │        v
  │      ┌── ActorRolloutRefWorker.compute_log_prob (each rank) ────┐
  │      │  self.actor.infer_batch(data)                             │
  │      │    │                                                       │
  │      │    │ TrainingWorker.infer_batch:                           │
  │      │    │   1. maybe_fix_3d_position_ids                        │
  │      │    │   2. engine.eval_mode() 上下文 (load model to GPU)    │
  │      │    │   3. engine.infer_batch(data, loss_fn=None):          │
  │      │    │       - tq.kv_batch_get(fields) 在 worker 内取张量    │
  │      │    │       - forward-only micro-batch                       │
  │      │    │       - tq.kv_batch_put(log_probs)                     │
  │      │    │   4. _postprocess_output                              │
  │      │    │   5. offload back to CPU                              │
  │      │    v                                                        │
  │      │  return TensorDict (只有 metrics)                          │
  │      └───────────────────────────────────────────────────────────┘
  │        │
  │        v collect: concat metrics
  │      Driver 拿到聚合后的 KVBatchMeta + metrics
  │
  │ (4) update_actor (类似 (3))
  │      ├── TrainingWorker.train_mini_batch:
  │      │     for epoch in ppo_epochs:
  │      │       for mini_batch in split(data, ppo_mini_batch_size):
  │      │         engine.train_mode() + engine.train_batch(mini_batch, ppo_loss)
  │      │           - engine.forward_backward_batch (micro-batch)
  │      │           - engine.optimizer_step (grad clip + skip inf/nan)
  │      └── 返回 grad_norm / actor/pg_loss / kl_loss / entropy_loss
  │
  │ (5) update_weights (colocate naive 模式)
  │      Driver call: self.actor_rollout_wg.update_weights(global_steps)
  │        │
  │        │ 每个 rank 独立执行:
  │        v
  │      ActorRolloutRefWorker.update_weights:
  │        1. self.rollout.resume(tags=["weights"])           # 恢复 vLLM 权重内存
  │        2. per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
  │        3. await self.rollout.update_weights(per_tensor_param)
  │             └─→ vllm ServerAdapter.update_weights:
  │                   bucketed_weight_transfer.py 分桶传递
  │                   → server.update_weights_from_tensor(bucket)
```

**关键观察**：
- 张量**从不经过 Driver**：Driver 只传 `KVBatchMeta`（keys+tags），Worker 内部 `tq.kv_batch_get` / `kv_batch_put` 直接与 TransferQueue 交互——这就是 [01-核心运行流程](01-核心运行流程.md) 中说的"零拷贝数据面"的技术实现。
- **每层抽象都负责自己的领域**：engine_workers 负责 Ray + dispatch，engine 负责训练循环 + 梯度，rollout 负责生成 + 权重同步，reward 负责打分。层次清晰，扩展点明确。

## 17.8 关键设计模式与扩展点

### 17.8.1 一以贯之的注册表模式

| 注册表 | 位置 | 装饰器 | 派发键 |
|---|---|---|---|
| `EngineRegistry` | `engine/base.py:339` | `@EngineRegistry.register` | (model_type, backend, device, vendor) |
| `REWARD_MANAGER_REGISTRY` | `reward_manager/registry.py:22` | `@register("name")` | name string |
| `_ROLLOUT_REGISTRY` | `rollout/base.py:88` | dict 直接注册（无装饰器）| (rollout_name, mode) |
| `RolloutReplicaRegistry` | `rollout/replica.py:302` | `RolloutReplicaRegistry.register` | name string |
| `CheckpointEngineRegistry` | `verl/checkpoint_engine/base.py` | 同型 | backend string |

**统一价值**：**所有横向可替换的组件都通过注册表实现"配置即路由"**——用户改一行 YAML（`backend: fsdp2` / `reward_manager.name: dapo` / `rollout.name: vllm`）就能切换实现，无需改代码。

### 17.8.2 上下文管理器模式统一 GPU 资源生命周期

```
BaseEngineCtx (train/eval): 自动 load model+opt → to GPU → 退出时 offload
RolloutReplica.wake_up/sleep: 手动控制，配合 vLLM CuMemAllocator
train_mode() 的 kwargs disable_auto_offload=True: 关闭自动 offload（Sync 训练里 trainer/rollout 需要精细互斥）
```

### 17.8.3 三大扩展点

**扩展点 1：新增训练后端**
1. 继承 `BaseEngine`，实现 `initialize/forward_backward_batch/get_per_tensor_param/optimizer_step/save_checkpoint`
2. 用 `@EngineRegistry.register(model_type="language_model", backend="mybackend", device=["cuda"])` 装饰
3. YAML 配 `actor_rollout_ref.actor.strategy: mybackend`

**扩展点 2：新增 rollout 引擎**
1. 继承 `BaseRollout`，实现 `resume/release/update_weights`（都是 async）
2. 继承 `RolloutReplica`，实现 `launch_servers`（拉起 http server）
3. 在 `rollout/base.py::_ROLLOUT_REGISTRY` 加一行映射
4. YAML 配 `actor_rollout_ref.rollout.name: myrollout`

**扩展点 3：新增 RewardManager**
1. 继承 `AbstractRewardManager`，实现 `__call__(self, data, return_dict)`
2. 用 `@register("myreward")` 装饰
3. YAML 配 `reward.reward_manager.name: myreward`

**扩展点 4：自定义 loss**
1. 在 `verl/trainer/ppo/core_algos.py` 用 `@register_policy_loss_fn("mymode")` 装饰新 loss
2. YAML 配 `actor_rollout_ref.actor.policy_loss.loss_mode: mymode`

（详见 [06-自定义扩展方式](06-自定义扩展方式.md)）

## 17.9 常见陷阱与最佳实践

| 陷阱 | 描述 | 规避方式 |
|---|---|---|
| `Dispatch.ONE_TO_ALL` 用错场景 | 用它包裹数据处理方法 → 每 rank 拿到全量数据 → 内存爆炸 | 数据类方法用 `make_nd_compute_dataproto_dispatch_fn(mesh_name=...)` |
| `update_weights` 忘 `resume` | rollout 睡眠时释放了 weight 内存，直接 update 会 CUDA error | `ActorRolloutRefWorker.update_weights` 已内置：先 `rollout.resume(["weights"])` |
| Megatron 里 `delta_pin_snapshots=True` | Megatron 本身大量 pinned → 再叠 pinned 快照 → cudaHostAlloc OOM | Megatron 类里已覆盖为 False |
| NaiveRewardManager 单卡卡死 | 打分函数 regex 灾难性回溯 → 训练整卡 hang | 用 `compute_score_timeout` 开 SIGALRM 保护 |
| 忘 `aggressive_empty_cache` | colocate 模式下 vLLM `cudaMemGetInfo` 看不到剩余显存 → wake_up 失败 | `init_model` 最后已内置 `aggressive_empty_cache(force_sync=True)` |
| 直接改 engine 内部状态 | 破坏 dispatch 的 mesh 假设 | 一律通过 `set_dispatch_collect(mesh_name=..., ...)` 声明 |
| dispatch 时 padding 忘补齐 | `dispatch_dp_compute_data_proto` 已自动 padding，但自定义 dispatch 得手动 | 复用现有 dispatch 或看 `decorator.py:167` 参考实现 |

## 17.10 一图总结：workers 模块的调用栈

```
┌──── Driver (PPOTrainer._step_once) ─────────────────────────────────┐
│                                                                       │
│  replay_buffer.sample() → KVBatchMeta(keys, tags)                    │
│         │                                                             │
│         v                                                             │
│  self.actor_rollout_wg.compute_log_prob(batch)                        │
│         │  (RayWorkerGroup 代理 Functor)                              │
│         v                                                             │
│  ┌──── Ray Actor (ActorRolloutRefWorker) ────────────────────────┐    │
│  │                                                                │    │
│  │  compute_log_prob (@register mesh_name="actor")                │    │
│  │         │                                                       │    │
│  │         v                                                       │    │
│  │  self.actor.infer_batch(data)   # 内部 TrainingWorker 对象      │    │
│  │         │                                                       │    │
│  │         v                                                       │    │
│  │  self.engine.infer_batch(data)  # BaseEngine 抽象接口           │    │
│  │         │                                                       │    │
│  │         v                                                       │    │
│  │  ┌── FSDPEngineWithLMHead (具体实现) ──────────────────────┐   │    │
│  │  │                                                          │   │    │
│  │  │  with eval_mode(): (BaseEngineCtx: 自动 load to GPU)     │   │    │
│  │  │    forward_backward_batch(forward_only=True):           │   │    │
│  │  │      1. prepare_micro_batches (workers/engine/utils.py) │   │    │
│  │  │      2. for micro_batch:                                 │   │    │
│  │  │           forward_step(micro_batch, loss_fn=None):      │   │    │
│  │  │             prepare_model_inputs (SP pad/slice)          │   │    │
│  │  │             self.module.forward (HF+monkey_patch)        │   │    │
│  │  │             prepare_model_outputs (logprobs_from_logits) │   │    │
│  │  │      3. postprocess_batch_func                          │   │    │
│  │  │    退出 eval_mode → offload back to CPU                  │   │    │
│  │  └─────────────────────────────────────────────────────────┘   │    │
│  │         │                                                       │    │
│  │         v                                                       │    │
│  │  _postprocess_output: all_reduce loss + allgather metrics      │    │
│  │         │                                                       │    │
│  │         v                                                       │    │
│  │  return TensorDict(metrics=...)                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│         │                                                             │
│         v (collect: concat)                                           │
│  Driver 拿到聚合后的 KVBatchMeta + metrics                            │
└──────────────────────────────────────────────────────────────────────┘

数据流关键：张量从不经过 Driver，全部通过 TransferQueue 在 Worker 内部读写。
```

---

**至此**，`verl/workers/` 五大模块的设计与实现全景已完整覆盖。深入细节继续参考：
- [15-训练后端与优化器配置](15-训练后端与优化器配置.md) - engine 配置层
- [16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md) - rollout 显存管理
- [12-decorator与dispatch机制详解](12-decorator与dispatch机制详解.md) - @register 装饰器细节
- [13-异步训练方案对比](13-异步训练方案对比.md) - Replica 三种部署模式的应用

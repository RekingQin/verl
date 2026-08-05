# 10 · Ray 在 verl 中的串联作用（从进程编排到一次远程调用）

> 返回索引：[README.md](README.md)
> 关联阅读：[01-核心运行流程.md](01-核心运行流程.md)（dispatch 概览）、[11-算法配置与跨Worker数据流转.md](11-算法配置与跨Worker数据流转.md)
> 源码基线：`single_controller/{base,ray}/`、`trainer/main_ppo.py`（V1，`TaskRunnerV1`）/ `trainer/main_ppo_v0.py`（legacy，`TaskRunner`）、`workers/engine_workers.py`
> 版本提示：`main_ppo.py` 现为 V1 入口（`TaskRunnerV1`，`run_ppo @ :34`、`TaskRunnerV1 @ :104`）；v0 的 `TaskRunner`/`RayPPOTrainer` 已迁至 `main_ppo_v0.py`。V1 的"建组"逻辑（原 `ray_trainer.init_workers`）在 `trainer/ppo/v1/trainer_base.py::PPOTrainer.init`（:217），下文 §10.8 以 v0 链路为主便于对照。

---

## 10.0 一句话定位

> **Ray 负责"进程/资源/对象"的编排与传递；torch.distributed(NCCL) 负责"GPU 间集合通信"。两者分工，互不替代。**

verl 用 Ray 把"一个 Driver 主进程"和"几十上百个 GPU Worker 进程"串成一个逻辑整体，让你能在 Driver 里写 `actor_rollout_wg.compute_log_prob(batch)` 这样的"单机式调用"，背后自动完成"切数据 → 派到每张卡 → 并发执行 → 收结果"。

Ray 在 verl 里干 4 件事：
1. **拉起集群与进程**（`ray.init` + `@ray.remote` actor）
2. **按资源池分配 GPU**（PlacementGroup / bundle）
3. **把 Worker 类实例化成分布在各卡上的 Ray actor**，并注入 `WORLD_SIZE/RANK/MASTER_ADDR` 等环境变量
4. **把 Driver 的方法调用变成对每个 actor 的 `.remote()` 远程调用**，再 `ray.get()` 收集

而真正的"多卡梯度同步、tensor 切分通信"用的是 **NCCL（torch.distributed）**，Ray 不参与——Ray 只是帮各 Worker 凑齐建组所需的 `MASTER_ADDR/PORT/RANK`。

---

## 10.1 三层进程拓扑

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │ Ray Cluster                                                          │
 │                                                                      │
 │  [1] Driver 进程: python main_ppo.py                                 │
 │        run_ppo(config):                                              │
 │          ray.init(...)                  <- 拉起/连接集群              │
 │          runner = TaskRunnerV1.remote() <- [2]（v0 为 TaskRunner）     │
 │          ray.get(runner.run.remote(config))                          │
 │                                                                      │
 │  [2] TaskRunnerV1 (@ray.remote, num_cpus=1)  "控制器 actor"          │
 │        - 解析 Hydra config                                           │
 │        - 建 role_worker_mapping / ResourcePoolManager                │
 │        - PPOTrainer.init() / fit()  （V1：trainer/ppo/v1/）           │
 │                                                                      │
 │  [3] Worker actors (每张 GPU 一个 Ray actor)  "重计算"               │
 │        ActorRolloutRefWorker / TrainingWorker(critic) ...            │
 │        每个 actor 内部跑 FSDP/Megatron + vLLM，                       │
 │        彼此之间用 NCCL 组通信（不经 Ray）                              │
 └─────────────────────────────────────────────────────────────────────┘
```

为什么 Driver 逻辑要再包一层 `TaskRunnerV1` actor（`main_ppo.py:104`；v0 见 `main_ppo_v0.py:30`）？注释明说：`please make sure main_task is not scheduled on head`——把重编排逻辑放进一个 `num_cpus=1` 的 actor，避免占用 Ray head 节点资源，也让 Driver 本身可被调度/隔离。

---

## 10.2 第①步：起集群（`run_ppo`，main_ppo.py:34）

```python
if not ray.is_initialized():
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray.init(**OmegaConf.to_container(ray_init_kwargs))   # 关键: 统一 runtime_env(环境变量)

task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
runner = task_runner_class.remote()
ray.get(runner.run.remote(config))      # 阻塞直到训练结束
```

要点：
- `runtime_env` 把 `TOKENIZERS_PARALLELISM / NCCL_DEBUG / VLLM_LOGGING_LEVEL / TRANSFER_QUEUE_ENABLE` 等环境变量统一下发给所有 actor（`get_ppo_ray_runtime_env`）。
- `ray.get(runner.run.remote(config))` 是整个程序的"主阻塞点"：Driver 把 config 这个对象序列化进 Ray object store，TaskRunner actor 拿到后开跑。

---

## 10.3 第②步：资源池 → PlacementGroup（`ResourcePoolManager`，ray/base.py:185）

Ray 用 **PlacementGroup（PG）** 预占 GPU。一个 PG 由若干 **bundle** 组成，每个 bundle = 一份资源（如 `{"CPU": k, "GPU": 1}`）。

```
config.trainer.nnodes=2, n_gpus_per_node=8
        │
        v  init_resource_pool_mgr（v0: main_ppo_v0.py:67；V1: trainer_base.py::PPOTrainer._init_resource_pool_mgr）
resource_pool_spec = {"global_pool": [8, 8]}   # 每个元素=一个节点的进程数
        │
        v  ResourcePoolManager.create_resource_pool
RayResourcePool(process_on_nodes=[8,8])
        │
        v  get_placement_groups (ray/base.py:131)
pg_scheme = [[bundle]*8, [bundle]*8]   # 2 个 PG, 各 8 个 bundle
placement_group(bundles, strategy="STRICT_PACK")  # STRICT_PACK=同一PG的bundle必须同节点
ray.get([pg.ready() for pg in pgs])    # 阻塞等待资源就位
sort_placement_group_by_node_ip(pgs)   # 按节点IP排序 -> 保证多次job间RANK稳定(断点续训关键)
```

关键设计：
- **`mapping: dict[Role -> pool_name]`**：哪个角色用哪个池。默认所有角色（actor/critic/ref/reward）共享 `global_pool`（**colocate** 同卡）；reward/teacher 可通过 `enable_resource_pool` 独立成 `reward_pool`/`teacher_pool`（`main_ppo.py:166/205`）。
- **`max_colocate_count`**：一个池里能塞几个 WorkerGroup。FSDP 用小值（actor/critic/ref 融合成一个 fused worker），Megatron 可用大值。
- **`num_gpus = 1 / max_colocate_count`**（`_create_worker`，ray/base.py:629）：每个 actor 只"声明"占 1/N 张卡，于是多个角色 actor 能 colocate 在同一张物理 GPU 上分时复用（配合 sleep/wake）。

---

## 10.4 第③步：把 Worker 类变成分布在各卡的 actor（`RayWorkerGroup`，ray/base.py:418）

`init_workers()` 为每个角色建一个 `RayWorkerGroup`。核心循环 `_init_with_resource_pool`（:538）：

```python
pgs = resource_pool.get_placement_groups(...)
rank = -1
local_world_size = resource_pool.store[0]            # 每节点进程数
for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
    if pg_idx == 0:
        self._get_master_addr_port(pg, ...)          # 选一个空闲端口当 MASTER
    for local_rank in range(local_world_size):
        rank += 1
        self._create_worker(rank, pg_idx, pg, local_rank, ...)
```

`_create_worker`（:623）做两件关键事：

**(a) 注入分布式环境变量**（Worker 进程靠它建 NCCL 组）：
```python
env_vars = {
    "WORLD_SIZE": str(world_size),   "RANK": str(rank),
    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
    "MASTER_ADDR": self._master_addr, "MASTER_PORT": self._master_port,
    "WG_BACKEND": "ray", ...
}
ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})
```

**(b) 用 PlacementGroupSchedulingStrategy 把 actor 钉到指定 bundle**（`RayClassWithInitArgs.__call__`，:369）：
```python
options = {"scheduling_strategy": PlacementGroupSchedulingStrategy(
              placement_group=pg, placement_group_bundle_index=local_rank)}
options.update(get_platform().ray_resource_options(num_gpus))   # 占 num_gpus=1/N 张卡
return self.cls.options(**options).remote(*args, **kwargs)       # 真正 .remote() 起 actor
```

于是：**每张 GPU 上有一个 Worker actor，它的 `RANK` 由 Ray 注入，进程内 `Worker.__init__`（worker.py:194）读 `os.environ["RANK"/"WORLD_SIZE"/"MASTER_ADDR"]` 建 `torch.distributed`**。Ray 把"凑齐建组信息"这件苦活做掉了。

### colocate / FusedWorker：多角色共享一个 actor
默认 actor/critic/ref 不是各起一组进程，而是 **FusedWorker**（`create_colocated_worker_cls_fused`，:1107）：把多个角色类塞进一个 `WorkerDict` actor，`spawn(prefix_set)` 再"分裂"出每个角色各自的 WorkerGroup 视图（方法名加前缀，`_rebind_actor_methods`，:730）。这样多个角色共享同一张卡与同一进程，靠 sleep/wake 切换显存占用——这是 verl 显存高效的根基。

---

## 10.5 第④步：一次"单机式调用"的完整链路（最重要）

这是把 Driver 的 `wg.compute_log_prob(batch)` 翻译成 Ray 远程调用的全过程，四个阶段：

### 声明期：`@register`（decorator.py:398）
Worker 方法被装饰，把 dispatch/execute/blocking 三属性写进方法的 `MAGIC_ATTR`：
```python
# engine_workers.py:697
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def compute_log_prob(self, data): ...
```

### 绑定期：`_bind_worker_method`（worker_group.py:185）
建 WorkerGroup 时扫描所有带 `MAGIC_ATTR` 的方法，用 `func_generator` 在 WorkerGroup 上动态生成同名代理方法，绑定对应的 dispatch_fn/collect_fn/execute_fn。

### 调用期：`Functor`（ray/base.py:49）
代理方法本质是个 Functor，`wg.compute_log_prob(batch)` 实际执行：
```python
def __call__(this, *args, **kwargs):
    args, kwargs = dispatch_fn(self, *args, **kwargs)   # ① 切分: 把整批切到各 rank
    padding_count = kwargs.pop(_padding_size_key, 0)
    output = execute_fn(method_name, *args, **kwargs)   # ② 并发: 向每个 actor .remote()
    if blocking: output = ray.get(output)               # ③ 收: 阻塞拿回 ObjectRef
    output = collect_fn(self, output)                   # ④ 聚合: concat 回整批
    if padding_count > 0: output = output[:-padding_count]  # ⑤ 去掉补齐样本
    return output
```

### 执行期：`execute_all_async`（ray/base.py:866）
`execute_fn` 解析为 `execute_all`，对每个 worker 发起远程调用：
```python
return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs)
        for worker in self._workers]      # 每个 -> getattr(actor, method).remote(...)
```
若 args 都是长度=world_size 的 list，则**按 rank 把第 i 份切片发给第 i 个 actor**（这正是 dispatch 切分后的形态）。

```
 Driver(WorkerGroup):  wg.compute_log_prob(batch)
        │
        │ dispatch_fn: batch -> [shard_0, shard_1, ..., shard_{N-1}]
        ▼
 ┌────────────┬────────────┬───────────────┐
 │ actor0     │ actor1     │ ...  actorN-1 │   每个 .remote(shard_i) 并发
 │ .remote()  │ .remote()  │               │
 └─────┬──────┴─────┬──────┴──────┬────────┘
       │ ObjectRef  │            │
       ▼            ▼            ▼
       ray.get([...])  收回各 rank 结果
        │
        │ collect_fn: concat([out_0, ..., out_{N-1}])
        ▼
   返回整批 output 给 Driver
```

### dispatch 模式如何选 rank（mesh-aware）
`make_nd_compute_dataproto_dispatch_fn("actor")`（decorator.py:300）是"惰性 N 维 dispatch"：它先向 WorkerGroup 查询每个 global rank 对应的 **DP rank**（`_query_dispatch_info`），只把数据切成 DP_size 份，再按 `dp_rank_mapping` 把同一份广播给同一 DP 组内的所有 TP/PP rank（`dispatch_nd_compute`，:202）。collect 时只从 `is_collect=True` 的 rank（每个 DP 组的 MP 源 rank）收集（`collect_nd_compute`，:236），避免 TP/PP 冗余重复。

这套信息从哪来？Worker 初始化时 `_register_dispatch_collect_info(mesh_name="train", dp_rank=..., is_collect=...)`（定义于 `single_controller/base/worker.py:86`，engine_workers.py:145 调用）把自己的 DP 拓扑登记进去；Driver 第一次调用该 mesh 的方法时惰性拉取并缓存（`dispatch_lazy_compute_data_proto`，decorator.py:266）。

---

## 10.6 Ray vs torch.distributed 的边界（关键澄清）

| 维度 | Ray | torch.distributed (NCCL) |
|------|-----|--------------------------|
| 角色 | 进程编排、资源调度、对象传递、RPC | GPU 间集合通信 |
| 粒度 | actor 级（一张卡一个 actor） | rank 级（FSDP shard / TP / PP / DP all-reduce） |
| 谁触发 | Driver 的 `.remote()` / `ray.get()` | Worker 内部 forward/backward 时自动触发 |
| 数据通道 | Ray object store（序列化，跨节点） | NCCL 显存直传（同/跨节点 GPU） |
| 建组所需 | 由 Ray 注入 `RANK/MASTER_ADDR/PORT` | 读上述 env 在 `Worker.__init__` 建组 |

一句话：**Driver↔Worker 走 Ray，Worker↔Worker（多卡梯度/张量）走 NCCL。** dispatch 切给各 rank 的数据经 Ray object store 下发；而一次 FSDP 前向里的参数 all-gather、反向里的梯度 reduce-scatter 全在 NCCL 里，Ray 完全无感。

---

## 10.7 串联全景一图流

```
 main_ppo.run_ppo
   ├─ ray.init                              # Ray: 起集群+统一runtime_env
   └─ TaskRunner.remote().run(config)       # Ray: 控制器actor
        ├─ ResourcePoolManager              # Ray: spec->PlacementGroup(bundle)预占GPU
        ├─ RayWorkerGroup(每角色)            # Ray: 每卡建actor, 注入RANK/MASTER_ADDR
        │     └─ Worker.__init__ 读env       # torch.dist: 建NCCL组
        └─ trainer.fit() 循环:
              wg.generate_sequences(batch)   # Ray RPC: dispatch->.remote->collect
              wg.compute_log_prob(batch)     #   ↑同上, Worker内NCCL前向
              wg.update_actor(batch)         #   ↑同上, Worker内NCCL前向+反向all-reduce
              wg.update_weights()            # Ray RPC: 触发权重->rollout同步(NCCL/CKPT引擎)
```

---

## 10.8 各 model 如何建 WorkerGroup、分资源、协调计算（v0：`init_workers`，ray_trainer.py:772；V1：`PPOTrainer.init`，trainer_base.py:217）

前面 §10.3/10.4 讲的是"通用机制"。这一节落到 verl 真实的 rollout/actor/critic/ref/reward 五类模型上，回答三个问题：**怎么建组、资源怎么分、计算怎么协调**。（下文以 v0 链路为主便于对照，V1 对应逻辑在 `trainer/ppo/v1/trainer_base.py`。）

### (1) 三个映射表是建组的"输入"

`TaskRunner.run`（v0：main_ppo_v0.py:30 起）先准备好两张表，`init_workers` 据此建组：

| 表 | 含义 | 来源 |
|----|------|------|
| `role_worker_mapping: {Role -> ray.remote(WorkerCls)}` | 每个角色用哪个 Worker 类 | `add_actor_rollout_worker`/`add_critic_worker` |
| `mapping: {Role -> pool_name}` | 每个角色放哪个资源池 | 默认全 `global_pool`，reward/teacher 可独立池 |
| `resource_pool_spec: {pool_name -> [每节点GPU数...]}` | 每个池占多少卡 | `init_resource_pool_mgr` |

关键事实：**actor、rollout、ref 三者不是三个独立 Worker，而是融合在一个 `ActorRolloutRefWorker` 里**（`add_actor_rollout_worker`，main_ppo_v0.py:35）。角色枚举取 `Role.ActorRolloutRef`（需独立 ref 时）或 `Role.ActorRollout`（ref 融进 actor，即 LoRA 的 `ref_in_actor`）。这就是 verl 的 **hybrid engine**：一个 actor 进程内同时持有训练引擎（FSDP/Megatron）和推理引擎（vLLM），靠 sleep/wake 切换显存。

### (2) 建组流程：按"资源池"聚合，colocate 进同一 actor 再 spawn 分裂

`init_workers` 的核心是把"同一资源池里的多个角色"塞进一个 colocated actor，再分裂出各自的 WorkerGroup 视图：

```python
# ray_trainer.py:784
self.resource_pool_to_cls = {pool: {} for pool in resource_pool_dict.values()}

# 1) 把每个角色的"类+初始化参数"登记到它所属的资源池
#    actor_rollout(+ref) -> global_pool
self.resource_pool_to_cls[actor_rollout_pool][str(actor_role)] = RayClassWithInitArgs(
    cls=role_worker_mapping[actor_role], config=..., role=str(actor_role))
#    critic -> global_pool (默认)
self.resource_pool_to_cls[critic_pool][str(Role.Critic)] = critic_cls
#    ref (仅当独立 ref 且非 ref_in_actor) -> global_pool
self.resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = ref_policy_cls

# 2) 每个资源池建一个 colocated WorkerGroup, 再 spawn 出每个角色的视图
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)   # 多角色合成一个 actor 类
    wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())   # 分裂: {"ActorRollout": wg, "Critic": wg, ...}
    all_wg.update(spawn_wg)

self.actor_rollout_wg = all_wg[str(actor_role)]   # 拿到各角色 WorkerGroup 句柄
self.critic_wg        = all_wg[str(Role.Critic)]
```

理解要点：
- **colocate 的本质**：同一资源池里若有多个角色（如 critic 也用 `global_pool`），`create_colocated_worker_cls` 会把它们合成一个 `WorkerDict` actor，多个角色**共享同一进程、同一张物理卡**（每个声明占 `1/max_colocate_count` 卡，见 §10.3）。`spawn` 再按角色前缀（`Critic_`/`ActorRollout_`）把方法 rebind 成独立 WorkerGroup 视图（ray/base.py:730）。
- **为什么 rollout 最后才 `init_model`**（:894）：注释明说"so that vllm can have a better estimation of kv cache memory"——先让 actor/critic/ref 占好显存，vLLM 再按剩余显存估算 KV cache，避免 OOM。
- **想给某角色不同并行度/独立卡**：就给它分**独立资源池**（如 reward 用 `reward_pool`），不要走 `create_colocated_worker_cls`，直接用独立 `resource_pool` 建 WorkerGroup（源码注释 :839 指明）。

### (3) 五类模型的归属与资源分配一览

| 模型 | Worker 类 | 默认资源池 | WorkerGroup 句柄 | 备注 |
|------|-----------|-----------|------------------|------|
| **Actor**（训练） | `ActorRolloutRefWorker`（含训练引擎） | `global_pool` | `actor_rollout_wg` | 与 rollout 同进程 |
| **Rollout**（生成） | 同上（hybrid，内嵌 vLLM/SGLang） | `global_pool` | `actor_rollout_wg` | sleep/wake 共享显存 |
| **Ref**（参考） | 同上（融合）或独立 `RefPolicy` | `global_pool` | `ref_policy_wg` | `ref_in_actor` 时直接=actor_rollout_wg（:898） |
| **Critic**（价值） | `TrainingWorker`（value_model） | `global_pool` | `critic_wg` | 仅 GAE 路线建 |
| **Reward**（奖励） | 不建训练 WorkerGroup | `reward_pool` 或无 | 经 `RewardLoopManager` | 见下 (4) |

### (4) Reward / Rollout 的特殊协调：Manager 而非 WorkerGroup

reward 和 rollout 的"生成/打分"不直接用 `wg.xxx` 同步调用，而是通过几个 **Manager** 协调异步流水线（`init_workers` 尾部，:900~969）：

```
LLMServerManager.create(worker_group=actor_rollout_wg, rollout_resource_pool=...)
        │  把 actor_rollout 的 rollout 引擎包装成可异步请求的 LLM 服务+副本(replica)
        ▼
AgentLoopManager.create(llm_client=..., reward_loop_worker_handles=...)
        │  每个 prompt 起 n 个异步 agent loop: 调 LLMServer 生成 -> 调 reward 打分
        ▼
RewardLoopManager(rm_resource_pool=...)
        │  reward model 打分(可 colocate 或独立 reward_pool)
        ▼
CheckpointEngineManager(trainer=actor_rollout_wg, replicas=...)
           负责把 actor 新权重同步给 rollout 副本(权重回灌)
```

- **`enable_agent_reward_loop`**（:940）：无 reward model、或 reward model 有独立资源池时，把 reward 计算**流式**嵌进 rollout（边生成边打分），而非生成完再整批打分。
- 这解释了为什么 §11 里 reward 站点"落在 reward_manager/模型"而非某个固定 WorkerGroup——它由 RewardLoopManager 调度。

### (5) 计算协调全景：谁调谁

```
 Driver(RayPPOTrainer.fit)
   │
   ├─ async_rollout_manager.generate_sequences(batch)   ──► AgentLoop ──► LLMServer(actor_rollout_wg 的 rollout 引擎)
   │                                                                  └─► RewardLoop(reward 打分)
   ├─ actor_rollout_wg.compute_log_prob(batch)           ──► actor 训练引擎前向(mesh=actor, NCCL)
   ├─ ref_policy_wg.compute_ref_log_prob(batch)          ──► ref 引擎前向(mesh=ref)   [可选]
   ├─ critic_wg.infer_batch(batch)                       ──► critic 前向(mesh=train)  [仅GAE]
   ├─ <Driver 本地> apply_kl_penalty + compute_advantage
   ├─ critic_wg.train_mini_batch(batch)                  ──► critic 前向+反向         [仅GAE]
   ├─ actor_rollout_wg.update_actor(batch)               ──► actor 前向+反向(NCCL all-reduce)
   └─ checkpoint_manager.update_weights()                ──► actor 新权重 -> rollout 副本
```

协调本质：**Driver 是唯一的"指挥"，所有 WorkerGroup/Manager 都被它顺序（或流水线）调用**；同一资源池里的多个角色 colocate 在同卡上分时复用显存（actor 训练时 rollout sleep，rollout 生成时 actor offload），这正是 §10.3 `num_gpus=1/max_colocate_count` 设计的目的。各 WorkerGroup 内部的多卡协同（FSDP/TP/PP）由 NCCL 负责，与 Ray 无关（§10.6）。

### (6) 澄清：Actor 和 Rollout 共用同一句柄 = 共享显存吗？

`actor_rollout_wg` 指向**同一批 Ray actor**（每卡一个 `ActorRolloutRefWorker` 进程），进程内同时持有两个**独立**对象：`self.actor`（训练引擎 FSDP/Megatron）与 `self.rollout`（推理引擎 vLLM/SGLang）。

| 维度 | 是否共享 | 说明 |
|------|----------|------|
| 物理 GPU / 进程 | ✅ 共享 | colocate，hybrid engine |
| Ray actor 句柄 | ✅ 同一批 | Driver 调 actor 方法走 `self.actor`，调生成/同步走 `self.rollout` |
| **权重张量内存** | ❌ **两份独立拷贝** | actor 是 FSDP 分片+优化器态；rollout 是 vLLM 自有格式+KV cache，布局不同 |
| 显存容量 | ⚠️ 分时复用 | sleep/wake + offload 错峰，避免两份同时吃满 |
| 同步方式 | 每步 `update_weights` | `actor.engine.get_per_tensor_param()` → `rollout.update_weights(...)` 拷贝 |

关键证据：`update_weights`（engine_workers.py:720 起的 async 方法体）若是同一份内存就无需"同步"。其调度为——训练时 `rollout.sleep()` 释放显存；同步时 `rollout.resume(["weights"])` → 拷贝权重 → `actor.engine.to("cpu")` offload → `rollout.resume(["kv_cache"])`；生成时 rollout 占显存、actor 在 CPU。**结论：共享同卡同进程与显存容量（分时），但权重是两份独立拷贝，每步显式拷贝同步以保证 on-policy。**

### (7) 都用 global_pool 的各 role：卡数、通信组、并行策略如何各自管理

**前提**：`global_pool = [n_gpus_per_node]*nnodes`，所有 role 映射到它，意味着 actor/critic/ref **colocate 在同一批进程**（每卡一个 `WorkerDict`，内含 `self.actor`/`self.critic`/`self.ref`），`actor_rollout_wg`/`critic_wg` 只是 `spawn` 出的**方法视图**，底层同一批 actor。

| 维度 | 如何管理 |
|------|----------|
| **卡数** | 每个 role 都用**全量** `world_size = nnodes×n_gpus_per_node`，不分区、分时复用。要独立卡数 → 分独立资源池 |
| **WORLD 进程组** | `initialize_global_process_group_ray`（distributed.py:80）带 `is_initialized()` 守卫，**每进程只建一次**；同进程 colocate 的 role **共用这一个全卡 NCCL WORLD 组** |
| **各 role 子通信组** | 每个 engine 在 `_init_device_mesh`（fsdp/transformer_impl.py:211）按**自己的 config** `init_device_mesh` 在 WORLD 之上切出独立子 mesh（`new_group` 各建 NCCL communicator）；分时执行，互不干扰 |
| **并行策略** | 来自各 role 独立 config：`actor/critic.strategy`(fsdp/megatron)、`fsdp_size`、`ulysses_sequence_parallel_size`、megatron `tp/pp_size`；rollout 另建 `(dp,infer_tp,infer_pp)` mesh + `stateless_init_process_group` |
| **与 dispatch 衔接** | 各 engine 算出 `get_data_parallel_rank()`/`is_mp_src_rank_with_outputs()`，经 `_register_dispatch_collect_info(mesh_name=...)`（定义于 worker.py:86，engine_workers.py:145 调用）注册到 `actor`/`ref`/`train` mesh，供 §10.5 的 `make_nd_compute_dataproto_dispatch_fn` 按该 role 拓扑切分/收集 |

要点：**同样 8 卡，各 role 按自己策略切出不同逻辑拓扑，但都占用全部 8 卡**。一句话：共享全卡 WORLD 组，子通信组与并行策略按 role 在 WORLD 之上独立建、分时跑、互不干扰。

> ⚠️ 易错点：`fsdp_size` 是 **FSDP 分片组大小，不是占用卡数**。`create_device_mesh(world=8, fsdp_size=4)` → 2D mesh `(ddp=2, fsdp=4)` 即 `HYBRID_SHARD`：参数在每组 4 卡内切片、跨 2 组复制，**8 卡全部参与、全员数据并行**。`fsdp_size=8`(或-1) → 1D 纯 `FULL_SHARD`(ZeRO-3)。它只决定"显存 vs 通信"权衡（小 fsdp_size=多复制少跨组 all-gather，多机友好），不改变卡数；数据并行度由 `get_data_parallel_size()=world_size//ulysses_sp_size` 决定，与 fsdp_size 无关。

### (8) Actor(FSDP) 与 Rollout(vLLM TP) 并行策略不同，weight 如何同步

靠**统一的"全量 per-tensor 张量流"中间表示**，两端各自(去)分片、互不需知对方布局：

```
 actor(FSDP分片) ──full_tensor()跨FSDP组all-gather还原──► (name, 完整张量bf16) 流式generator
        │                                                          │ 统一接口 Generator[(str,Tensor)]
        ▼                                                          ▼
 get_per_tensor_param (fsdp/transformer_impl.py:796)      rollout.update_weights(...)
                                                            vLLM 按自己 infer_tp 重新切片 model.load_weights
                                                            (colocate走CUDA IPC/共享内存; 分离走checkpoint_engine)
```

- **完整张量是"公共货币"**：FSDP/Megatron 的 `get_per_tensor_param` 实现各异，但都产出"完整、HF 命名的 per-tensor 权重"；rollout 只认这个契约 → **训练并行与推理并行彻底解耦**。
- **generator 流式 + 分桶**（`update_weights_bucket_megabytes`）：一次只还原/传输一个（或一桶）张量，避免完整模型同时铺满显存。
- naive(colocate) 走 CUDA IPC 零拷贝；async(分离) 走 `checkpoint_engine.send_weights`（NCCL/mooncake/nixl），同一 per-tensor 契约。

理解这张图后，再看 [11-算法配置与跨Worker数据流转.md](11-算法配置与跨Worker数据流转.md)，就能把"每个 `wg.xxx` 调用具体传什么张量、算什么、回什么"对应起来。

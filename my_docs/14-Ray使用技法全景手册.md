# 14 · Ray 使用技法全景手册（全仓库 use-case 汇总）

> 返回索引：[README.md](README.md)
> 关联阅读：[10-Ray串联机制详解.md](10-Ray串联机制详解.md)（控制流/串联机制）、[13-异步训练方案对比.md](13-异步训练方案对比.md)
> 定位区别：`10` 讲"Ray 如何把 Driver 和 Worker 串成训练循环"（**控制流主线**）；本篇是**技法 cookbook**——把全仓库每一处 Ray API 的用法、参数、用途按类别穷尽汇总，让你看到 Ray 在 verl 里到底用了哪些"招式"。

---

## 14.0 一句话总览

verl 把 Ray 当成 **"分布式进程操作系统"** 来用，覆盖六大职责：

1. **集群/运行时**：`ray.init` + `runtime_env` 统一下发环境变量
2. **进程编排**：`@ray.remote` / `ray.remote(cls)` 把"类"变成分布在各卡的 actor
3. **资源调度**：PlacementGroup + 两种 SchedulingStrategy 精确钉进程到 bundle/节点
4. **对象传递**：`ray.put`/`ray.get` + object store 做跨进程数据通道
5. **服务化与并发**：async actor + `max_concurrency` 把 rollout 引擎变成高并发 HTTP/RPC 服务
6. **全局协调**：named/detached actor 做 rendezvous、单例、消息队列

下面逐类拆解。每条给出 **技法 → 代表出处 → 用途**。

---

## 14.1 集群启动与运行时（runtime_env）

| 技法 | 出处 | 用途 |
|------|------|------|
| `ray.is_initialized()` 守卫后再 `ray.init()` | `trainer/main_ppo.py:57` | 复用已有集群/避免重复初始化；脚本既能本地起集群也能 attach 到已有集群 |
| `ray.init(runtime_env={...})` 统一下发环境变量 | `main_ppo.py:75`，`get_ppo_ray_runtime_env`（`constants_ppo.py`） | 把 `TOKENIZERS_PARALLELISM / NCCL_DEBUG / VLLM_LOGGING_LEVEL / VLLM_ALLOW_RUNTIME_LORA_UPDATING / TRANSFER_QUEUE_ENABLE` 一次性注入**所有** actor，保证全集群环境一致 |
| `.options(runtime_env={"nsight": nsight_options})` | `main_ppo.py:91` | 给 TaskRunnerV1 actor 单独挂 nsys profiler，做性能剖析 |
| `ray.timeline(filename=...)` | `main_ppo.py:100` | 导出 Ray 任务时间线 trace，事后分析调度 |
| `ray.remote(num_cpus=1)(TaskRunnerV1)` 动态包装 | `main_ppo.py:91/93`（v0 见 `main_ppo_v0.py:30`） | 把控制器逻辑放进 `num_cpus=1` 的 actor，**避免重编排逻辑被调度到 head 节点** |

要点：**`runtime_env` 是 verl 统一环境变量的唯一入口**——不要在 Worker 里散落 `os.environ`，要全局生效就走这里。

---

## 14.2 把"类"变 actor 的两种写法

verl 同时用两种方式，区别在于"是否提前知道要 remote"：

| 写法 | 代表出处 | 适用场景 |
|------|----------|----------|
| **装饰器** `@ray.remote` | `NCCLIDStore`（`rendezvous/ray_backend.py:34`）、`TrajectoryTracker`（`trajectory_tracker.py:50`）、`MessageQueue`（`message_queue.py:26`） | 类天生就是 actor，定义即声明 |
| **动态包装** `ray.remote(cls)` | `ActorRolloutRefWorker`/`TrainingWorker`（`main_ppo_v0.py:53/64`；V1 见 `trainer/ppo/v1/trainer_base.py`）、`CheckpointEngineWorker`（`checkpoint_engine/base.py`、`replica.py`）、`vLLMHttpServer`（`vllm_async_server.py`） | 同一个普通类既要能本地实例化（测试/复用），又要能按需变成 actor；解耦"业务逻辑"与"分布式封装" |
| **包装普通函数为 task** `ray.remote(copy)` | `trajectory_tracker.py:30`（`remote_copy`），`save_to_hdfs`（`:34` 装饰器） | 无状态计算（如 HDFS 上传）fan-out 成并行 task |

关键设计：**Worker 业务类本身不带 `@ray.remote`**，由 `main_ppo.py`（V1 在 `trainer_base.py`）在建 `role_worker_mapping` 时才 `ray.remote(cls)`，这样 `single_controller` 的机制能透明地处理 actor/非 actor 两种形态。

---

## 14.3 资源调度：PlacementGroup + 两种 SchedulingStrategy

这是 verl 把进程精确"钉"到物理卡/节点的核心，全部集中在 `single_controller/ray/base.py` 与各 rollout server。

### (1) PlacementGroup 预占资源
```python
placement_group(bundles=bundles, strategy="STRICT_PACK", name=..., lifetime=...)  # base.py:156
ray.get([pg.ready() for pg in pgs])              # 阻塞等资源就位
sort_placement_group_by_node_ip(pgs)             # 按节点IP排序 -> 多job间RANK稳定(断点续训关键)
```
- `STRICT_PACK`：同一 PG 的所有 bundle 必须落在同一节点（保证一组卡同机，NCCL 高带宽）。
- 一个 bundle = `{"CPU":k, "GPU":1}`，一张卡一个 bundle。

### (2) `PlacementGroupSchedulingStrategy` —— 钉到指定 bundle
```python
# base.py:398
PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=local_rank)
```
用途：让 actor 精确落到 PG 内第 `local_rank` 个 bundle，从而**保证 RANK↔物理卡的稳定映射**（这是 §10 dispatch 切分能对齐的前提）。

### (3) `NodeAffinitySchedulingStrategy` —— 钉到指定节点
```python
# vllm_async_server.py:1028 / async_sglang_server.py:790 / trtllm_async_server.py:601
# reward_loop.py:316 / reward_manager/remote.py:63 / agent_loop.py:1091
NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
```
用途：rollout HTTP server actor 必须和它驱动的 GPU worker **同节点**（否则跨节点驱动引擎不可行）。先用 `__ray_call__` 探出每个 worker 的 `node_id`，再把 server 用 `soft=False`（硬亲和）钉过去。reward/agent loop 同理把计算 actor 钉到数据所在节点减少传输。

---

## 14.4 Named / Detached actor —— 全局协调三大场景

Ray 默认 actor 句柄随创建者生命周期；verl 用命名/分离 actor 实现"跨进程发现"与"全局单例"。

### 场景 A：NCCL rendezvous 协调点（`rendezvous/ray_backend.py`）
```python
# rank0 发布
nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)    # :65 named actor
# 其它 rank 发现 + 轮询重试
all_actors = list_named_actors(all_namespaces=True)                     # :44 跨namespace枚举
actor = ray.get_actor(**matched)                                       # :48 按名取句柄
for i in range(max_retries): ...; time.sleep(interval_s)               # :75 轮询等待出现
```
用途：用一个轻量 named actor 当"白板"，把 NCCL unique-id 从 rank0 广播给其它 rank——**这是用 Ray 实现的 rendezvous，替代 TCP store / 文件**。

### 场景 B：全局单例 actor（`trajectory_tracker.py:83`）
```python
TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(...)
```
三连参数含义：
- `name=` 全局唯一名字；
- `get_if_exists=True` 已存在则直接复用（任意进程调用都拿到同一个）；
- `lifetime="detached"` 脱离创建者存活，driver 退出也不回收。
用途：调试时所有进程把中间张量 dump 给**同一个** tracker，由它统一异步上传 HDFS。

### 场景 C：rollout server 按名复用（`sglang_rollout.py:237`）
```python
self.server_actor = ray.get_actor(actor_name)
```
用途：worker 进程内反查同名 server actor 句柄，建立 worker↔server 关联。

> ⚠️ 坑点（`base.py:512`）：`ray.get_actor` 持有的是**弱引用**，可能导致 actor 被意外 GC——所以 `RayWorkerGroup` 优先用直接持有的 `worker_handles`，仅在没有时才退化为按名 `ray.get_actor`。

---

## 14.5 ObjectRef 与数据传递

| 技法 | 出处 | 用途 |
|------|------|------|
| `parallel_put`：ThreadPool 并行 `ray.put` | `utils/ray_utils.py:51` | 把一个 list 并行塞进 object store（默认 16 线程），返回**保持原序**的 ObjectRef 列表，规避逐个 `ray.put` 串行瓶颈 |
| `ray.get(list_a + list_b)` 拼接 future 一起收 | `checkpoint_engine/base.py:392` | 把 trainer WG 与 rollout WG 两批 future 合并成一个 `ray.get`，形成**统一 barrier**（两边都完成才返回） |
| ObjectRef 当 `.remote()` 参数按 rank 切分下发 | `base.py:866` `execute_all_async` | dispatch 切好的 `[shard_0,...,shard_{N-1}]`，第 i 份发给第 i 个 actor（见 §10.5） |
| `RayWorkerGroup(worker_handles=workers,...)` 复用现有 actor | `checkpoint_engine/base.py:489` | 用已存在的 actor 句柄**临时组装**一个 WorkerGroup，复用 dispatch 机制驱动权重同步，无需新建进程 |

---

## 14.6 异步 actor + max_concurrency（rollout 服务化的核心）

这是 verl 把推理引擎变成"高并发服务"的关键技法，集中在各 `*_async_server.py` 与 `replica.py`。

### (1) `max_concurrency` —— 单 actor 内并发执行 async 方法
```python
# replica.py:257
@property
def max_concurrency(self):
    # 1000 是 Ray async actor 默认上限；再加控制方法余量
    return max(1000, self.config.max_num_seqs + CONTROL_METHOD_CONCURRENCY)  # =16
```
```python
# vllm_async_server.py:1034 创建 server actor 时传入
self.server_class.options(
    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
    runtime_env={"env_vars": env_vars},
    name=name,
    max_concurrency=self.max_concurrency,    # 允许成百上千个 generate 请求并发驻留
).remote(...)
```
用途：一个 vLLM server actor 要**同时**处理 `max_num_seqs` 个 `generate` 协程 + 若干控制方法（sleep/wake/abort）。默认 actor 串行执行方法会卡死并发生成，故显式放大并发度。

### (2) async actor 方法 + `await actor.method.remote()`
Ray 对 async actor 的 ObjectRef 可直接 `await`。verl 大量用 `asyncio.gather` 对一个 replica 内多个 server 节点**并发 fan-out**：
```python
# replica.py:267 等
await asyncio.gather(*[server.wake_up.remote() for server in self.servers])
await asyncio.gather(*[server.sleep.remote() for server in self.servers])
await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])
```
用途：生命周期操作（wake/sleep/abort/profile/kv_cache）对所有节点并发下发，最大化吞吐。

### (3) Ray ObjectRef ↔ asyncio.Future 桥接（`message_queue.py:189`）
```python
future = self.queue_actor.put_sample.remote(sample)   # Ray ObjectRef
return await asyncio.wrap_future(future.future())      # .future() 转 concurrent.futures.Future, 再包成 asyncio
```
用途：在**普通 async 函数**（非 actor 内）里 await Ray 调用，而不阻塞事件循环。fully_async 的 rollouter/trainer（`fully_async_trainer.py:508/521/524`、`fully_async_rollouter.py:382`）都用这招与 actor 通信。

### (4) `auto_await` 装饰器（`ray_utils.py:97`）
处理"同一个 async 方法被 await / 被同步直接调 / 在已运行 loop 里调"三种情形，避免事件循环冲突。被 `CheckpointEngineManager`、`LLMServerManager` 等复用。

---

## 14.7 用 async actor 自造消息队列（替代 ray.util.queue）

`fully_async_policy/message_queue.py` 没有用 Ray 自带的 `ray.util.queue.Queue`，而是**手写**一个 async actor 队列：
```python
@ray.remote(num_cpus=2, max_concurrency=20)     # :26
class MessageQueue:
    def __init__(...):
        self.queue = deque(maxlen=max_queue_size)
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)   # 阻塞式消费
    async def put_sample(...):  # 满了丢最旧(背压), notify_all 唤醒消费者
    async def get_sample(...):  # 空了 await condition, 阻塞等待生产
```
为什么自造：需要**自定义背压策略**（队列满丢最旧样本，控制 staleness）、统计指标（produced/consumed/dropped）、validate 独立队列、阻塞式 `get`——这些 `ray.util.queue` 不直接提供。配套 `MessageQueueClient`（:180）用 §14.6(3) 的 future 桥接做异步访问。

用途：解耦 fully-async 训练里 **Rollouter（生产样本）↔ Trainer（消费样本）**，让两者各自按节奏跑（见 [13-异步训练方案对比.md](13-异步训练方案对比.md)）。

---

## 14.8 Runtime context 内省（拿进程的"身份证"）

actor 进程需要知道自己被 Ray 分到了哪张卡/哪个节点，用 `ray.get_runtime_context()`：

| API | 出处 | 用途 |
|-----|------|------|
| `.get_accelerator_ids()[device][0]` | `worker.py:279`、`distributed.py:46` | 拿到本进程实际分到的 GPU 物理 id → 设 `LOCAL_RANK`/`CUDA_VISIBLE_DEVICES` |
| `.get_node_id()` | `vllm_async_server.py:997` 等 | 探出 worker 所在节点，供 server 做 `NodeAffinity` 亲和 |
| `.get_job_id()` | `vllm_async_server.py:124`、`vllm_rollout.py:107` | 把 Ray job id 注入 vLLM 子进程，使 colocate 权重传输的 IPC socket 路径**按 job 唯一**，避免同节点两个 job 撞 socket（EADDRINUSE） |
| `ray.util.get_node_ip_address()` | `vllm_async_server.py:146` | 拿本机 IP 作为 HTTP server 监听地址 / DP master 地址 |
| `worker.__ray_call__.remote(lambda self: ...)` | `vllm_async_server.py:995`、`async_sglang_server.py:746`、`sglang_pd_replica.py:101` | **在远程 actor 上执行任意闭包**：一次 round-trip 同时取回 `(node_id, accelerator_ids)`，无需在 Worker 类预定义专门方法 |

`__ray_call__` 是很灵活的技法：不用给 actor 加方法，就能远程跑一段逻辑读取它的运行时状态。

---

## 14.9 fully-async 编排：ray.wait / ray.cancel

`fully_async_policy/fully_async_main.py` 用底层 future 原语做"赛跑式"编排：
```python
done_futures, remaining = ray.wait(futures, num_returns=1, timeout=None)  # :188 等任一完成
...
ray.cancel(remaining_future)   # :197/:205 取消其余未完成的 actor 任务
```
用途：rollouter 和 trainer 两个长跑 actor 任务并行，**谁先结束/出错就响应**，并取消另一个，实现异步训练的主循环控制。

配套的高并发控制器 actor：
```python
@ray.remote(num_cpus=10, max_concurrency=100)   # one_step_off_policy/main_ppo.py:34, fully_async_rollouter.py:394
```
`num_cpus=10` 抢足 CPU 避免被压在 head；`max_concurrency=100` 允许控制器并发处理大量协调请求。

---

## 14.10 checkpoint_engine：用 Ray 驱动权重回灌

`checkpoint_engine/base.py` 把"训练侧新权重 → 推理侧 rollout"的同步也用 Ray 编排：
- `@register(dispatch_mode=Dispatch.ONE_TO_ALL/DP_COMPUTE, blocking=False) async def`（:322/327）：注册**非阻塞**远程方法，配合 dispatch 机制下发。
- `ray.get(trainer.execute_checkpoint_engine([...]) + rollout.execute_checkpoint_engine([...]))`（:392）：两侧 future 合并 barrier。
- `asyncio.gather(*[r.xxx() for r in self.replicas])`（:431+）+ `auto_await`：对所有 rollout replica 并发执行 sleep/wake/release_kv_cache 等生命周期。
- 不同 backend（naive CUDA-IPC / nccl / nixl / mooncake / kimi / hccl）共享同一套 Ray 编排，只是底层传输通道不同。

用途：见 §10.8(8)——Ray 负责"触发与编排"权重同步，真正的张量搬运走 NCCL/IPC/mooncake。

---

## 14.11 Ray vs torch.distributed 边界（再次强调）

| 走 Ray | 走 NCCL/torch.distributed |
|--------|---------------------------|
| Driver→Worker RPC、dispatch 切片下发、对象传递、actor 生命周期、资源调度、服务并发、消息队列、rendezvous 协调 | Worker↔Worker 的 all-gather/reduce-scatter/all-reduce（FSDP/TP/PP/DP）、权重张量实际搬运 |

Ray 只帮各 Worker 凑齐 `RANK/MASTER_ADDR/PORT`（注入 env），建组与集合通信本身完全在 NCCL（详见 [10-Ray串联机制详解.md](10-Ray串联机制详解.md) §10.6）。

---

## 14.12 全仓库技法速查表（按 API）

| Ray API / 技法 | 主要出处 | 一句话用途 |
|----------------|----------|-----------|
| `ray.init(runtime_env=)` | main_ppo.py:75 | 起集群+统一环境变量 |
| `ray.is_initialized()` | main_ppo.py:57 | 复用/避免重复初始化 |
| `ray.remote(cls)` 动态包装 | main_ppo_v0.py:53、replica.py | 普通类按需变 actor |
| `@ray.remote(num_cpus=,max_concurrency=)` | message_queue.py:26 | 声明高并发 async actor |
| `ray.remote(func)` | trajectory_tracker.py:30 | 普通函数变并行 task |
| `.options(name=,scheduling_strategy=,runtime_env=,max_concurrency=,lifetime=,get_if_exists=)` | vllm_async_server.py | 创建 actor 时定制调度/命名/并发/生命周期 |
| `placement_group(strategy="STRICT_PACK")` | ray/base.py | 同机预占 GPU bundle |
| `PlacementGroupSchedulingStrategy(bundle_index=)` | ray/base.py | 钉 actor 到指定卡 |
| `NodeAffinitySchedulingStrategy(node_id,soft=False)` | vllm_async_server.py | 钉 server 到 worker 同节点 |
| `list_named_actors(all_namespaces=True)` | ray_backend.py:44 | 跨 namespace 枚举命名 actor |
| `ray.get_actor(name=)` | ray_backend.py:48、ray/base.py | 按名取 actor 句柄 |
| `get_if_exists=True, lifetime="detached"` | trajectory_tracker.py:83 | 全局单例、跨 driver 存活 |
| `ray.put` + ThreadPool | ray_utils.py:51 | 并行入 object store |
| `ray.get(list_a+list_b)` | checkpoint_engine/base.py | 合并 future 做统一 barrier |
| `ray.wait(num_returns=1)` | fully_async_main.py:188 | 等任一任务完成 |
| `ray.cancel(ref)` | fully_async_main.py:197 | 取消未完成任务 |
| `await actor.m.remote()` + `asyncio.gather` | replica.py | async actor 并发 fan-out |
| `ref.future()` + `asyncio.wrap_future` | message_queue.py:189 | Ray ObjectRef→asyncio 桥接 |
| `ray.get_runtime_context().get_accelerator_ids()` | worker.py:279 | 拿本进程 GPU 物理 id |
| `.get_node_id()/.get_job_id()` | vllm_async_server.py | 节点亲和 / job 级 socket 隔离 |
| `worker.__ray_call__.remote(lambda)` | vllm_async_server.py | 远程执行任意闭包读运行时状态 |
| `ray.util.get_node_ip_address()` | vllm_async_server.py | 取本机 IP 作监听地址 |
| `ray.timeline(filename=)` | main_ppo.py:100 | 导出调度时间线 |

---

## 14.13 学习路线建议

1. 想懂"训练主循环怎么被 Ray 串起来"→ 先读 [10](10-Ray串联机制详解.md)。
2. 想懂"一次 `wg.xxx()` 怎么变成远程调用"→ 读 [12](12-decorator与dispatch机制详解.md)。
3. 想懂"异步训练里 actor 怎么解耦协作"→ 读本篇 §14.6~14.9 + [13](13-异步训练方案对比.md)。
4. 想做 rollout 服务化/自定义引擎 → 重点看本篇 §14.3(3)/14.6（NodeAffinity + max_concurrency + async actor 是三件套）。

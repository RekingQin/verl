# 12 · decorator.py 设计与用途详解（声明式分布式的核心）

> 返回索引：[README.md](README.md)
> 关联阅读：[10-Ray串联机制详解.md](10-Ray串联机制详解.md)（远程调用全链路）、[11-算法配置与跨Worker数据流转.md](11-算法配置与跨Worker数据流转.md)
> 源码：`verl/single_controller/base/decorator.py`（444 行）

---

## 12.0 它解决什么问题（设计定位）

verl 的单控制器哲学要求：Driver 用"近似单机"的命令式代码 `wg.compute_log_prob(batch)` 调用，而**不关心数据如何切分到多卡、并发执行、再聚合回来**。

`decorator.py` 就是把这件"脏活"从业务逻辑里抽离出来，变成 **一行声明**：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_log_prob(self, data): ...
```

它本身**不执行任何远程调用**，只做两件事：
1. **定义"数据分发/聚合策略"**（一组 dispatch_fn / collect_fn 函数族）。
2. **`@register` 把策略以元数据形式贴到方法上**（写进 `MAGIC_ATTR`），供 WorkerGroup 绑定期读取（见 [10](10-Ray串联机制详解.md) §10.5）。

可以理解为：`decorator.py` 是"声明层"，`worker_group.py` + `ray/base.py` 是"执行层"。两者通过 `MAGIC_ATTR` 这个约定解耦。

---

## 12.1 文件结构总览

```
decorator.py
├── Dispatch(DynamicEnum)        # 分发模式枚举
├── Execute(DynamicEnum)         # 执行模式枚举 (ALL / RANK_ZERO)
├── init_predefined_*_mode()     # 注册内置模式 (import 时即执行)
│
├── 切分工具
│   ├── _split_args_kwargs_data_proto              # 纯 chunk
│   └── _split_args_kwargs_data_proto_with_auto_padding  # chunk 前自动 padding
│
├── dispatch / collect 函数族 (策略实现)
│   ├── one_to_all      / collect_all_to_all
│   ├── all_to_all      / collect_all_to_all
│   ├── dp_compute(_data_proto / _with_func / _metric)
│   ├── nd_compute(_dataproto)   # mesh-aware (DP/TP/PP)
│   └── lazy_compute(_data_proto) + make_nd_compute_dataproto_dispatch_fn
│
├── DISPATCH_MODE_FN_REGISTRY    # 模式 -> {dispatch_fn, collect_fn}
├── register_dispatch_mode / update_dispatch_mode   # 自定义扩展入口
├── get_predefined_dispatch_fn / get_predefined_execute_fn
│
└── register(...)                # 主装饰器 (核心出口)
```

---

## 12.2 两个 DynamicEnum：Dispatch 与 Execute

```python
class Dispatch(DynamicEnum):     # 可动态扩展的枚举
    _registry = {}; _next_value = 0

def init_predefined_dispatch_mode():   # import 时执行
    Dispatch.register("RANK_ZERO");  Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL"); Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO"); ...
    Dispatch.register("DIRECT_ROLLOUT_METHOD")
```

设计要点：用 **`DynamicEnum` 而非 Python `Enum`**，是为了支持运行时**注册自定义 dispatch 模式**（`register_dispatch_mode`），让外部 recipe 能扩展而无需改框架源码。

- **`Dispatch`**：决定"数据怎么切、结果怎么聚"。
- **`Execute`**：决定"在哪些 rank 上执行"——`ALL`（全部）或 `RANK_ZERO`（仅 0 号）。映射到 WorkerGroup 的 `execute_all` / `execute_rank_zero`（`get_predefined_execute_fn`，:357）。

---

## 12.3 dispatch / collect 函数族（策略的具体实现）

每个 dispatch 模式对应一对 `(dispatch_fn, collect_fn)`，统一签名 `fn(worker_group, *args, **kwargs)`。下面按"由简到繁"讲解，并标注用途。

### (1) ONE_TO_ALL：广播
```python
def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg]*world_size for arg in args)   # 同一份复制 N 份
    return args, kwargs
```
**用途**：所有 rank 收到完全相同的参数。collect 用 `collect_all_to_all`（原样返回 list）。
**真实场景**（engine_workers.py）：`init_model` / `reset` / `to(device)` / `save_checkpoint` / `load_checkpoint` / `set_loss_fn` / `update_weights`——这些是"对每个 rank 做同样的控制动作"，不涉及数据切分。

### (2) ALL_TO_ALL：透传
`dispatch_all_to_all` 原样返回。**用途**：参数已是 per-rank 形态、或不需切分的默认模式（`register` 默认值）。

### (3) DP_COMPUTE_PROTO：数据并行切分（最常用）
```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    # 先 auto-padding 到能被 world_size 整除, 再 chunk(world_size)
    return _split_args_kwargs_data_proto_with_auto_padding(world_size, *args, **kwargs)

def collect_dp_compute_data_proto(worker_group, output):
    return BatchData(output).concat()    # 各 rank 结果拼回整批
```
**用途**：把一个大 batch 均分到各 DP rank 并发算，再 `concat` 回来。这是 logprob / train / value 的主力模式。

**auto-padding 的精妙处**（:91）：DataProto 长度未必能被 world_size 整除，于是 dispatch 时补齐 `padding_size` 个样本，并把这个数通过 `_padding_size_key` 透传出去；`func_generator`（ray/base.py:53）在 collect 后**再把补齐的样本切掉**，对 Driver 完全透明：
```python
padding_count = kwargs.pop(_padding_size_key, 0)
...
if padding_count > 0: output = output.select_idxs(range(len(output))[:-padding_count])
```

### (4) DP_COMPUTE_PROTO_WITH_FUNC / DP_COMPUTE_METRIC
- `_with_func`：第一个参数是**函数本身**，广播给所有 rank，其余参数 chunk（用于把自定义 func 一起下发执行，如 `Worker.execute_with_func_generator`）。
- `_metric`：dispatch 同 data_proto，但 collect 用 `collect_dp_compute`（返回 list 不 concat），用于收集各 rank 的指标。

### (5) nd_compute：mesh-aware（DP/TP/PP 感知，3D 并行关键）
普通 DP 切分假设"world_size == DP size"，但 Megatron 下一个 worker group 同时有 TP/PP，**只能按 DP 维切，且同一 DP 组内的多个 TP/PP rank 要拿同一份数据**。`nd_compute` 解决这个：
```python
def dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *args, **kwargs):
    # 数据只切成 dp_size 份; 第 i 个 global rank 拿它所属 DP 组的那份
    for i in range(world_size):
        transformed_args.append(arg[dp_rank_mapping[i]])

def collect_nd_compute(collect_mask, worker_group, output):
    # 只从 is_collect=True 的 rank (每个 DP 组的 MP 源 rank) 收, 避免 TP/PP 冗余
    return [output[r] for r in range(world_size) if collect_mask[r]]
```

`dp_rank_mapping` / `collect_mask` 从哪来？**惰性查询**（`dispatch_lazy_compute_data_proto`，:266）：第一次调用某 mesh 的方法时，向各 worker 查询并缓存：
```python
def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    return {"dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
            "collect_fn":  partial(collect_lazy_compute_data_proto, mesh_name)}
```
而 worker 侧在初始化时登记自己的 DP 拓扑（engine_workers.py:137 `_register_dispatch_collect_info(mesh_name="train", dp_rank, is_collect)`）。
**真实场景**：`compute_log_prob`(mesh=actor)、`compute_ref_log_prob`(mesh=ref)、`train_batch`/`infer_batch`/`train_mini_batch`(mesh=train)——所有重计算都用这个 mesh-aware 模式。

### (6) DIRECT_ROLLOUT_METHOD：禁用占位
`dummy_direct_rollout_call` 直接抛异常。它是 vLLM external executor 的特殊标记（绕过常规 dispatch，由 `_bind_workers_method_to_parent` 特判直接绑定，ray/base.py:955），并非真要执行。

### 注册表与扩展
```python
DISPATCH_MODE_FN_REGISTRY = { Dispatch.ONE_TO_ALL: {...}, Dispatch.DP_COMPUTE_PROTO: {...}, ... }
def register_dispatch_mode(name, dispatch_fn, collect_fn): ...   # 外部扩展自定义模式
```
`dispatch_mode` 也可直接传一个 `{"dispatch_fn":..., "collect_fn":...}` dict（`_check_dispatch_mode` 校验），`make_nd_compute_dataproto_dispatch_fn` 返回的正是这种 dict——**这是不进全局注册表也能用自定义策略的口子**。

---

## 12.4 register()：主装饰器本体（:398）

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL,
             blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode); _check_execute_mode(execute_mode)

    def decorator(func):
        func = tqbridge(dispatch_mode=dispatch_mode)(func)   # ① 套 TransferQueue 桥

        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)  # ② DataProtoFuture -> 实体
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs): ...              # ③ async 版本

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)                     # ④ 贴元数据
        return wrapper
    return decorator
```

四个设计点：

**① tqbridge 套层**（transferqueue_utils.py:298）：同步链路里 Driver 传的是 `KVBatchMeta`（元数据句柄）。`tqbridge` 在 Worker 内自动把句柄 `_meta_to_realdata` 还原成真实张量（从 TransferQueue `kv_batch_get`），执行后再把输出 `TensorDict` 写回 TQ 并返回更新后的 meta（`_update_meta_with_output`）。**TQ 未启用时 `_find_meta` 返回 None，直接透传原函数**——这让同一份 worker 方法同时支持经典 DataProto 链路和同步 TQ 链路（见 [11](11-算法配置与跨Worker数据流转.md) §11.4）。注意它还用 `dispatch_mode` 算 `need_collect`，与 nd_compute 的 collect_mask 语义对齐，避免 TP/PP 冗余写回。

**② materialize_futures**（`_materialize_futures`，:383）：若参数是 `DataProtoFuture`（异步流水线里尚未就绪的数据），先 `.get()` 阻塞取实体再执行。让上游可以 fire-and-forget 传 future。

**③ async 双版本**：`update_weights` 等是 `async def`，`register` 用 `inspect.iscoroutinefunction` 自动选 `async_inner`，保持协程语义。

**④ MAGIC_ATTR**（`"attrs_3141562937"`，故意用魔数避免与用户属性冲突）：把 dispatch/execute/blocking 三属性贴到方法上。WorkerGroup 绑定期靠 `hasattr(method, MAGIC_ATTR)` 识别哪些方法要代理（worker_group.py:204）。

**blocking 参数**：`blocking=False` 时 `func_generator` 不立即 `ray.get`，返回 ObjectRef，供 Driver 流水线并发（如 `train_batch`/`update_actor`/`update_weights` 都是 `blocking=False`）。

---

## 12.5 声明 → 执行 的完整衔接（一图）

```
 [声明期] @register(dispatch_mode, execute_mode, blocking)
            └─ setattr(method, MAGIC_ATTR, {三属性})
                         │
 [绑定期] WorkerGroup._bind_worker_method (worker_group.py:185)
            扫描带 MAGIC_ATTR 的方法
            ├─ dispatch_mode -> get_predefined_dispatch_fn -> (dispatch_fn, collect_fn)
            ├─ execute_mode  -> get_predefined_execute_fn  -> execute_all / execute_rank_zero
            └─ func_generator(self, name, dispatch_fn, collect_fn, execute_fn, blocking)
                         │  在 WorkerGroup 上生成同名代理方法
 [调用期] wg.compute_log_prob(batch)  -> Functor.__call__ (ray/base.py:49)
            dispatch_fn 切分 -> execute_fn 各 rank .remote() -> ray.get(if blocking)
            -> collect_fn 聚合 -> 去 padding -> 返回
```

注意：`Worker` 基类本身也用 `@register` 装饰了 `_query_dispatch_info`(ONE_TO_ALL)、`execute_with_func_generator`(DP_COMPUTE_PROTO_WITH_FUNC) 等"系统方法"——dispatch 机制自身的元信息查询也复用同一套装饰器。

---

## 12.6 真实使用场景速查（来自 `engine_workers.py`）

| 方法 | dispatch_mode | blocking | 为什么这么选 |
|------|---------------|----------|--------------|
| `init_model` / `reset` / `to` | `ONE_TO_ALL` | True | 对每个 rank 做相同初始化/搬运，无数据切分 |
| `set_loss_fn` | `ONE_TO_ALL` | True | 把 loss 函数广播给所有 rank |
| `save_checkpoint` / `load_checkpoint` | `ONE_TO_ALL` | True | 各 rank 存/读自己分片 |
| `compute_log_prob` | `nd(mesh="actor")` | True | 按 actor 的 DP 切分前向，TP/PP 不冗余 collect |
| `compute_ref_log_prob` | `nd(mesh="ref")` | True | 同上，ref 的 mesh |
| `train_batch` / `train_mini_batch` / `infer_batch` | `nd(mesh="train")` | **False** | 训练前向反向；非阻塞以便流水线并发 |
| `update_actor` | `nd(mesh="actor")` | **False** | 同上 |
| `update_weights` | `ONE_TO_ALL` | **False** | async 权重同步，各 rank 同时触发 |
| `execute_checkpoint_engine` | `DP_COMPUTE` | False | 按 DP 分发 ckpt 引擎调用 |
| RewardManager 的 `@register("naive")` | —（**另一个 register**） | — | ⚠️ 这是 reward_manager 的注册表，与本文 decorator 同名不同物 |

> ⚠️ 易混点：`workers/reward_manager/*.py` 里的 `@register("naive")` 是 **reward_manager 注册表**（`reward_manager/registry.py`），和 `single_controller` 的 `register` 完全是两套东西，别看到 `@register` 就以为是 dispatch。

### 澄清：`update_weights` 用 `ONE_TO_ALL` 不代表"权重经 rank0 中转"

常见误解：以为 `ONE_TO_ALL` 是"把权重 all-gather 到 rank0 再 broadcast 给各 inference engine 切片"。**不是。**

- **dispatch 只管"调用与标量参数"**：`update_weights(global_steps, mode)` 参数全是标量，没有权重张量。`ONE_TO_ALL` 仅把这两个标量复制给每个 rank，让**每张卡各自独立执行一次** `update_weights`。权重张量自始至终不经 Driver、不经 dispatch。
- **权重还原是"组内 all-gather，人人得全量"**：每个 rank 内部 `get_per_tensor_param()→full_tensor()` 是 FSDP 组的 collective all-gather，所有 rank 都拿到完整张量，**没有 rank0 中心节点**。
- **slice 在本卡 vLLM 上做**：colocate 下每个训练 rank 把完整张量经 CUDA IPC/ZMQ 喂给**同卡**的 vLLM worker，后者按**自己的 infer_tp** 切片加载（`update_weights_from_ipc`）。

```
误解:  actor ──all-gather──► rank0 ──broadcast──► 各engine ──► slice            ❌
实际:  Driver ──ONE_TO_ALL(广播标量)──► 每rank独立跑; rank内组内all-gather(人人全量)
       ──CUDA IPC──► 本卡vLLM按自己TP切片                                        ✅
```

即 `ONE_TO_ALL` = "这次调用对所有 rank 各跑一遍"，权重的搬运/切分是 Worker 内部按 rank 并行的事，与 dispatch 模式无关（详见 [10](10-Ray串联机制详解.md) §10.8(8)）。

---

## 12.7 设计精髓总结

1. **声明与执行解耦**：`decorator.py` 只声明策略并贴元数据，真正的切分/远程/聚合在 worker_group + ray/base，靠 `MAGIC_ATTR` 约定连接。改一个 `dispatch_mode` 就能改变并行行为，业务代码零改动。
2. **策略即函数 + 注册表**：dispatch/collect 是一对纯函数，存在 `DISPATCH_MODE_FN_REGISTRY`；`DynamicEnum` + `register_dispatch_mode` 支持运行时扩展自定义并行策略。
3. **对 Driver 透明的工程细节**：auto-padding（凑整除）、去 padding、future materialize、async 适配、TransferQueue 桥接（`tqbridge`）全被这一层吸收，Driver 只写单机式调用。
4. **mesh-aware 是 3D 并行的关键**：`make_nd_compute_dataproto_dispatch_fn` + 惰性查询 DP 拓扑，让同一套装饰器同时适配纯 DP（FSDP）与 DP/TP/PP（Megatron）。

新增并行策略/算法时，本文件通常**不用改**——要么用现成 `dispatch_mode`，要么 `register_dispatch_mode` 注册新策略；详见 [06-自定义扩展方式.md](06-自定义扩展方式.md)。

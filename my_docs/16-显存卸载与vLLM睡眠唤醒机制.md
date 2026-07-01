# 显存卸载（offload）与 vLLM 睡眠/唤醒机制

> 解读基线：`feat/verl-learning` 分支。核心代码：`verl/workers/engine_workers.py`、
> `verl/workers/rollout/vllm_rollout/vllm_rollout.py`、`verl/third_party/vllm/__init__.py`、
> `verl/utils/{megatron_utils,fsdp_utils}.py`。

在 hybrid engine（actor / critic / ref / rollout 共享同一批 GPU）下，显存峰值不能叠加，
因此 verl 对「训练态模型」和「推理引擎」采用两套完全不同的显存管理策略。

## 一、训练侧（actor / critic / ref）offload

对象是 PyTorch 训练张量，按需在 CPU↔GPU 搬运。三个开关：

| 开关 | 对象 | actor | critic | ref |
|------|------|-------|--------|-----|
| `param_offload` | 模型参数 | ✓ | ✓ | ✓ |
| `optimizer_offload` | 优化器状态 | ✓ | ✓ | ✗（无优化器）|
| `grad_offload` | 梯度 | ✓ | ✓ | ✗（forward_only）|

调用模式：**用时 load，用完 offload**，包在每个计算方法首尾（`update_actor` / `update_critic` /
`compute_log_prob` 等）：

```python
def update_actor(self, data):
    if self._is_offload_param:      load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:  load_fsdp_optimizer(self.actor_optimizer, device_id)
    ...  # 训练
    if self._is_offload_param:      offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:  offload_fsdp_optimizer(self.actor_optimizer)
```

- **FSDP**：`offload_fsdp_model_to_cpu` / `offload_fsdp_optimizer`（张量 `.to("cpu")`）。
- **Megatron**：`offload_megatron_model_to_cpu`，粒度更细（bf16 param / fp32 grad / fp32 main
  param / fp32 优化器状态分别搬运）；优化器还可用 `optimizer_cpu_offload` +
  `optimizer_offload_fraction` 部分 offload（见文档 15）。
- **ref model**：只读、forward_only，仅 `param_offload`。

## 二、Rollout（vLLM）offload：sleep / wake_up

vLLM 是推理引擎，**verl 不销毁/重建引擎**，而是用 vLLM 的 sleep / wake_up（release /
resume memory occupation）机制，由 `free_cache_engine=True`（默认）启用，并**分标签（tags）**
管理 `"weights"` 与 `"kv_cache"` 两块显存。

同步 colocate 场景生命周期（`engine_workers.py::update_weights`，mode="naive"）：

```
[训练刚结束，准备 rollout]
1. rollout.resume(tags=["weights"])    # 唤醒权重显存（恢复 VA 映射）
2. update_weights(per_tensor_param)    # 从 trainer 经 CUDA IPC 灌入最新权重
3. actor.engine.to("cpu")              # 训练权重 offload 让路
4. rollout.resume(tags=["kv_cache"])   # 唤醒 KV cache 显存
[generate_sequences ...]
[rollout 结束] rollout.release()       # sleep：释放 weights + kv_cache
```

`sleep_level`（`verl/third_party/vllm/__init__.py`）：
- vLLM ≥ 0.8.5 → **level 2**（默认）
- NPU / `layered_summon` / EP>1 且旧版本 → **level 1**

## 三、核心问题：offload 后 CUDA graph 要重新 capture 吗？——不需要

重新 capture 不现实（耗时且频繁）。sleep/wake 能与 cudagraph 共存，根源在于它基于
**CUDA 虚拟内存 API（CuMemAllocator，`cuMemCreate`/`cuMemMap`）的"物理换页"**，而非销毁重建：

1. CUDA graph capture 记录的是 **kernel 启动序列 + 固定的虚拟地址（VA）指针**，replay 时对这些
   固定 VA 读写。
2. sleep 时 CuMemAllocator **只释放物理显存页，保留虚拟地址预留（VA reservation）不动**。
3. wake_up 时重新申请物理页并 **映射回相同的虚拟地址**。
4. cudagraph 里的指针在 wake_up 后仍指向有效显存 → **无需重新 capture，直接 replay**。

一句话：**换的是物理内存，不动虚拟地址；cudagraph 绑定虚拟地址，故天然兼容。**

### level 1 vs level 2（都不需要重新 capture）

| | level 1 (offload) | level 2 (discard) |
|--|------------------|-------------------|
| weights 物理页 | 拷到 CPU pinned mem 保留 | 直接丢弃（不拷 CPU）|
| wake_up 后 weights | 从 CPU 拷回原 VA | 垃圾值，**必须 update_weights 重灌**|
| KV cache | 丢弃，wake 重新分配 | 丢弃，wake 重新分配 |
| VA 布局 & cudagraph | 保留 | 保留 |

RL 默认 level 2 的原因：每个训练 step 后 actor 权重都变，反正要从 trainer 重灌
（`load_format=dummy` 初始化也是此逻辑），没必要浪费 CPU 内存/带宽去 offload 旧权重。

### 执行顺序的深意

流程先 `resume(tags=["weights"])` **再** `update_weights`：先恢复权重显存的 VA 映射（分配物理页
+ 映射回 cudagraph 引用的同一地址），再把新权重数据拷进这些地址，从而保证 cudagraph 引用的
权重指针始终有效。

## 四、相关默认配置（`verl/trainer/config/rollout/rollout.yaml`）

```yaml
enforce_eager: False          # 默认启用 cudagraph（配合 sleep/wake 无冲突）
cudagraph_capture_sizes: null # 可指定 [1,2,4,8,...] 减少 capture 的 batch 尺寸省显存
free_cache_engine: True       # 默认启用 sleep/wake
load_format: dummy            # vLLM 初始化随机权重，真实权重由 trainer 灌入
layered_summon: False         # 大模型分层收集权重防 OOM（会强制 sleep_level=1）
```

兜底：若某模型 / vLLM 版本的 cudagraph 与 sleep 有兼容问题，可 `enforce_eager=True` 关闭
cudagraph。

## 五、速查结论

- 训练模型（actor/critic/ref）：`param_offload` / `optimizer_offload` / `grad_offload`，用时
  load、用完 offload；ref 仅 param。
- vLLM rollout：用 sleep/wake（非销毁重建），`free_cache_engine=True`，按 `weights` /
  `kv_cache` 两 tag 分别管理。
- CUDA graph 不需重新 capture：sleep/wake 基于 CuMemAllocator 物理换页、保留虚拟地址，cudagraph
  绑定虚拟地址故仍有效。
- 默认 sleep level 2（权重丢弃 + 重灌），level 1 仅在 NPU / layered_summon / 旧版本 EP 场景。

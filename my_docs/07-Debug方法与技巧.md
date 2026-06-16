# 07 · Debug 方法与技巧

> 返回索引：[README.md](README.md)

RL 训练的 bug 往往不是"崩溃"而是"训练不涨/数值漂移"，所以本文档重点在**可观测性 + 数值一致性**两条线。

## 7.1 内置工具（`verl/utils/debug/`）

### marked_timer —— 阶段计时
`marked_timer(name, timing_raw, color)`（底层 `simple_timer`）把各阶段耗时写进 `timing_raw` 字典，再经 `compute_timing_metrics` 上报。step 主流程里每个阶段都已包裹（gen/old_log_prob/ref/values/adv/update_actor…），日志/wandb 里直接看 `timing_s/*` 与 `timing_per_token_ms/*` 即可定位最慢阶段。

### calculate_debug_metrics —— rollout vs actor 数值一致性（最重要）
`metrics.py::calculate_debug_metrics`（debug/metrics.py:63）对比"rollout 引擎记录的 logprob"与"actor 重算的 logprob"，是定位 **rollout-train mismatch** 的核心。开 `rollout.calculate_log_probs=True` 后会输出：

| 指标 | 含义 | 健康参考 |
|------|------|----------|
| `training/rollout_probs_diff_max` | 最大概率差 | 越小越好 |
| `training/rollout_probs_diff_mean` | 平均概率差 | 接近 0 |
| `training/rollout_probs_diff_std` | 差异标准差 | 小且稳定 |
| `training/rollout_actor_probs_pearson_corr` | 皮尔逊相关系数（[arXiv:2506.13585](https://arxiv.org/pdf/2506.13585)） | 接近 1 |

实战判断：若 `mean/max` 明显偏大、`pearson_corr` 明显 < 1，说明 rollout 引擎（vLLM/SGLang，常 BF16）与训练侧（actor）的前向不一致——常见原因是精度（fp8/bf16）、kernel 实现差异、weight 同步未生效。这时优先用 **decoupled 模式**（重算 old_log_probs 做锚，见 [01 文档](01-核心运行流程.md) old_log_prob 两种模式）或排查权重同步。

### trajectory_tracker —— 轨迹级追踪
`trajectory_tracker.py` 跟踪单条轨迹的生成与得分，配合 dump 出的 JSONL 可还原"某个 prompt 到底生成了什么、拿了多少 reward"。

### profiler —— Nsight / torch / NPU
由 `global_profiler` 配置控制（profiler/config.py）：

```yaml
global_profiler:
  tool: nsight              # nsight(nsys) / torch / npu(mstx) / torch_memory
  steps: [1, 5, 10]         # 只在这些 global_step 采集
  profile_continuous_steps: false   # true 则连续区间采集
  save_path: outputs/profile
  global_tool_config:
    nsight: {discrete: false}        # discrete=true 则每个被装饰函数单独起停
```
`_start_profiling`/`_stop_profiling`（main_ppo_sync.py:1170）按 step 触发，对 actor_rollout / ref / critic 三个 WorkerGroup 分别 `start_profile(role=..., profile_step=...)`。`torch_memory` 工具会 dump 显存快照，专门排 OOM。

## 7.2 实用技巧（环境变量 / 配置 / 脚本）

```bash
# ---- 环境变量 ----
VERL_LOGGING_LEVEL=DEBUG       # 提升 verl 自身日志级别
VERL_AUTO_PADDING=TRUE         # 自动 padding，排查 DP 整除/碎片问题
RAY_DEDUP_LOGS=0               # 关 Ray 日志去重，看清每个 worker 的输出
CUDA_LAUNCH_BLOCKING=1         # 让 CUDA kernel 同步，定位真实报错栈
TORCH_NCCL_BLOCKING_WAIT=1     # NCCL 卡死时定位是哪个集合通信 hang
VERL_USE_EXTERNAL_MODULES=pkg  # 注入外部注册模块（见 06 文档）
```

```bash
# ---- 配置技巧（命令行 override）----
trainer.val_only=True                 # 只跑验证，快速验证 rollout/reward 链路
trainer.total_training_steps=2        # 跑 2 步冒烟
data.train_max_samples=64             # 小数据集快速迭代
trainer.n_gpus_per_node=1 trainer.nnodes=1   # 单卡复现，排除分布式
actor_rollout_ref.rollout.calculate_log_probs=True  # 打开 mismatch 指标
trainer.rollout_data_dir=/tmp/rollouts              # dump 生成结果(JSONL) 人工检查
trainer.validation_data_dir=/tmp/val
```

```bash
# ---- 工具脚本（scripts/）----
# 1. 看 resolve 后的完整 Hydra 配置（排查 override 是否生效）
python scripts/print_cfg.py --config-name=ppo_trainer
# 或用 generate_trainer_config.sh 生成完整 config 快照

# 2. 可视化检查 rollout 输出（TUI，需 textual==0.52.1）
python scripts/rollout_viewer.py /tmp/rollouts   # 传 rollout_data_dir，支持 mask 占位符

# 3. 环境/依赖诊断（torch/cuda/vllm/sglang 版本与可见设备）
python scripts/diagnose.py

# 4. 造一个随机权重小模型，秒级跑通全链路（不依赖大模型下载/显存）
python scripts/init_random_model.py \
    --hf_model_path Qwen/Qwen2.5-0.5B \
    --new_config_path my_tiny_config.json \
    --output_path /tmp/tiny_model
```

## 7.3 排查路径建议（从快到慢、从局部到全局）

1. **配置层**：先 `print_cfg.py` 看 resolve 后的完整 config（很多"行为不符预期"其实是 override 没生效或被默认值覆盖）；`validate_config` 会拦截非法组合。
2. **小模型 + 小数据冒烟**：`init_random_model.py` 造 tiny model + `data.train_max_samples=64` + `total_training_steps=2`，秒级跑通，验证链路不报错。
3. **单机单卡复现**：`n_gpus_per_node=1, nnodes=1` 排除分布式/通信干扰，定位是不是算法/数据问题。
4. **CPU 纯逻辑单测**：`pytest tests/**/test_*_on_cpu.py` 验证 DataProto/算法等不依赖 GPU 的逻辑。
5. **数值不一致（训练不涨/漂移）**：开 `calculate_log_probs`，看 `calculate_debug_metrics` 的 `rollout_probs_diff_*` 与 `pearson_corr`；偏大则查精度/权重同步，必要时切 decoupled 模式。
6. **训练 hang / NCCL 卡死**：`RAY_DEDUP_LOGS=0` + `TORCH_NCCL_BLOCKING_WAIT=1` 看是哪个 rank/哪个集合通信卡住；常见于 DP 不整除导致某 rank 没数据。
7. **OOM**：① 看 seqlen 均衡是否生效（5.1）；② 调 `max_token_len_per_gpu` / micro batch；③ 开 FSDP `param/optimizer/activation offload`；④ 确认 rollout `sleep`/`free_cache_engine` 生效（5.4）；⑤ 用 profiler `tool=torch_memory` dump 快照定位显存峰值。
8. **rollout 质量差（reward 上不去）**：用 `rollout_data_dir` dump JSONL + `rollout_viewer.py` 人工看生成内容，区分是"生成本身差"还是"reward 函数判错"。

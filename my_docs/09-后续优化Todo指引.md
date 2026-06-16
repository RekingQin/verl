# 09 · 后续优化 Todo 指引（结合业界进展）

> 返回索引：[README.md](README.md)
> 结合当前代码现状与 2025–2026 业界 RL-for-LLM 系统进展，给出可执行方向 + 简略方案设计。
> 方案仅为落地思路草图，具体需结合代码现状评估。

## 9.1 架构演进

- [ ] **完成 `main_ppo.py` → `main_ppo_sync.py` 迁移**（v0.8.0 将移除旧版）。
  - *方案*：先把 `ray_trainer.py` 中被双链路共享的算子（`apply_kl_penalty`/`compute_advantage`/`_balance_batch`）抽到独立模块，两边 import 同一份；再给旧链路加 deprecation warning；最后用 CI 跑通 sync 链路的等价测试（同 seed 下 metrics 对齐）后删除旧 `RayPPOTrainer`。风险点：recipe/ 下仍继承旧 trainer，需同步迁移。

- [ ] **全异步 / one-step-off-policy 主线化**（`experimental/fully_async_policy`、`one_step_off_policy` 合入主库）。
  - *方案*：复用现有 `ReplayBuffer` 的 `batch_size` 采样模式（已支持，main_ppo_sync.py:261）替代 `global_steps` 同步屏障，让 train 不必等齐整个 step 的 rollout；用版本号 tag（weight version）标记每条样本来自哪一代权重，训练侧按 IS 权重（已有 `rollout_is_weights`）做 off-policy 修正，容忍 1 步滞后。关键是把 `step()` 拆成"生产者(rollout)"与"消费者(train)"两个独立 asyncio 循环，经 TransferQueue 解耦（参考 AReaL/rllm-pipeline）。

- [ ] **Agentic RL 一等公民**：多轮/工具调用、partial rollout、router replay 纳入核心调度。
  - *方案*：`agent_loop` 已支持多 session/多输出（key=`{uid}_{sid}_{idx}`）；进一步把"工具调用结果"作为 observation token 标进 `response_mask=0`（GAE/loss 已会跳过），并在 `CheckpointEngineWithCache` 基础上让长程 agent 任务在权重更新时走"本地缓存续跑"（Laminar），避免长轨迹被打断。

## 9.2 性能 / 并行

- [ ] **扩大融合算子覆盖**：把 `linear_cross_entropy`（不物化 logits）思路推广到 MoE router/aux-loss、distillation top-k KL。
  - *方案*：复用 `entropy_from_logits_with_chunking` 的分块范式，对 distillation KL 写一个 chunked kernel：按 vocab 分块算 top-k KL，逐块累加，峰值显存从 `O(B·L·V)` 降到 `O(B·L·chunk)`。MoE aux-loss 同理在 router logits 上分块。

- [ ] **EP/PP 下的 seqlen 均衡**：当前 `_balance_batch` 主要面向 DP。
  - *方案*：现有 `get_seqlen_balanced_partitions`（Karmarkar-Karp）只按 DP rank 均分；扩展为两级——先按 DP 分组，组内再按 PP stage 的 micro-batch 做二次均衡（让每个 PP stage 的 bubble 一致）；EP 下额外考虑专家激活不均，可引入按"token→专家"路由统计的预估 workload。

- [ ] **权重同步零拷贝**：推广 nixl/mooncake/RDMA 为默认，评估超大 MoE 带宽瓶颈。
  - *方案*：在 `CheckpointEngineManager` 增加"按 backend 自动选型"逻辑（colocate→naive，分离+固定→nccl，弹性/异构→nixl）；对 DeepSeek-671B 级 MoE，按专家分片流式传输（专家维 all-gather + ring p2p），并用 `update_weights_bucket_megabytes` 调优 bucket 大小做计算/通信重叠。

- [ ] **batch-invariant 数值一致性**：集成 batch-invariant kernel，从根本消除 rollout-train mismatch。
  - *方案*：当前靠 `calculate_debug_metrics` 事后观测（07 文档）。引入 vexact 的 batch-invariant attention/RMSNorm/matmul kernel（rms_norm_implementation="triton" 已有钩子），保证 rollout（vLLM/SGLang）与 train 前向逐位一致 → 可关掉 decoupled 重算、省一次 forward。先在小模型上验证 `rollout_probs_diff_max < 1e-3`。

## 9.3 容错 / 弹性

- [ ] **弹性训练(elastic)**：当前进程死亡即 SIGABRT 退出（04 文档机制一），缺节点级弹性。
  - *方案*：把"崩溃→整体退出→外层重拉"升级为"崩溃→剔除死节点→动态调整 world_size→从最近 ckpt 恢复"。依赖三件事：① 权重同步用 nixl/mooncake（弹性高，可动态增删 replica，已有 `add_replicas`/`remove_replicas`）；② FSDP/Megatron 支持变 world_size 重切分 ckpt（resharding）；③ `ResourcePoolManager` 支持运行时重建 PlacementGroup。

- [ ] **异步 ckpt 全后端化**：FSDP 侧补齐 Megatron 已有的 `async_save`。
  - *方案*：Megatron 已用 `AsyncCallsQueue` FIFO + 延后写 tracker（04 文档）。FSDP 侧可在 `FSDPCheckpointManager` 用后台线程把 sharded state_dict 先拷到 pinned CPU memory（不阻塞训练），再异步落盘；统一两者的 tracker 原子更新语义（先写临时文件再 rename，保证 `latest_checkpointed_iteration.txt` 永远指向完整 ckpt）。

- [ ] **细粒度抢占恢复**：在 ESI 检测基础上做"步内"安全点。
  - *方案*：当前 `should_save_ckpt_esi`（04 文档）只在 step 边界判断。对长 step（大 batch/长序列），可在 micro-batch 累积梯度的间隙插入轻量安全点：保存"已完成 micro-batch 数 + 优化器状态"，抢占恢复时从该 micro-batch 续算，避免重跑整个 step。

## 9.4 可观测性 / 工程

- [ ] **统一 profiling 视图**：聚合 nsys/torch-profiler + marked_timer + MFU 到单一 trace。
  - *方案*：给 `marked_timer` 的各阶段打 nvtx range（profiler 已有 `mark_start_range`），让 nsys trace 与 `timing_raw` 字典共用同一套 stage name；再做一个聚合脚本把 rollout WorkerGroup 与 train WorkerGroup 的 trace 按时间轴对齐，直观看出 rollout/train 的重叠率（异步化的核心指标）。

- [ ] **mismatch 自动告警**：把 rollout vs actor logprob 差异做成阈值告警。
  - *方案*：在 `_compute_old_log_prob` 后读 `calculate_debug_metrics` 的 `rollout_probs_diff_mean`，超阈值（如 > 0.05）时打 warning + 可选 abort；CI 里加一个小模型 e2e 测试断言 `pearson_corr > 0.99`，防止精度/kernel 改动悄悄引入 mismatch。

- [ ] **配置可发现性**：`generate_trainer_config.sh` 生成的 `_generated_*.yaml` 与文档联动。
  - *方案*：写一个脚本扫描各 `@register_*` 注册表，自动生成"可选 adv_estimator / policy_loss / engine backend"清单与默认 yaml 片段，挂到文档；新算法接入时只需对照清单填 yaml。

## 9.5 算法

- [ ] **熵机制系列**（clip_cov/kl_cov）与 GSPO/CISPO 的统一基准与文档化。
  - *方案*：用同一 base config（同模型/数据/seed）跑各 policy_loss 变体，统一记录 entropy/kl/clipfrac/reward 曲线，沉淀为对比基准表，便于选型（这些都已在 `POLICY_LOSS_REGISTRY`，切换仅改 `loss_mode`）。

- [ ] **多模态 / Omni RL**：对齐 VeOmni（diffusion/omni-modal）后端，复用同一调度与数据面。
  - *方案*：TensorDict 已支持 `multi_modal_inputs`（02 文档），Engine 已有 `veomni` 后端；补齐多模态 reward（图文一致性打分）与多模态 rollout 的 partial/abort 语义，复用现有 TransferQueue + ReplayBuffer 数据面，无需新调度。

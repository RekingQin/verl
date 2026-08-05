# verl 框架源码深度解读（主题索引）

> 目标读者：后续需全面负责 verl 框架底层、调度、并行与架构优化的开发者。
> 解读基线：当前仓库 `feat/verl-learning` 分支（V1 PPO 主链路 `verl/trainer/main_ppo.py` + `verl/trainer/ppo/v1/`）。
> 说明：本系列按主题拆分为独立文档，每个主题可独立维护迭代。架构/流程图均使用 ASCII 字符绘制。
> 版本提示：代码已重构为 V1 trainer 体系（`main_ppo.py` + `trainer/ppo/v1/`），旧版 `main_ppo.py` 已移至 `main_ppo_v0.py`（deprecated，v0.9.0 移除），`main_ppo_sync.py` 已并入 V1 的 `PPOTrainerSync`。

## 主题文档

| 编号 | 主题 | 文档 |
|------|------|------|
| 00 | 新手代码阅读指南（目录地图 / 阅读路线）**← 新人从这里开始** | [00-新手代码阅读指南.md](00-新手代码阅读指南.md) |
| 01 | 核心运行流程（数据流 / 控制流 / 分布式架构） | [01-核心运行流程.md](01-核心运行流程.md) |
| 02 | 核心数据结构（数据 / 模型 / 通信） | [02-核心数据结构.md](02-核心数据结构.md) |
| 03 | 优化算法（各类 PO 算法实现细节） | [03-优化算法.md](03-优化算法.md) |
| 04 | 容错与恢复机制 | [04-容错与恢复机制.md](04-容错与恢复机制.md) |
| 05 | 性能优化设计 | [05-性能优化设计.md](05-性能优化设计.md) |
| 06 | 自定义扩展方式 | [06-自定义扩展方式.md](06-自定义扩展方式.md) |
| 07 | Debug 方法与技巧 | [07-Debug方法与技巧.md](07-Debug方法与技巧.md) |
| 08 | 算子库构成与原理 | [08-算子库构成与原理.md](08-算子库构成与原理.md) |
| 09 | 后续优化 Todo 指引（结合业界进展） | [09-后续优化Todo指引.md](09-后续优化Todo指引.md) |
| 10 | Ray 串联机制详解（进程编排 / 资源池 / 远程调用） | [10-Ray串联机制详解.md](10-Ray串联机制详解.md) |
| 11 | 算法配置与跨 Worker 数据流转（输入/计算/输出/流转） | [11-算法配置与跨Worker数据流转.md](11-算法配置与跨Worker数据流转.md) |
| 12 | decorator.py 设计与 dispatch 机制详解 | [12-decorator与dispatch机制详解.md](12-decorator与dispatch机制详解.md) |
| 13 | 异步训练方案对比（V1 sync / colocate_async / separate_async） | [13-异步训练方案对比.md](13-异步训练方案对比.md) |
| 14 | Ray 使用技法全景手册（全仓库 use-case / API cookbook） | [14-Ray使用技法全景手册.md](14-Ray使用技法全景手册.md) |
| 15 | 训练后端（FSDP/Megatron）与优化器配置 | [15-训练后端与优化器配置.md](15-训练后端与优化器配置.md) |
| 16 | 显存卸载与 vLLM 睡眠/唤醒机制（offload / sleep-wake / cudagraph） | [16-显存卸载与vLLM睡眠唤醒机制.md](16-显存卸载与vLLM睡眠唤醒机制.md) |

## 源码导航速查

- 启动/编排：`verl/trainer/main_ppo.py`（V1 主入口，`trainer.use_v1=true`）、`verl/trainer/main_ppo_v0.py`（legacy，deprecated）
- V1 trainer：`verl/trainer/ppo/v1/`（`trainer_base.py` 基类 + `trainer_sync.py` / `trainer_colocate_async.py` / `trainer_separate_async.py`）
- 算法：`verl/trainer/ppo/core_algos.py`、`metric_utils.py`、`rollout_corr_helper.py`、`ppo/v1/utils.py`
- 调度：`verl/single_controller/{base,ray}/`（`decorator.py` 看 dispatch）
- 数据：`verl/protocol.py`、`verl/utils/transferqueue_utils.py`、`verl/utils/tensordict_utils.py`
- 引擎：`verl/workers/engine/{fsdp,megatron,...}`、`verl/workers/engine_workers.py`
- 生成：`verl/workers/rollout/{vllm,sglang,trtllm}_rollout/`、`llm_server.py`
- 容错：`verl/utils/checkpoint/`、`verl/checkpoint_engine/`（含新增 `delta_checkpoint_engine.py`）
- 算子：`verl/utils/kernel/`、`verl/models/mcore/`、`verl/models/transformers/`

# verl 框架源码深度解读（主题索引）

> 目标读者：后续需全面负责 verl 框架底层、调度、并行与架构优化的开发者。
> 解读基线：当前仓库 `feat/verl-learning` 分支（同步 PPO 主链路 `verl/trainer/main_ppo_sync.py`）。
> 说明：本系列按主题拆分为独立文档，每个主题可独立维护迭代。架构/流程图均使用 ASCII 字符绘制。

## 主题文档

| 编号 | 主题 | 文档 |
|------|------|------|
| 01 | 核心运行流程（数据流 / 控制流 / 分布式架构） | [01-核心运行流程.md](01-核心运行流程.md) |
| 02 | 核心数据结构（数据 / 模型 / 通信） | [02-核心数据结构.md](02-核心数据结构.md) |
| 03 | 优化算法（各类 PO 算法实现细节） | [03-优化算法.md](03-优化算法.md) |
| 04 | 容错与恢复机制 | [04-容错与恢复机制.md](04-容错与恢复机制.md) |
| 05 | 性能优化设计 | [05-性能优化设计.md](05-性能优化设计.md) |
| 06 | 自定义扩展方式 | [06-自定义扩展方式.md](06-自定义扩展方式.md) |
| 07 | Debug 方法与技巧 | [07-Debug方法与技巧.md](07-Debug方法与技巧.md) |
| 08 | 算子库构成与原理 | [08-算子库构成与原理.md](08-算子库构成与原理.md) |
| 09 | 后续优化 Todo 指引（结合业界进展） | [09-后续优化Todo指引.md](09-后续优化Todo指引.md) |

## 源码导航速查

- 启动/编排：`verl/trainer/main_ppo_sync.py`、`verl/trainer/main_ppo.py`
- 算法：`verl/trainer/ppo/core_algos.py`、`metric_utils.py`、`rollout_corr_helper.py`
- 调度：`verl/single_controller/{base,ray}/`（`decorator.py` 看 dispatch）
- 数据：`verl/protocol.py`、`verl/utils/transferqueue_utils.py`
- 引擎：`verl/workers/engine/{fsdp,megatron,...}`、`verl/workers/engine_workers.py`
- 生成：`verl/workers/rollout/{vllm,sglang,trtllm}_rollout/`、`llm_server.py`
- 容错：`verl/utils/checkpoint/`、`verl/checkpoint_engine/`
- 算子：`verl/utils/kernel/`、`verl/models/mcore/`、`verl/models/transformers/`

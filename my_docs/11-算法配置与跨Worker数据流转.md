# 11 · 算法配置与跨 Worker 数据流转（输入/计算/输出/流转）

> 返回索引：[README.md](README.md)
> 关联阅读：[01-核心运行流程.md](01-核心运行流程.md)（step 控制流）、[03-优化算法.md](03-优化算法.md)（算法数学）、[10-Ray串联机制详解.md](10-Ray串联机制详解.md)（远程调用）
> 源码基线：`workers/engine_workers.py`、`workers/utils/losses.py`、`trainer/ppo/{core_algos,ray_trainer}.py`、`trainer/config/`

本篇回答四个问题：**一个 RL 算法需要配什么 → 每个阶段输入什么张量 → 在哪个 worker 算什么 → 输出什么 → 数据如何在 worker/engine 间接力。**

---

## 11.0 总览：一个训练 step 的"接力棒"

把训练数据想象成一根接力棒，沿下面的链路传递，每一站由不同 worker 往棒上**追加字段**：

```
 字段累积流（→ 表示该站新增的字段）

 Dataset/collate      → input_ids, attention_mask, position_ids, (raw_prompt, uid)
        │ generate_sequences (Rollout 引擎: vLLM/SGLang)
        ▼              → responses, response_mask, (rollout_log_probs)
 [Reward 阶段]         → token_level_scores   (reward_manager / reward model)
        │ compute_log_prob (Actor engine, 前向)
        ▼              → old_log_probs
 [Ref 阶段, 可选]      → ref_log_prob          (Ref engine, 前向)
        │ compute_values (Critic engine, 前向, 仅GAE)
        ▼              → values
 [Advantage 阶段]      → token_level_rewards, advantages, returns   (Driver, 纯CPU计算)
        │ update_critic (Critic engine, 前向+反向, 仅GAE)
        ▼              → 更新 critic 参数
        │ update_actor (Actor engine, 前向+反向)
        ▼              → 更新 actor 参数
 update_weights        → actor 新权重同步给 Rollout 引擎
```

关键认知：
- **生成、前向打分、训练分别落在不同 engine**：Rollout 引擎（推理）/ Actor·Ref·Critic 训练引擎（FSDP/Megatron）。
- **advantage 计算在 Driver 上做（纯标量/向量运算，无需 GPU 大模型）**，其余重计算都在 Worker 内。
- 每一站的输入/输出都是同一根 batch（经典版是 `DataProto`，同步版是 TransferQueue 里的 nested `TensorDict` + `KVBatchMeta`），靠**字段名**对齐。

---

## 11.1 配置：一个算法要配哪些开关（`trainer/config/`）

verl 把 RL 算法拆成"可插拔注册项 + YAML 开关"，**换算法基本只改 config，不改 Driver**。核心开关：

| 配置项 | 作用 | 典型值 |
|--------|------|--------|
| `algorithm.adv_estimator` | 选优势估计器（决定是否需要 critic） | `gae` / `grpo` / `rloo` / `remax` / `reinforce_plus_plus` |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | 选策略损失 | `vanilla` / `gspo` / `clip_cov` / `cispo` ... |
| `actor_rollout_ref.actor.use_kl_loss` + `kl_loss_coef` + `kl_loss_type` | KL-in-loss（B 位置） | `True` / `0.001` / `low_var_kl` |
| `algorithm.use_kl_in_reward` + `kl_ctrl` + `kl_penalty` | KL-in-reward（A 位置） | `True` / adaptive / `kl` |
| `actor_rollout_ref.actor.entropy_coeff` | 熵正则系数 | `0`（默认关） |
| `reward.reward_model.enable` | 是否用 reward model 打分 | `True/False` |
| `data.train_batch_size` / `actor.ppo_mini_batch_size` / `ppo_micro_batch_size_per_gpu` | 三级 batch | — |
| `rollout.n` | 每个 prompt 采样条数（GRPO 组大小） | `5` / `8` ... |

**算法差异的本质就两点**（见 [03](03-优化算法.md)）：
1. **要不要 critic**：`gae` 要（PPO 路线，多一个 Critic worker + values + returns + update_critic）；`grpo/rloo/remax/rf++` 不要（组内/baseline 归一）。
2. **policy_loss / KL 怎么算**：选不同注册函数。

### 注册表机制（加新算法只需注册）
```python
# core_algos.py
@register_adv_est("grpo")          # -> ADV_ESTIMATOR_REGISTRY
def compute_grpo_outcome_advantage(...): ...

@register_policy_loss("vanilla")   # -> POLICY_LOSS_REGISTRY
def compute_policy_loss_vanilla(...): ...
```
Driver/engine 通过 `get_adv_estimator_fn(name)` / `get_policy_loss_fn(loss_mode)` 按 config 字符串取函数。新增算法：写函数 + 注册 + 改 yaml 指向它即可，详见 [06-自定义扩展方式.md](06-自定义扩展方式.md)。

---

## 11.2 逐站详解：输入 → 计算 → 输出 → 落在哪个 worker

下面每一站给出 **Driver 调用 → Worker/engine 方法 → 输入字段 → 计算 → 输出字段**。worker 方法都在 `engine_workers.py`，统一经 §10.5 的 dispatch/collect 远程执行。

### 站 1：Rollout 生成
| 项 | 内容 |
|----|------|
| Driver 调用 | `actor_rollout_wg.generate_sequences(batch)` |
| 落点 | **Rollout 引擎**（vLLM/SGLang），非训练引擎 |
| 输入 | `input_ids, attention_mask, position_ids`（prompt），采样参数 `rollout.n / temperature / top_p` |
| 计算 | 自回归生成 `n` 条响应；多轮/工具走 AgentLoop |
| 输出 | `responses`, `response_mask`，可选 `rollout_log_probs`(π_rollout 的 logprob) |

### 站 2：Reward 打分
| 项 | 内容 |
|----|------|
| 落点 | `reward_manager`（规则函数，CPU/沙箱）或 reward model worker（GPU 前向） |
| 输入 | `responses` + `ground_truth`（数据集自带） |
| 计算 | `reward_score/` 内置打分（gsm8k/math/...）或模型打分 |
| 输出 | `token_level_scores`（通常只在 response 末 token 给标量分） |

### 站 3：old_log_prob（Actor 前向）
| 项 | 内容 |
|----|------|
| Driver 调用 | `actor_rollout_wg.compute_log_prob(batch)`（engine_workers.py:697） |
| 落点 | **Actor 训练引擎**，`self.actor.infer_batch(data)`（前向 no_grad） |
| 输入 | `input_ids, responses, response_mask`（full sequence） |
| 计算 | 用**当前 π_θ**（更新前）重算每个生成 token 的 logprob |
| 输出 | `old_log_probs`（PPO ratio 的分母锚） |

> 为何要重算而非直接用 `rollout_log_probs`？rollout 引擎与训练引擎数值/并行实现不同，重算保证 on-policy 一致；同步链路里还有 bypass/decoupled 两种模式（见 [01](01-核心运行流程.md) §1.4）。

### 站 4：ref_log_prob（Ref 前向，可选）
| 项 | 内容 |
|----|------|
| Driver 调用 | `ref_policy_wg.compute_ref_log_prob(batch)`（engine_workers.py:690） |
| 落点 | **Ref 引擎**（冻结的初始模型；LoRA 时即 actor 关 adapter） |
| 输入 | 同上 full sequence |
| 计算 | 用 π_ref 算 logprob |
| 输出 | `ref_log_prob`（供 KL 约束，A 或 B 位置） |
| 开关 | `use_kl_loss` 或 `use_kl_in_reward` 任一为真才需要（`need_reference_policy`） |

### 站 5：values（Critic 前向，仅 GAE）
| 项 | 内容 |
|----|------|
| Driver 调用 | `critic_wg.infer_batch(batch)`（TrainingWorker.infer_batch，engine_workers.py:392） |
| 落点 | **Critic 引擎** |
| 输入 | full sequence |
| 计算 | 逐 token 价值估计 |
| 输出 | `values` |
| 开关 | 仅 `adv_estimator=gae` 需要（`need_critic`）；GRPO 系不建 critic |

### 站 6：Advantage（Driver 上算，无大模型）
| 项 | 内容 |
|----|------|
| 落点 | **Driver**（v0 在 `ray_trainer.py`；V1 在 `trainer/ppo/v1/trainer_base.py::_compute_advantage`，纯 tensor 运算） |
| 输入 | `token_level_scores`, `old_log_probs`, `ref_log_prob`(A), `values`(GAE), `uid`(index) |
| 计算 | ① `apply_kl_penalty`(可选,A位置)：`token_level_rewards = token_level_scores - kl_coef*KL`；② `get_adv_estimator_fn(name)`：GAE 用 TD(λ)+values；GRPO 用组内 (r-μ)/σ |
| 输出 | `token_level_rewards`, `advantages`, `returns`(GAE) |

### 站 7：update_critic（Critic 训练，仅 GAE）
| 项 | 内容 |
|----|------|
| Driver 调用 | `critic_wg.train_mini_batch(batch)` |
| 落点 | Critic 引擎，loss=`value_loss`（losses.py:147） |
| 输入 | `values`, `returns`, `response_mask` |
| 计算 | clipped value loss `compute_value_loss`，前向+反向 |
| 输出 | 更新 critic 参数 + metrics |

### 站 8：update_actor（Actor 训练，核心）
| 项 | 内容 |
|----|------|
| Driver 调用 | `actor_rollout_wg.update_actor(batch)`（engine_workers.py:705 → `actor.train_mini_batch`） |
| 落点 | Actor 引擎，loss=`ppo_loss`（losses.py:57，`init_model` 时 `set_loss_fn` 注入） |
| 输入 | `old_log_probs`, `advantages`, `response_mask`，可选 `ref_log_prob`/`rollout_is_weights` |
| 计算 | 见下 §11.3，前向（算 `log_prob`/`entropy`）+ 组装 loss + 反向 |
| 输出 | 更新 actor 参数 + metrics（pg_loss/kl/entropy/grad_norm...） |

### 站 9：update_weights（权重回灌 Rollout）
`actor_rollout_wg.update_weights()`（engine_workers.py:720，async）：把刚更新的 actor 权重同步给 colocate 的 rollout 引擎（`naive` 模式同进程直传 / 否则走 checkpoint_engine）。下一个 step 的生成才用上新策略——这保证 on-policy。（V1 中由 `PPOTrainerSync.on_step_end` 的 `checkpoint_manager.update_weights()` 触发。）

---

## 11.3 actor loss 的组装（`ppo_loss`，losses.py:57）

这是算法"计算什么"的落地核心。一次 `update_actor` 在 engine 内：

```
engine.train_batch(data, loss_fn=ppo_loss):
   model_output = forward(data)            # 得到 log_prob, entropy(可选)
   ppo_loss(config, model_output, data):
     # 1. 取出 dispatch 来的字段
     old_log_prob, advantages, response_mask = data[...]
     # 2. 策略损失 (按 loss_mode 选注册函数)
     pg_loss, m = get_policy_loss_fn(loss_mode)(
         old_log_prob, log_prob, advantages, response_mask, config, rollout_is_weights)
     policy_loss = pg_loss
     # 3. 熵正则 (entropy_coeff>0 时, 减号=最大化熵)
     if entropy is not None:
         policy_loss -= entropy_coeff * agg_loss(entropy, response_mask, ...)
     # 4. KL-in-loss (use_kl_loss 时, 加号约束偏离 ref)
     if config.use_kl_loss:
         kld = kl_penalty(log_prob, ref_log_prob, kl_loss_type)
         policy_loss += kl_loss_coef * agg_loss(kld, response_mask, ...)
   loss.backward()                         # NCCL 反向 all-reduce
```

对应公式：
```
loss = pg_loss(clip)  - entropy_coeff * H(π)  + kl_loss_coef * KL(π‖π_ref)
       ↑PPO主项         ↑鼓励探索(默认关)        ↑稳定约束(GRPO配方常开)
```

vanilla PPO 的 `pg_loss`：`ratio = exp(log_prob - old_log_prob)`，`-min(ratio*A, clip(ratio,1±ε)*A)`，按 `loss_agg_mode` 聚合。算法差异（gspo/cispo/...）只换 `pg_loss` 的算法，外层组装不变。

---

## 11.4 这些字段如何"跨 worker"传（两种数据面）

| | v0 经典版 `main_ppo_v0.py` | V1 主链路 `main_ppo.py`（ppo/v1/） |
|---|---|---|
| 载体 | `DataProto`（batch 张量 + meta_info） | TransferQueue 里 nested `TensorDict` + `KVBatchMeta`(元数据) |
| 切分 | dispatch 时 `chunk(world_size)`，collect 时 `concat` | dispatch 切的是 key 句柄，张量留在 TQ |
| 张量是否经 Driver | 是（Driver 持有整批 DataProto） | 否（Driver 只过 keys/tags，零拷贝） |
| 字段对齐 | 靠 DataProto 的 batch key 名 | 靠 TQ 的 field 名 |

两种方式都遵循同一抽象：**每个计算阶段 = 取所需字段 → 在 Worker 内计算 → 写回新字段**。同步版把"取/写"换成 `tq.kv_batch_get/put`，张量不经 Driver（见 [01](01-核心运行流程.md) §1.5）。`compute_log_prob` 等方法之所以能无缝切到 TQ，是因为 `@register` 内部套了 `tqbridge`（decorator.py:425）。

---

## 11.5 速记表：算法选型对数据流的影响

| 配置 | 是否建 Critic | 多出的站 | 多出的字段 |
|------|--------------|----------|-----------|
| `adv_estimator=gae`（PPO） | ✅ | 站5 values + 站7 update_critic | `values`, `returns` |
| `adv_estimator=grpo/rloo/dr.grpo` | ❌ | 无 | 仅 `advantages`（组内归一） |
| `adv_estimator=remax` | ❌ | 额外一次 greedy 生成做 baseline | greedy reward |
| `use_kl_in_reward=True`（A） | — | advantage 前 `apply_kl_penalty` | 需 `ref_log_prob` |
| `use_kl_loss=True`（B） | — | 无额外站 | actor loss 里用 `ref_log_prob` |
| `reward_model.enable=True` | — | 站2 改为模型打分 | `token_level_scores` |

一句话收尾：**配置决定"建哪些 worker、走哪些站、传哪些字段"；Driver 的编排代码对所有算法基本不变，差异被收敛进注册表函数与 YAML 开关。** 这正是 HybridFlow"算法与系统解耦"的价值。

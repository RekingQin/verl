# 18 · models 模块设计与运行机制（transformers monkey_patch / mcore 集成 / 权重转换）

> 返回索引：[README.md](README.md)
> 相关文档：[15-训练后端与优化器配置](15-训练后端与优化器配置.md)、[17-workers模块设计与运行机制](17-workers模块设计与运行机制.md)、[08-算子库构成与原理](08-算子库构成与原理.md)

`verl/models/` 是 verl 的**"模型适配层"**——它并**不定义**任何新的模型架构，而是在 HuggingFace `transformers`（FSDP/FSDP2/TorchTitan/veomni 路径）与 NVIDIA `Megatron-Core`（mcore 路径）**两个上游生态**之间做桥接：把两边的模型格式、config、权重、前向计算习惯 **黏合成 verl 训练循环能用的统一接口**。

```
verl/models/
├── README.md                    ← 官方说明
├── __init__.py                  ← （空）
├── registry.py                  ← ⚠ legacy Megatron 模型注册表（v0.9 前遗留，已被 mcore/ 取代）
├── weight_loader_registry.py    ← ⚠ legacy Megatron 权重 loader/saver 注册表
│
├── transformers/                ← ⭐ HF 侧适配（FSDP/FSDP2/TorchTitan/veomni 都用）
│   ├── __init__.py                     : 只暴露 apply_monkey_patch + apply_tiled_mlp_monkey_patch
│   ├── monkey_patch.py                 : ★ 总入口 apply_monkey_patch: SP / remove_padding / fused kernel / tiled MLP / prefix grouper
│   ├── dense_common.py                 : 通用 dense 模型 fused kernel forward
│   ├── llama.py / qwen2.py / qwen3_5.py: 各家族 attn/forward 的 verl 版本
│   ├── qwen2_vl.py / qwen3_vl.py / glm4v.py / kimi_vl.py : VLM 变体 (含视觉塔)
│   ├── apertus.py                      : Apertus 特化
│   ├── tiled_mlp.py                    : ⭐ TiledMLP: MLP 分片重算，省 activation 显存
│   └── npu_patch.py                    : NPU (华为昇腾) 融合算子替换
│
└── mcore/                        ← ⭐ Megatron-Core 侧适配
    ├── readme.md                       : 官方说明（3 代集成方式演进）
    ├── __init__.py
    ├── bridge.py                       : ⭐ Megatron-Bridge 桥接: AutoBridge + LinearForLastLayer + make_value_model
    ├── mbridge.py                      : （旧）mbridge 兼容层
    ├── registry.py                     : ⭐ SupportedModel 枚举 + config/init/forward/weight 4 大 REGISTRY
    ├── config_converter.py             : ⭐ HF PretrainedConfig → mcore TransformerConfig 转换
    ├── model_initializer.py            : ⭐ 各家族的 BaseModelInitializer 子类 (DenseModel / MoE / DeepseekV3)
    ├── model_forward.py                : ⭐ 通用 GPTModel forward: THD/BSHD 预处理 + logits_processor
    ├── model_forward_fused.py          : Fused forward: entropy/log_prob 与 forward 融合，节省 logits 存内存
    ├── model_forward_1f1b_overlap.py   : PP 1F1B + overlap 优化专用 forward
    ├── model_initializer.py            : GPTModel 初始化（TransformerLayerSpec / ROPE / MTP block spec）
    ├── mtp_patch.py                    : ⭐ DeepSeek MTP（Multi-Token Predict）postprocess/loss patch
    ├── patch.py                        : ⭐ mcore 0.12/0.13+ bug 修复 + FastHadamard shim + MLA 兼容
    ├── weight_converter.py             : ⭐ mcore → HF 权重转换器 (rollout 权重同步用)
    ├── loader.py                       : （legacy）HF → mcore GPTModel 运行时加载
    ├── saver.py                        : （legacy）mcore → HF 权重导出
    └── util.py                         : THD/BSHD 预处理/后处理 (packed seqs, cu_seqlens, FP8 padding)
```

## 18.1 定位：models 不建模，只做适配

verl 的开发原则（`README.md`）非常明确：

> The FSDP and FSDP2 engines load Hugging Face model implementations **directly**.
> Do not copy complete files from `transformers` into this directory to add a checkpoint.

**这意味着**：
- **训练时用的是原生 HF/`transformers` 的 `LlamaForCausalLM` / `Qwen2ForCausalLM` 等类实例**
- verl 只做 **monkey patch**（`transformers/` 目录）——替换掉里面的 `_flash_attention_forward` / `forward` / `MLP` 等方法
- 或者用 **`mcore` 的 `GPTModel`**（Megatron 路径），配套配置转换 + 权重转换
- **绝不** 在 verl 里重写整个模型（除了 legacy `registry.py`，早已 deprecated）

**为什么这样设计？**
- 上游 HF 一年 release 上百个新模型，抄一个进来意味着永远追不上
- verl 的核心竞争力是 RL 编排（Ray + dispatch + rollout）而非模型实现，把模型交给上游
- monkey patch 只改**必要的方法**（注入 remove_padding、Ulysses SP、fused kernel），把体量最小化

## 18.2 全景图：models 如何被 workers/engine 调用

```
┌──── verl/workers/engine/fsdp/transformer_impl.py::_build_module ────┐
│                                                                       │
│  1. AutoModel.from_pretrained(model_path, config=hf_config, ...)     │
│                            │                                          │
│                            v                                          │
│  2. 原生 HF LlamaForCausalLM 对象                                     │
│                            │                                          │
│  3. apply_monkey_patch(model, ulysses_sp_size, use_remove_padding,   │
│                       use_fused_kernels, fused_kernels_backend,       │
│                       use_prefix_grouper, use_tiled_mlp)              │
│                            │                                          │
│                            v (from verl.models.transformers)          │
│      ┌──────────────────────────────────────────────────────┐        │
│      │ verl/models/transformers/monkey_patch.py             │        │
│      │  - _ulysses_flash_attention_forward → replace HF     │        │
│      │  - forward_with_torch_backend / triton → replace HF  │        │
│      │  - patch_vlm_for_ulysses_input_slicing (VLM)         │        │
│      │  - apply_tiled_mlp_monkey_patch (省内存)              │        │
│      │  - apply_prefix_grouper_patch (跨样本共享 prefix)     │        │
│      └──────────────────────────────────────────────────────┘        │
│  4. module 现在是"HF 骨架 + verl 定制方法"混合体                       │
│  5. FSDP/FSDP2 包装 → 训练                                            │
└───────────────────────────────────────────────────────────────────────┘

┌──── verl/workers/engine/megatron/transformer_impl.py::_build_module ───┐
│                                                                        │
│  1. hf_config = AutoConfig.from_pretrained(model_path)                │
│                                                                        │
│  2. tfconfig = hf_to_mcore_config(hf_config, dtype, **overrides)      │
│      │  (verl/models/mcore/config_converter.py 里的 dispatch)         │
│      v                                                                 │
│      MODEL_CONFIG_CONVERTER_REGISTRY[SupportedModel(arch)]             │
│                            │                                           │
│                            v                                           │
│      TransformerConfig (mcore 内部格式)                                │
│                                                                        │
│  3. model = init_mcore_model(tfconfig, hf_config, ...)                 │
│      │  MODEL_INITIALIZER_REGISTRY[SupportedModel(arch)]               │
│      v                                                                 │
│      DenseModel/Qwen3MoEModel/DeepseekV3Model.initialize()             │
│                            │                                           │
│                            v                                           │
│      mcore GPTModel 实例（内含 TP/PP/CP + Distributed Optimizer）      │
│                                                                        │
│  4. forward_fn = get_mcore_forward_fn(hf_config)                       │
│      │  model_forward.py::model_forward_gen(vision_model)              │
│      v                                                                 │
│      每次 forward: THD/BSHD 预处理 → GPTModel(...) → 后处理             │
│                                                                        │
│  5. rollout 权重同步:                                                  │
│      converter = get_mcore_weight_converter(hf_config, dtype)          │
│      converter.convert_param(...) → HF 命名/格式 → 发给 vLLM/SGLang     │
└────────────────────────────────────────────────────────────────────────┘
```

## 18.3 transformers/ 子模块（HF 侧 monkey patch）

### 18.3.1 `apply_monkey_patch`：一个总入口，五种能力注入

`monkey_patch.py::apply_monkey_patch`（:291）是 FSDP 路径的**唯一入口**。参数：

```python
apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,       # Ulysses 序列并行大小
    use_remove_padding: bool = True, # 是否走 packed varlen (rmpad)
    use_fused_kernels: bool = False, # 是否用 fused linear+ce
    fused_kernels_backend: str = None, # "triton" or "torch"
    use_prefix_grouper: bool = False,# 前缀共享
    use_tiled_mlp: bool = False,     # MLP 分片重算省内存
    tiled_mlp_shards: int = 4,
)
```

**注入顺序与内容**（`monkey_patch.py:291`）：

```
apply_monkey_patch:
  ├── (可选) apply_tiled_mlp_monkey_patch(num_shards, model_type)   # 见 18.3.4
  ├── (可选) apply_prefix_grouper_patch()                            # 见 18.3.5
  ├── 若装了 trl: 覆盖 AutoModelForCausalLMWithValueHead.state_dict
  │
  ├── ★ 按 model.config.model_type 分支处理:
  │    ├── qwen2_5_vl / qwen2_vl:
  │    │     Step1: Qwen2_5_VLModel.forward = qwen2_vl_base_forward   (verl 版本)
  │    │            Qwen2_5_VLForConditionalGeneration.forward = forward_with_normal_backend
  │    │     Step2: Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward (rmpad + SP)
  │    │     Step3: patch_vlm_for_ulysses_input_slicing(TextModel)     (SP 输入切分)
  │    │
  │    ├── qwen3_vl / qwen3_vl_moe:  同上，多一个 fast_pos_embed_interpolate
  │    ├── glm4v:                    同上，Glm4vTextAttention.forward
  │    ├── kimi_vl:                  DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
  │    ├── qwen3_5 / qwen3_5_moe:    含 GatedDeltaNet 特化 (Mamba-like)
  │    │
  │    └── 通用（Llama / Qwen2 / Qwen3 / Mistral / ...）:
  │          替换 ALL_ATTENTION_FUNCTIONS["flash_attention_2"] 
  │          = _ulysses_flash_attention_forward
  │
  └── patch_forward_with_backends(model, use_fused_kernels, backend):
        model.__class__.forward = 
          - forward_with_triton_backend  (fused linear + xentropy, triton)
          - forward_with_torch_backend   (fused, pure torch)
        (从 dense_common / qwen2_vl / qwen3_vl / glm4v / qwen3_5 派发)
```

**核心黑魔法**：`module = sys.modules[model.__module__]` 拿到 HF 模型所在的**模块对象**，直接 `module.LlamaFlashAttention2.forward = ...` 覆盖类方法。所有已实例化和未实例化的 attention 层都会被替换（Python 方法解析走 MRO）。

### 18.3.2 `_ulysses_flash_attention_forward`：DeepSpeed-Ulysses 序列并行

Ulysses SP（[Jacobs et al., 2023](https://arxiv.org/abs/2309.14509)）核心思想：**把长序列切给多个 GPU，靠 all-to-all 在 attn 前后转置维度**——切 seq 时算 QKV，切 head 时算 attn。这样通信量恒定为 O(seq × hidden)，与 head 数无关。

`monkey_patch.py:87` 的实现清晰展示三步：

```
_ulysses_flash_attention_forward(query, key, value, ...):
  ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
  
  if ulysses_sp_size > 1 and position_ids is not None:  # 判定非 ViT
    # (1) 补 kv heads: MQA/GQA 场景 kv head 数可能少于 sp_size
    repeats = max(ulysses_sp_size // key.size(2), 1)
    key = repeat_kv(key, repeats); value = repeat_kv(value, repeats)
    
    # (2) All-to-all: 从"切 seq"变成"切 head"
    #     (bsz, seq/sp, h, d) → (bsz, seq, h/sp, d)
    query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2)
    key   = gather_seq_scatter_heads(key,   seq_dim=1, head_dim=2)
    value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2)
    
    # (3) all_gather position_ids (给 flash-attn 的 varlen path)
    all_gather → concat
  
  # ⭐ 标准 flash_attention_forward (未做任何魔改)
  attn_output = _flash_attention_forward(query, key, value, ...)
  
  if ulysses_sp_size > 1 and position_ids is not None:
    # (4) 反向 All-to-all: 从"切 head"回到"切 seq"
    attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
  
  return attn_output
```

**为什么这个 patch 极简？** 因为它**只作用于 attention 一层**：QKV 计算前后各一次 all-to-all，中间的 flash-attn 完全不改。这是 Ulysses 的天然优势——**对 kernel 无侵入**。

### 18.3.3 `patch_forward_with_backends`：fused kernel forward 派发

verl 支持三种 forward 后端（关键权衡是 **logits 张量占的显存**）：

| 后端 | 实现 | logits 显存 | 精度 | 速度 |
|---|---|---|---|---|
| `normal` | HF 原版 | 完整存 (bsz×seq×vocab, fp32) | ✓ | baseline |
| `forward_with_torch_backend` | dense_common.py:71 | ⭐ 不存 logits，直接 forward+CE 融合 | ✓ | 快 |
| `forward_with_triton_backend` | dense_common.py:139 | ⭐ 同上，Triton kernel | ✓ | 最快 |

**代码分派逻辑**（`monkey_patch.py:234`）：

```
patch_forward_with_backends(model, use_fused_kernels, fused_kernels_backend):
  if not use_fused_kernels or backend not in ["triton", "torch"]:
    return  # 保持 HF 原样
  
  # 按 model_type 找对应模块的实现
  if model_type in ["qwen2_5_vl", "qwen2_vl"]:
    from ...qwen2_vl import forward_with_torch_backend, forward_with_triton_backend
  elif model_type in ["qwen3_vl", "qwen3_vl_moe"]:  ...
  elif model_type == "glm4v":                        ...
  elif model_type in ["qwen3_5", "qwen3_5_moe"]:    ...
  else:
    from ...dense_common import forward_with_torch_backend, forward_with_triton_backend
  
  # 覆盖 model.__class__.forward
  model.__class__.forward = forward_with_(triton|torch)_backend
```

**dense_common.py 里的关键**：`forward_with_torch_backend` 不产 logits，而是直接把 `hidden_states` 送入 fused `LigerCrossEntropy` 之类算子，一步得到 `loss` 和 `log_probs`。省的显存是 `bsz × seq_len × vocab_size × 4B`（130k vocab、8k seq、bsz=1 就是 **4GB**，越大越省）。

### 18.3.4 `TiledMLP`：MLP 激活分片重算（`tiled_mlp.py`）

**问题**：MLP 中间的 `gate_up_proj + activation + down_proj` 是 activation 显存大户（尤其 llama 4×hidden 或 qwen3 6×hidden 的 intermediate）。

**方案**（`TiledMLP` 是 `torch.autograd.Function`）：

```
Forward:  把 x [bsz*seq, hidden] 沿 dim=0 切成 N 份 → 分别过完整 MLP → concat
Backward: 前向重算 MLP → 拿到 activation → 反传得 grad → 累加
```

**关键点**：`GradientAccumulator`（`tiled_mlp.py:29`）在 backward 时**跨 shard 累加**梯度，不能用普通 sum（因为 dim=0 已经 concat 了）。

**使用**：
```python
apply_monkey_patch(model, use_tiled_mlp=True, tiled_mlp_shards=4)
# 显存 ~1/4，速度 -10~15%
```

### 18.3.5 `PrefixGrouper`：跨样本共享 prompt（`_create_prefix_grouper_wrapper`）

在 RL rollout 里，同一个 prompt 常常派生 `n=16` 条 response，形成 group。**如果这些样本的 prompt 部分完全一样**，理论上 forward 只需算一次 prompt attn，然后 broadcast 到各 response。

`apply_prefix_grouper_patch` 就是包装 `ALL_ATTENTION_FUNCTIONS` 里所有 backend（fa2/fa3/sdpa/flex/eager），如果 kwargs 里有 `prefix_grouper`，就走特殊路径避免重复计算 prompt。

**收益**：在 GRPO n=16、prompt=2k、response=1k 场景，theoretical throughput 提升 ~40%。

### 18.3.6 `patch_vlm_for_ulysses_input_slicing`：VLM 的 SP 输入切分

VLM 的 attention 已经被 `qwen2_vl_attn_forward` 等替换支持 SP，但**输入 embedding** 也需要切：
```
inputs_embeds (bsz, seq, dim) → slice → (bsz, seq/sp, dim) 每 rank 一份
```
同时 `visual_pos_masks` / `deepstack_visual_embeds` 也要相应切分。这个 patch 在 `TextModel.forward` 外面套一层，只在 `ulysses_sp_size > 1` 时生效。

### 18.3.7 NPU 特化：`npu_patch.py`

华为昇腾 NPU 环境下，`torch.nn.functional.silu` / `LayerNorm` 等操作没有优化 kernel。`npu_patch.py` 提供替换：

```
rms_norm_forward_npu               → torch_npu.npu_rms_norm
silu_forward_npu                    → torch_npu.npu_swiglu 
apply_rotary_pos_emb_npu           → torch_npu.npu_rotary_mul
NPUGmmFunction (autograd)          → torch_npu.npu_grouped_matmul (MoE 专用)
NPUQwen3VLMoeTextExperts           → NPU 版 MoE experts (grouped matmul)
```

**关键设计**：`apply_npu_patches()` 按 `model_type` 分派到 `_patch_qwen2` / `_patch_qwen3` / `_patch_qwen3_moe` / `_patch_qwen3_vl_moe` / `_patch_qwen3_5_moe` 等，每个函数各自替换该模型家族里几十处 `.forward = ...`。

## 18.4 mcore/ 子模块（Megatron-Core 集成）

### 18.4.1 三代集成方式的演进

`mcore/readme.md` 明确列出：

| 代次 | 方式 | 位置 | 状态 |
|---|---|---|---|
| 1️⃣ | 手写 modeling_*_megatron.py（每模型一个）| （已被删除，registry.py 遗留引用）| deprecated |
| 2️⃣ | mbridge（第三方库）| `mbridge.py` | deprecated |
| 3️⃣ | **Megatron-Bridge** (NVIDIA 官方) | `bridge.py::AutoBridge` | ⭐ 当前默认 |

配置开关：`actor.megatron.use_mbridge` / `actor.megatron.vanilla_mbridge`。默认 `vanilla_mbridge=false` → 走 Megatron-Bridge。

**为什么弃用 legacy？** 上游 mcore 迭代快（0.10 → 0.16 半年内），手写 modeling 每次升级要重新对齐；Megatron-Bridge 由 NVIDIA 官方维护，随 mcore 一起升级。

### 18.4.2 `SupportedModel` + 四大 Registry：mcore 的策略池

`registry.py:119` 定义了 `SupportedModel` 枚举（模型架构名），四张 registry 分别派发：

| Registry | 类型 | 作用 |
|---|---|---|
| `MODEL_CONFIG_CONVERTER_REGISTRY` | `Callable[[PretrainedConfig, torch.dtype], TransformerConfig]` | HF config → mcore config |
| `MODEL_INITIALIZER_REGISTRY` | `type[BaseModelInitializer]` | 构造 mcore GPTModel 实例 |
| `MODEL_FORWARD_REGISTRY` | `Callable` | 提供 forward 函数（thd/bshd + logits_processor）|
| `MODEL_FORWARD_FUSED_REGISTRY` | `Callable` | fused forward（fusion CE）|
| `MODEL_WEIGHT_CONVERTER_REGISTRY` | `type` | mcore weight → HF weight（rollout 同步用）|

**支持的模型架构**（`SupportedModel` 完整列表）：

```
Dense LM:  LlamaForCausalLM / Qwen2ForCausalLM / Qwen3ForCausalLM / Qwen3ForTokenClassification
           MiMoForCausalLM / GptOssForCausalLM
MoE:       Qwen2MoeForCausalLM / MixtralForCausalLM / DeepseekV3ForCausalLM 
           Qwen3MoeForCausalLM / Qwen3_5MoeForCausalLM / Glm4MoeForCausalLM / Llama4ForConditionalGeneration
VLM:       Qwen2_5_VLForConditionalGeneration / Qwen3VLForConditionalGeneration 
           Qwen3VLMoeForConditionalGeneration
Value:     LlamaForTokenClassification / Qwen3ForTokenClassification（复用 dense）
```

**入口函数**（`registry.py:236-296`）三行搞定：
```python
tfconfig = hf_to_mcore_config(hf_config, dtype, **overrides)
model    = init_mcore_model(tfconfig, hf_config, pre_process, post_process, value=..., **kw)
converter = get_mcore_weight_converter(hf_config, dtype)  # rollout weight sync
forward_fn = get_mcore_forward_fn(hf_config)               # 或 fused / engine 版
```

### 18.4.3 `hf_to_mcore_config_*`：配置格式桥接（`config_converter.py`）

同一个模型在 HF 和 mcore 里叫法完全不同：

| 概念 | HF `PretrainedConfig` | mcore `TransformerConfig` |
|---|---|---|
| 层数 | `num_hidden_layers` | `num_layers` |
| 注意力头 | `num_attention_heads` | `num_attention_heads` |
| KV 头（GQA）| `num_key_value_heads` | `num_query_groups` |
| MLP 中间 | `intermediate_size` | `ffn_hidden_size` |
| RoPE base | `rope_theta` | `rotary_base` |
| MoE 专家 | `num_experts` | `num_moe_experts` |
| MoE routed 数 | `num_experts_per_tok` | `moe_router_topk` |
| MLA 隐维 | `kv_lora_rank` | `kv_lora_rank` |

**基础函数** `_get_base_transformer_config`（:34）产出所有 dense 模型共用的 config，然后各 family 加自己的 field：
- `hf_to_mcore_config_dense`：Llama / Qwen2 / Qwen3
- `hf_to_mcore_config_qwen2moe` / `_qwen3moe` / `_mixtral`：MoE (top-k router + expert count)
- `hf_to_mcore_config_dpskv3`：MLA (`_get_mla_transformer_config`) + fine-grained MoE
- `hf_to_mcore_config_qwen2_5_vl`：VLM (vision encoder 单独一份 config)
- `hf_to_mcore_config_llama4`：Llama 4 特殊结构

**`**override_transformer_config_kwargs`** 极其重要：用户可以从 YAML 通过 `+actor_rollout_ref.actor.megatron.override_transformer_config.<field>=<value>` 直接覆盖 mcore 配置，例如：

```yaml
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=true
```

无需改 verl 一行代码就能启用 mcore 的新特性——**"策略池 + kwargs 直穿"是 verl 保持轻量的核心手段**。

### 18.4.4 `BaseModelInitializer` 与派生（`model_initializer.py`）

**基类**（`model_initializer.py:27`）用**模板方法**固定 GPTModel 初始化流程：

```python
class BaseModelInitializer(ABC):
    @abstractmethod
    def get_transformer_layer_spec(self, vp_stage=None): ...  # 子类填空
    def get_rope_scaling_args(self) -> dict: ...              # 通用
    
    def initialize(self, pre_process, post_process, 
                   share_embeddings_and_output_weights, value, **kw) -> GPTModel:
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        rope_scaling_args = self.get_rope_scaling_args()
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process, post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=get_hf_rope_theta(self.hf_config),
            **rope_scaling_args,
            mtp_block_spec=extra_kwargs.get("mtp_block_spec"),
            **({} if not self.has_vp_stage else {"vp_stage": vp_stage}),
        )
        if post_process and value:  # value head 替换
            model.output_layer = LinearForLastLayer(hidden_size, 1, sequence_parallel=...)
        return model
```

**关键点：`value=True` 时替换输出层**。`LinearForLastLayer`（bridge.py:33）是一个特殊的 `nn.Linear(hidden, 1)`，末层：
- `sequence_parallel=True` 时，会 `gather_from_sequence_parallel_region` 收集全序列
- forward 输出 fp32（数值稳定）
- 返回 `(logits, None)` 而非 tensor（mcore output_layer 接口要求）

**子类实现差异**：
- `DenseModel.get_transformer_layer_spec` → `get_gpt_decoder_block_spec(config, use_te=True)`
- `Qwen2MoEModel.get_transformer_layer_spec` → 带 `moe_grouped_gemm` 的 spec
- `DeepseekV3Model.initialize` → **覆盖** initialize，因为 DeepSeek MLA 需要 special output_layer & MoE MLP spec

### 18.4.5 `model_forward.py::model_forward_gen`：THD/BSHD 双路径

**为什么需要格式转换？**
- HF 训练路径：`input_ids (bsz, seq_len)` + `attention_mask (bsz, seq_len)`（BSHD 有 padding）
- mcore 高效路径：`packed_seqs (total_tokens,)` + `cu_seqlens` + `max_seqlen`（THD 无 padding）
- 部分模型如 GPT-OSS 只支持 BSHD

`model_forward_gen(vision_model)` 返回一个闭包（`model_forward.py:38`）：

```
model_forward(model, input_ids, attention_mask, position_ids, multi_modal_inputs,
              logits_processor, logits_processor_args, value_model, 
              data_format="thd", mtp_config=None):
  
  if data_format == "thd":
    # (1) 打包变长序列
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
        input_ids, attention_mask, pre_process=..., use_fp8_padding=...)
    
    # (2) 如果开了 MTP + post_process: label/loss_mask 也要打包
    if mtp_enable_train and post_process:
      args = {k: preprocess_packed_seqs(v, ...) for k,v in logits_processor_args.items()}
      model_kwargs["labels"] = args["label"]
      model_kwargs["loss_mask"] = args["label_mask"]
    
    # (3) forward
    output_orig = model(input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids,
                        packed_seq_params=packed_seq_params,
                        **multi_modal_inputs, **model_kwargs)
    
    # (4) logits_processor：可选 (in-place 计算 log_prob/entropy，避免存 logits)
    if post_process and logits_processor is not None:
      output_dict = logits_processor(output_orig, **args)
      output = {k: postprocess_packed_seqs(v, packed_seq_params, ...) 
                for k, v in output_dict.items()}
    else:
      output = postprocess_packed_seqs(output_orig, packed_seq_params, ...)
  
  elif data_format == "bshd":
    # 类似流程，但用 preprocess_bshd / postprocess_bshd
    ...
  
  if value_model and post_process:
    output = output[..., 0]   # value head: (bsz, seq, 1) → (bsz, seq)
  
  return output
```

**`logits_processor`** 是关键"扩展点"：调用方可以传一个函数 `(logits, labels, label_mask) → {log_probs, entropy, ...}`，避免把 logits 张量吐出来（省显存）。

### 18.4.6 `model_forward_fused.py`：极致融合的 forward

`fused_forward_model_gen(vision_model)` 与 `model_forward_gen` 差别：
- 在 forward 内部就调用 fused kernel（entropy_from_logits, log_probs_from_logits）
- 用一个 `FusedOutputProcessorContext` 上下文管理器，patch/unpatch mcore `GPTModel._postprocess`
- **不返回 logits**，直接返回 `{log_probs, entropy, values, ...}`

`patch_fused_forward(model)` / `unpatch_fused_forward(model)` 在 engine 的 forward loop 里成对使用，只在这一段 forward 期间修改 mcore 行为，避免全局影响。

### 18.4.7 `mtp_patch.py`：DeepSeek MTP 支持

**MTP（Multi-Token Prediction）** 是 DeepSeek-V3 的架构创新：在最后一层多接几个 head，同时预测 t+1, t+2, ..., t+n 位置。

`mtp_patch.py` 做三件事：
1. `patch_postprocess(model)`：替换 mcore `GPTModel._postprocess` → `_megatron_gptmodel_postprocess`（同时算 main + MTP head 的 loss）
2. `patch_mtp_layer_get_embeddings(model)`：MTP layer 需要读取 t 位置的 hidden state，patch embedding lookup
3. `patch_mtp_layer_checkpointed_forward(model)`：MTP layer 的 activation checkpoint（省显存）

对应的 `unpatch_*` 用于 rollout 之前恢复 mcore 原状（避免 rollout 引擎撞到 verl 定制方法）。

### 18.4.8 `patch.py`：mcore 上游 bug 修复 + 兼容性 shim

`apply_patch()`（:95）是 mcore 使用之前**必调用**的补丁大合集：

| 补丁 | 修的什么 |
|---|---|
| `apply_fast_hadamard_transform_shim` | DeepSeek DSA (sparse attention) 依赖 CUDA `fast_hadamard_transform` 包，ROCm 装不上 → 用 pure-torch 实现兜底 |
| MLA `get_query_key_value_tensors` 修正 | mcore 0.12 里 packed_seq_params 场景 MLA 位置编码计算错 |
| `apply_patch_megatron_v012_with_torch_v28_v29` | mcore 0.12 与 torch 2.8/2.9 的兼容问题 |
| `apply_mtp_inference_patch` | rollout inference 时 MTP layer 的问题 |
| `apply_patch_megatron_recomputation_backward` | 激活重算 + backward 触发的 crash |
| `apply_patch_mbridge` | mbridge 兼容层 |

**特点**：所有补丁都是**"只加不删"**——即使 mcore 后来修了，patch 里也用版本判断 (`mcore_ge_013 = version.parse(...)`) 兼容新旧。

### 18.4.9 `weight_converter.py`：mcore → HF 权重转换（rollout 同步核心）

**问题**：训练用 mcore GPTModel（TP shard、layer name 是 `decoder.layers.0.self_attention.linear_qkv.weight`）；rollout 用 vLLM 里的 HF 格式（layer name 是 `model.layers.0.self_attn.q_proj.weight`）。二者**layer 名 + 权重排布**都不一样。

`McoreToHFWeightConverterBase`（:25）+ 各家族子类（`Dense/Qwen2Moe/Qwen3Moe/Mixtral/Dpskv3/Qwen2_5_VL`）负责：

1. **名字映射**：`decoder.layers.{i}.self_attention.linear_qkv.weight` → `model.layers.{i}.self_attn.{q,k,v}_proj.weight`
2. **形状转换**：mcore linear_qkv 是**融合后的** `(qkv_dim, hidden)`，HF 是分离的 3 个 `(dim, hidden)` → 拆分 + `reshape/transpose`
3. **MoE 专家权重合并**：mcore 用 `grouped_gemm` 存成 `(num_experts, in, out)`，HF 是逐 expert 的独立参数 → 拆开
4. **DeepSeek MLA 特化**：`linear_q_down_proj` / `linear_q_up_proj` / `linear_kv_down_proj` 等 → HF `q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj`

**关键调用**：`checkpoint_engine.get_per_tensor_param` 里，对 mcore 后端会走：

```
for name, tensor in mcore_state_dict:
    for hf_name, hf_tensor in converter.convert_param(name, tensor):
        yield hf_name, hf_tensor
```

**为什么每个 family 一个 Converter？** 因为 MoE / MLA / VLM 都有独特的 param 组织方式，共用一个 Converter 会退化成大 if-else，不如按 family 拆开。子类都继承 `McoreToHFWeightConverterDense`，共享 dense 部分逻辑，只覆盖 `_convert_moe_param` / `_convert_mla_param`。

### 18.4.10 `util.py`：THD/BSHD 预/后处理（39KB 的隐形基础设施）

15 个函数，几乎都是关于**变长序列打包/解包**的样板代码：

```
preprocess_packed_seqs         → (input_ids, attn_mask) → (packed_ids, cu_seqlens, ...) 
postprocess_packed_seqs        → 反向解包，恢复 (bsz, seq_len) 形状
preprocess_bshd                → BSHD 路径（padding to max_seq_len）
postprocess_bshd               → BSHD 后处理
preprocess_thd_engine          → engine.forward 专用 (含 SP slice)
postprocess_thd_engine         → engine.forward 专用后处理
preprocess_for_mindspeed       → MindSpeed (NPU) 特化路径
build_vlm_attn_mask_thd        → VLM 的 attention mask 构造
_compute_fp8_thd_align_size    → FP8 训练要求序列 pad 到 16 或 32 对齐
```

**关键抽象 `_packed_seq_params_supports(field_name)`**：mcore 不同版本 `PackedSeqParams` 支持的字段不一样（0.12 没有 `use_te_padding_free`，0.16 有），这个探针函数运行时探测有没有，动态决定要不要传。

## 18.5 legacy: 顶层 `registry.py` + `weight_loader_registry.py`

### 18.5.1 现状

顶层 `registry.py` 只支持 4 个模型：

```python
_MODELS = {
    "LlamaForCausalLM":    ("llama",    ("ParallelLlamaForCausalLMRmPadPP", ...)),
    "Qwen2ForCausalLM":    ("qwen2",    ...),
    "MistralForCausalLM":  ("mistral",  ...),
    "ApertusForCausalLM":  ("apertus",  ...),
}

ModelRegistry.load_model_cls(model_arch, value=False):
    # 动态 import: verl.models.{llama,qwen2,...}.megatron.modeling_*_megatron
```

**但 verl 里已经没有 `verl/models/llama/megatron/` 这种子目录了**——这是 v0.9 之前手写 modeling 的遗留代码，一旦真被调用会 `ModuleNotFoundError`。

**保留原因**：外部脚本 `verl/scripts/converter_hf_to_mcore.py` 和一些老的 checkpoint 加载路径**还在引用**，删除会破坏兼容性；等 v0.9 正式移除。

### 18.5.2 `weight_loader_registry.py`

同样是 legacy，服务于**离线权重转换**（`hf → mcore dist_checkpointing`）：

```
get_weight_loader(arch): load_state_dict_to_megatron_gptmodel (loader.py 里)
get_weight_saver(arch):  merge_megatron_ckpt_gptmodel + 各 family 变体 (saver.py 里)
```

被 `verl/scripts/model_merger` 和 `converter_hf_to_mcore.py` 调用。生产训练**在线**权重同步走的是 mcore 的 `dist_checkpointing` + `weight_converter.py`，不走这条路径。

## 18.6 数据流：一次训练 step 里 models 的作用点

```
Trainer (Driver)
    │
    v
ActorRolloutRefWorker.update_actor(data)   # @register mesh_name=actor
    │
    v
TrainingWorker.train_mini_batch(data)
    │
    v
engine.train_batch(data, loss_fn):         # FSDP or Megatron
    │
    ├── forward_step(micro_batch, loss_fn):
    │     
    │     ┌── FSDP 路径 ─────────────────────────────┐
    │     │  self.module (HF LlamaForCausalLM 已 patched):│
    │     │    ├── attention layers: _ulysses_flash_attention_forward  ← models/transformers/monkey_patch.py
    │     │    ├── MLP: TiledMLP.apply (若开启)             ← models/transformers/tiled_mlp.py
    │     │    └── forward: forward_with_torch_backend    ← models/transformers/dense_common.py
    │     └────────────────────────────────────────────┘
    │     
    │     ┌── Megatron 路径 ─────────────────────────┐
    │     │  forward_fn = get_mcore_forward_fn(hf_config)  ← models/mcore/registry.py
    │     │  forward_fn(self.module, input_ids, ...):
    │     │    ├── preprocess_packed_seqs (thd)             ← models/mcore/util.py
    │     │    ├── mcore GPTModel(**args) (含 apply_patch()) ← models/mcore/patch.py
    │     │    ├── logits_processor 可选融合                 ← models/mcore/model_forward_fused.py
    │     │    └── postprocess_packed_seqs                  ← models/mcore/util.py
    │     └────────────────────────────────────────────┘
    │
    ├── loss = loss_fn(config, model_output, data, dp_group)   # ppo_loss / value_loss
    │
    └── backward + optimizer_step

------------------------------------------------------------------
Rollout weight sync (engine_workers.update_weights)
    │
    v
engine.get_per_tensor_param():
    │
    ┌── FSDP 路径 ──────────────────────────────┐
    │   FSDP.summon_full_params → yield (name, tensor)   （HF 命名，直接给 vLLM）
    └───────────────────────────────────────────┘
    
    ┌── Megatron 路径 ──────────────────────────┐
    │   for name, tensor in gptmodel_iter():
    │     converter = get_mcore_weight_converter(hf_config, dtype)  ← models/mcore/registry.py
    │     for hf_name, hf_tensor in converter.convert_param(name, tensor):  ← models/mcore/weight_converter.py
    │       yield hf_name, hf_tensor   （已转成 HF 命名/排布）
    └───────────────────────────────────────────┘
    │
    v
rollout.update_weights(...) → vLLM/SGLang 内部 load_weights
```

## 18.7 扩展点：如何支持新模型

### 18.7.1 FSDP 路径（推荐，成本低）

前提：模型已在上游 `transformers` 库支持。

1. **不动 verl 代码**试跑：`config.model.path=<hf_hub_or_local>` 走 `apply_monkey_patch` 通用分支（Llama/Qwen2 结构 attention 会走 `_ulysses_flash_attention_forward` 通用替换）
2. 如果 SP 有问题：在 `monkey_patch.py::apply_monkey_patch` 加一个 `elif model.config.model_type == "mymodel":` 分支，替换 `MyModelAttention.forward`
3. 如果想开 fused kernel：在 `verl/models/transformers/` 新增 `mymodel.py`，实现 `forward_with_torch_backend` / `forward_with_triton_backend`，然后在 `patch_forward_with_backends` 里加分支

### 18.7.2 Megatron 路径（复杂，但可拿 CP/EP 加速）

1. 在 `mcore/registry.py::SupportedModel` 加枚举项 `MY_MODEL = "MyModelForCausalLM"`
2. `config_converter.py` 加 `hf_to_mcore_config_mymodel(hf_config, dtype, **overrides)`，注册到 `MODEL_CONFIG_CONVERTER_REGISTRY`
3. `model_initializer.py` 加 `MyModelInitializer(BaseModelInitializer)` 或复用 `DenseModel`，注册到 `MODEL_INITIALIZER_REGISTRY`
4. `MODEL_FORWARD_REGISTRY` 用 `model_forward_gen()` 兜底（除非有 VLM 特化）
5. `weight_converter.py` 加 `McoreToHFWeightConverterMyModel(McoreToHFWeightConverterDense)`，覆盖 `_convert_*` 方法处理独特结构
6. **优先**：把上游支持提交到 `Megatron-Bridge`（NVIDIA 仓库），verl 侧就近乎零改动

## 18.8 关键设计模式与最佳实践总结

| 模式 | 位置 | 价值 |
|---|---|---|
| **Monkey patch 而非重写** | `transformers/monkey_patch.py` | 与上游 HF 保持零抄袭；升级 transformers 只需改极少 patch |
| **策略池 + Registry 派发** | `mcore/registry.py` 五张表 | 一行 YAML 切换模型架构；扩展点集中 |
| **模板方法（TemplateMethod）** | `BaseModelInitializer.initialize` | 固定 GPTModel 构造骨架，子类只填 `get_transformer_layer_spec` |
| **上下文管理器 patch/unpatch 对** | `mtp_patch.py`、`model_forward_fused.py::FusedOutputProcessorContext` | 只在必要窗口修改 mcore 全局行为，避免污染 rollout |
| **`**override_transformer_config_kwargs` 直穿** | `config_converter.py` 所有函数末尾 | 无需改代码即可访问 mcore 全部新特性 |
| **版本感知 shim** | `patch.py` 里的 `version.parse(mcore_version)` 判断 | 一份代码兼容 mcore 0.12 ~ 0.16+ |
| **依赖降级兜底** | `apply_fast_hadamard_transform_shim` | 硬件（ROCm/NPU）装不上 CUDA 包时用 pure-torch 兜底 |
| **fused kernel 显存换 forward 速度** | `dense_common.forward_with_torch_backend` + `logits_processor` | logits 不落地，130k vocab × 8k seq ≈ 4GB/sample 显存省掉 |

## 18.9 常见陷阱

| 陷阱 | 现象 | 规避 |
|---|---|---|
| `apply_monkey_patch` 忘了调 | attention 走 HF 原版，SP 大小 > 1 时 hang 或结果错 | Engine 的 `_build_module` 里已内置调用；扩展新 engine 时别忘 |
| Ulysses SP 头数不整除 | `num_key_value_heads` % `ulysses_sp_size` != 0 | 用 `repeat_kv` 在 attn 前扩展 kv 头（monkey_patch 已实现） |
| 权重同步 name 对不上 | vLLM 报 `Missing key 'model.layers.0.self_attn.q_proj.weight'` | 检查 `weight_converter.py` 是否覆盖到该 param（MoE/MLA 特化）|
| Megatron config override 未生效 | `+actor.megatron.override_transformer_config.foo=bar` 写了但没用 | 确认 field 名与 mcore `TransformerConfig` 完全一致（不是 HF 名）|
| MTP 加载 rollout 报错 | rollout 用了 verl patch 过的 mcore GPTModel | `unpatch_postprocess(model)` 在 rollout 前恢复；`update_weights` 里已做 |
| DeepSeek DSA 在 ROCm 挂 | `hadamard_transform` is None | 引入 `apply_fast_hadamard_transform_shim()`（`apply_patch()` 已内置）|
| VLM SP 输入没切 | 视觉 embed 全 rank 都拿全量 → OOM | 确保 `patch_vlm_for_ulysses_input_slicing(TextModel)` 生效 |
| NPU 上不用 npu_patch | silu / rms_norm 掉到 python 慢 fallback | `apply_npu_patches()` 在 NPU device 初始化流程里调 |

## 18.10 一图总结

```
                       ┌─────────────────────────────────────┐
                       │       verl/models/                   │
                       │       "适配层"（不建模，只桥接）        │
                       └──────────────┬──────────────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                v                     v                     v
        ┌───────────────┐   ┌───────────────────┐   ┌──────────────┐
        │ transformers/ │   │ mcore/            │   │ registry.py  │
        │  (HF 适配)     │   │  (Megatron 集成)   │   │  (legacy,    │
        │               │   │                   │   │   v0.9 移除) │
        │  apply_       │   │  Megatron-Bridge  │   └──────────────┘
        │  monkey_      │   │  (AutoBridge)      │
        │  patch:       │   │                    │
        │  ├ Ulysses SP │   │  5 大 Registry:    │
        │  ├ RmPad      │   │  ├ config          │
        │  ├ Fused CE   │   │  ├ initializer     │
        │  ├ TiledMLP   │   │  ├ forward         │
        │  ├ Prefix     │   │  ├ fused forward   │
        │  │  Grouper   │   │  └ weight_conv     │
        │  └ NPU 特化    │   │                    │
        │               │   │  util.py:          │
        │  dense_common │   │  THD/BSHD 打包解包  │
        │  llama/qwen2/ │   │                    │
        │  qwen3/       │   │  patch.py:         │
        │  qwen2_vl/    │   │  mcore bug fix +   │
        │  qwen3_vl/    │   │  shim              │
        │  glm4v/       │   │                    │
        │  kimi_vl/...  │   │  mtp_patch.py:     │
        │               │   │  DeepSeek MTP      │
        └───────┬───────┘   └──────────┬─────────┘
                │                      │
                v                      v
     workers/engine/fsdp/*   workers/engine/megatron/*
        (FSDP1/FSDP2/          (mcore GPTModel + Distributed Optim)
         TorchTitan/            + get_per_tensor_param(_shard/_delta_shard)
         veomni 共用)                                     │
                                                          v
                                             rollout weight sync (vLLM/SGLang)
                                             (HF 命名 + 排布 已经匹配)
```

**核心一句话**：`verl/models/` 是**"上游模型生态"和"verl 训练/rollout 生态"之间的粘合层**——`transformers/` 用 monkey patch 就地改造 HF 模型，`mcore/` 用 Registry + Bridge 桥接 Megatron，共同保证 **上游一行不抄、verl 定制最小化** 的架构定位。

---

**深入相关模块**：
- [15-训练后端与优化器配置](15-训练后端与优化器配置.md) - 训练后端选择与 engine 配置
- [17-workers模块设计与运行机制](17-workers模块设计与运行机制.md) - engine 如何调用 models
- [08-算子库构成与原理](08-算子库构成与原理.md) - fused kernel 的实现细节
- [16-显存卸载与vLLM睡眠唤醒机制](16-显存卸载与vLLM睡眠唤醒机制.md) - weight_converter 在 rollout 同步的位置

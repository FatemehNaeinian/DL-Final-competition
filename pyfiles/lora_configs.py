from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from peft import LoraConfig


CfgName = Literal[
    # Legacy / notebook-aligned
    "section9_lora",
    "section10_dora",
    "section12_lora",
    # New comparison set
    "attn_only_lora",       # q/v on last 6 text layers (small, attention-only)
    "attn_mlp_lora",        # q/v + MLP (gate/up/down) on last 6 text layers
    "attn_mlp_dora",        # same as attn_mlp_lora but use_dora=True
    "mlp_only_lora",       # MLP only (gate/up/down), higher r, last 6 text layers
    "attn_mlp_lora_r2",    # q/v + MLP at r=2 (lower rank under same budget)
    "attn_mlp_lora_r1_last16",  # legal last-16 adaptation (very low rank)
    "attn_mlp_lora_r2_last16",  # legal last-16 adaptation (low rank)
    "attn_mlp_lora_r4_last16",  # higher rank still under 5M cap
    "attn_mlp_lora_r2_last24",  # more layers (last 24) at low rank
    "attn_mlp_dora_r2_last16",  # DoRA variant, last 16
    "attn_mlp_dora_r2_last24",  # DoRA variant, last 24
]


@dataclass(frozen=True)
class LoraRecipe:
    name: str
    cfg: LoraConfig


def text_num_hidden_layers_from_config(cfg) -> int:
    tc = getattr(cfg, "text_config", None)
    if tc is not None and getattr(tc, "num_hidden_layers", None) is not None:
        return int(tc.num_hidden_layers)
    raise ValueError("Could not read text_config.num_hidden_layers from config")


def last_k_layers(num_layers: int, last_k: int) -> list[int]:
    last_k = min(last_k, num_layers)
    return list(range(num_layers - last_k, num_layers))


def make_cfg(name: CfgName, *, num_layers: int) -> LoraRecipe:
    if name == "section9_lora":
        cfg = LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "section10_dora":
        cfg = LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05, bias="none", use_dora=True,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "section12_lora":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_only_lora":
        # Attention-only LoRA: compare against attn+MLP variant.
        # With only q/v in last 6 text layers we can afford a larger r.
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora":
        # Expand LoRA to MLP layers (gate/up/down) on top of attention.
        cfg = LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_dora":
        # DoRA upgrade of attn+MLP recipe.
        cfg = LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05, bias="none", use_dora=True,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "mlp_only_lora":
        # MLP adapters only; fewer modules than q+v+MLP so use larger rank within 5M cap.
        cfg = LoraConfig(
            r=10, lora_alpha=20, lora_dropout=0.05, bias="none",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora_r2":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 6),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora_r1_last16":
        cfg = LoraConfig(
            r=1, lora_alpha=8, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 16),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora_r2_last16":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 16),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora_r4_last16":
        cfg = LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 16),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_lora_r2_last24":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 24),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_dora_r2_last16":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none", use_dora=True,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 16),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    if name == "attn_mlp_dora_r2_last24":
        cfg = LoraConfig(
            r=2, lora_alpha=16, lora_dropout=0.05, bias="none", use_dora=True,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=last_k_layers(num_layers, 24),
            layers_pattern="layers", task_type="CAUSAL_LM",
        )
        return LoraRecipe(name=name, cfg=cfg)

    raise ValueError(f"Unknown cfg name: {name}")

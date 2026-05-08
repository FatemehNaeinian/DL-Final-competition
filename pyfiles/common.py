from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image


MODEL_ID_DEFAULT = "HuggingFaceTB/SmolVLM-500M-Instruct"


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    img_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_paths(root: Path) -> Paths:
    data_dir = root / "data"
    img_dir = data_dir / "images"
    return Paths(root=root, data_dir=data_dir, img_dir=img_dir)


REQUIRED_COLS_TRAINVAL = ("id", "image_path", "question", "choices", "num_choices", "answer")
REQUIRED_COLS_TEST = ("id", "image_path", "question", "choices", "num_choices")


def load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Parse choices JSON strings into Python lists
    for df in (train, val, test):
        if "choices" in df.columns:
            df["choices"] = df["choices"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Drop only missing required columns (keep optional columns like lecture/hint if present)
    train = train.dropna(subset=[c for c in REQUIRED_COLS_TRAINVAL if c in train.columns]).reset_index(drop=True)
    val = val.dropna(subset=[c for c in REQUIRED_COLS_TRAINVAL if c in val.columns]).reset_index(drop=True)
    test = test.dropna(subset=[c for c in REQUIRED_COLS_TEST if c in test.columns]).reset_index(drop=True)

    return train, val, test


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MCQ_COMPLETION_MODES = frozenset({"letter", "letter_choice", "choice_text"})


def mcq_completion_suffix(
    row: pd.Series,
    option_idx: int,
    mode: str = "letter",
    *,
    choice_max_chars: int = 160,
) -> str:
    """Suffix after ``Answer:`` for MCQ scoring / contrastive training.

    ``letter`` matches the default notebook scorer (``' A'``, ``' B'``, …).
    """
    if mode not in MCQ_COMPLETION_MODES:
        raise ValueError(f"Unknown mcq completion mode {mode!r}; allowed {MCQ_COMPLETION_MODES}")
    letter = CHOICE_LETTERS[option_idx]
    choices = row["choices"]
    if not isinstance(choices, (list, tuple)):
        choices = list(choices)
    text = str(choices[option_idx]).strip().replace("\n", " ")
    if len(text) > choice_max_chars:
        text = text[: choice_max_chars - 1] + "…"
    if mode == "letter":
        return f" {letter}"
    if mode == "letter_choice":
        return f" {letter}. {text}"
    return f" {text}"


def build_prompt(row: pd.Series, *, include_answer: bool = False) -> str:
    """Matches the notebook baseline prompt: context (lecture+hint), question, choices, 'Answer:' cue."""
    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    n = int(row["num_choices"]) if "num_choices" in row else len(choices)
    n = min(n, len(CHOICE_LETTERS))
    choices_str = "\n".join(f"  {CHOICE_LETTERS[i]}. {choices[i]}" for i in range(n))

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"

    if include_answer:
        answer_idx = int(row["answer"])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"

    return prompt


def load_image(img_dir: Path, image_path: str, *, size: int) -> Image.Image:
    return Image.open(img_dir / image_path).convert("RGB").resize((size, size))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def read_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def is_peft_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def load_multimodal_mcq_model(
    model_id: str,
    weights_dir: Path | None,
    *,
    device_map: str | dict | None = "auto",
    torch_dtype: torch.dtype | None = None,
):
    """Load a vision-language MCQ model from the Hub, a PEFT folder, or a full fine-tuned checkpoint.

    If ``weights_dir`` is None, loads ``model_id`` from the Hub.
    If it contains ``adapter_config.json``, loads Hub ``model_id`` + PEFT adapters.
    Otherwise loads a standalone fine-tuned checkpoint from ``weights_dir`` (no LoRA).
    """
    from transformers import AutoModelForVision2Seq
    from peft import PeftModel

    kw: dict = {"low_cpu_mem_usage": True}
    if device_map is not None:
        kw["device_map"] = device_map
    if torch_dtype is not None:
        kw["torch_dtype"] = torch_dtype

    if weights_dir is None:
        return AutoModelForVision2Seq.from_pretrained(model_id, **kw)

    wd = Path(weights_dir)
    if not wd.exists():
        raise FileNotFoundError(wd)
    if is_peft_adapter_dir(wd):
        base = AutoModelForVision2Seq.from_pretrained(model_id, **kw)
        return PeftModel.from_pretrained(base, str(wd))
    full_kw: dict = {"low_cpu_mem_usage": True}
    if device_map is not None:
        full_kw["device_map"] = device_map
    return AutoModelForVision2Seq.from_pretrained(str(wd), **full_kw)


from __future__ import annotations

import argparse
import gc
import random as random_mod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import get_peft_model, prepare_model_for_kbit_training

from .common import (
    CHOICE_LETTERS,
    MCQ_COMPLETION_MODES,
    MODEL_ID_DEFAULT,
    Paths,
    build_prompt,
    get_paths,
    load_csvs,
    mcq_completion_suffix,
    save_json,
    set_seed,
)
from .scoring import mcq_mean_completion_logprobs_options
from .lora_configs import make_cfg, text_num_hidden_layers_from_config

_FT_METADATA_ALLOWED = frozenset({"grade", "subject", "topic", "hint", "lecture"})
_FT_METADATA_ORDER = ("grade", "subject", "topic", "hint", "lecture")


def parse_ft_metadata_fields(spec: str | None) -> frozenset[str]:
    """Comma-separated subset of grade, subject, topic, hint, lecture for multimodal FT prompts."""
    if spec is None or not str(spec).strip():
        return frozenset()
    parts = {p.strip().lower() for p in str(spec).split(",") if p.strip()}
    bad = parts - _FT_METADATA_ALLOWED
    if bad:
        raise ValueError(f"Unknown ft-metadata fields {bad}; allowed {_FT_METADATA_ALLOWED}")
    return frozenset(parts)


def load_captions_csv(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    cap_df = pd.read_csv(path)
    return {str(r["id"]): str(r["caption"]) for _, r in cap_df.iterrows()}


def build_prompt_ft(
    row: pd.Series,
    *,
    answer_prefix: str = "Answer:",
    use_baseline_prompt: bool = False,
    metadata_fields: frozenset[str] | None = None,
    choice_max_chars: int = 160,
    hint_max_chars: int = 320,
    lecture_max_chars: int = 320,
    caption: str | None = None,
) -> str:
    """Fine-tuning prompt for multimodal MCQ.

    If `use_baseline_prompt` is True, uses `common.build_prompt(...)` (lecture+hint context-first),
    then optionally injects caption/metadata before the Question block. This keeps FT aligned with eval.
    """
    if use_baseline_prompt:
        prompt = build_prompt(row, include_answer=False)
        lines = prompt.splitlines()
        # Best-effort: insert caption/metadata immediately after <image> line.
        insert_at = 1 if lines and lines[0].strip() == "<image>" else 0

        extras: list[str] = []
        if caption is not None and str(caption).strip():
            extras.append(f"Image description: {str(caption).strip()}")

        mf = metadata_fields or frozenset()
        for field in _FT_METADATA_ORDER:
            if field not in mf or field not in row.index:
                continue
            val = row.get(field)
            if pd.isna(val) or not str(val).strip():
                continue
            s = str(val).strip()
            if field == "hint":
                extras.append(f"Hint: {clip(s, hint_max_chars)}")
            elif field == "lecture":
                extras.append(f"Lecture: {clip(s, lecture_max_chars)}")
            elif field == "grade":
                extras.append(f"Grade: {s}")
            elif field == "subject":
                extras.append(f"Subject: {s}")
            elif field == "topic":
                extras.append(f"Topic: {s}")

        if extras:
            lines[insert_at:insert_at] = extras

        ap = (answer_prefix or "Answer:").strip()
        if not ap.endswith(":"):
            ap = ap + ":"
        # Replace final Answer cue if present; otherwise append.
        if lines and lines[-1].strip().lower().startswith("answer"):
            lines[-1] = ap
        else:
            lines.append(ap)
        return "\n".join(lines) + "\n"

    """Short prompt for finetuning (avoid huge lecture blocks; metadata optional)."""
    n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n, len(CHOICE_LETTERS))

    q = str(row["question"]).strip()
    choices = row["choices"]
    if not isinstance(choices, (list, tuple)):
        choices = list(choices)

    def clip(s: str, max_chars: int) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= max_chars else (s[: max_chars - 1] + "…")

    lines: list[str] = ["<image>"]
    if caption is not None and str(caption).strip():
        lines.append(f"Image description: {str(caption).strip()}")
    lines.append("You are given a multiple-choice question.")

    mf = metadata_fields or frozenset()
    for field in _FT_METADATA_ORDER:
        if field not in mf or field not in row.index:
            continue
        val = row.get(field)
        if pd.isna(val) or not str(val).strip():
            continue
        s = str(val).strip()
        if field == "hint":
            lines.append(f"Hint: {clip(s, hint_max_chars)}")
        elif field == "lecture":
            lines.append(f"Lecture: {clip(s, lecture_max_chars)}")
        elif field == "grade":
            lines.append(f"Grade: {s}")
        elif field == "subject":
            lines.append(f"Subject: {s}")
        elif field == "topic":
            lines.append(f"Topic: {s}")

    lines.append(f"Question: {q}")
    lines.append("Choices:")
    for i in range(n):
        lines.append(f"{CHOICE_LETTERS[i]}. {clip(str(choices[i]), choice_max_chars)}")

    ap = (answer_prefix or "Answer:").strip()
    if not ap.endswith(":"):
        ap = ap + ":"
    lines.append(ap)
    return "\n".join(lines) + "\n"


_TEXT_ONLY_FIELD_ALLOWED = frozenset({"hint", "grade", "lecture"})


def parse_text_only_fields(spec: str | None) -> frozenset[str]:
    """Comma-separated subset of hint, grade, lecture (for ``--text-only-fields``)."""
    if spec is None or not str(spec).strip():
        return frozenset()
    parts = {p.strip().lower() for p in str(spec).split(",") if p.strip()}
    bad = parts - _TEXT_ONLY_FIELD_ALLOWED
    if bad:
        raise ValueError(f"Unknown text-only fields {bad}; allowed {_TEXT_ONLY_FIELD_ALLOWED}")
    return frozenset(parts)


def build_prompt_text_only_mcq(
    row: pd.Series,
    *,
    fields: frozenset[str],
    choice_max_chars: int = 220,
    hint_max_chars: int = 512,
    lecture_max_chars: int = 512,
) -> str:
    """MCQ prompt without ``<image>`` for text-only fine-tuning.

    Always includes **Question**, **Choices**, and ``Answer:``.
    Optionally prepends **Grade**, **Hint**, **Lecture** when requested and present in the row.
    """
    n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n, len(CHOICE_LETTERS))

    choices = row["choices"]
    if not isinstance(choices, (list, tuple)):
        choices = list(choices)

    def clip(s: str, max_chars: int) -> str:
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= max_chars else (s[: max_chars - 1] + "…")

    lines: list[str] = ["You are given a multiple-choice question."]

    if "grade" in fields and "grade" in row.index:
        g = row.get("grade")
        if pd.notna(g) and str(g).strip():
            lines.append(f"Grade: {str(g).strip()}")

    if "lecture" in fields and "lecture" in row.index:
        lec = row.get("lecture")
        if pd.notna(lec) and str(lec).strip():
            lines.append(f"Lecture: {clip(str(lec), lecture_max_chars)}")

    if "hint" in fields and "hint" in row.index:
        h = row.get("hint")
        if pd.notna(h) and str(h).strip():
            lines.append(f"Hint: {clip(str(h), hint_max_chars)}")

    lines.append(f"Question: {str(row['question']).strip()}")
    lines.append("Choices:")
    for i in range(n):
        lines.append(f"{CHOICE_LETTERS[i]}. {clip(str(choices[i]), choice_max_chars)}")
    lines.append("Answer:")
    return "\n".join(lines) + "\n"


class LoRAMCQDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        img_dir: Path,
        processor,
        img_size: int,
        captions_by_id: dict[str, str] | None = None,
        augment: bool = False,
        ft_metadata_fields: frozenset[str] | None = None,
        answer_prefix: str = "Answer:",
        use_baseline_prompt: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.img_size = img_size
        self.captions_by_id = captions_by_id or {}
        self.ft_metadata_fields = frozenset() if ft_metadata_fields is None else frozenset(ft_metadata_fields)
        self.answer_prefix = answer_prefix
        self.use_baseline_prompt = bool(use_baseline_prompt)
        self._aug = None
        if augment:
            try:
                from torchvision import transforms

                self._aug = transforms.Compose(
                    [
                        transforms.RandomApply([transforms.ColorJitter(0.08, 0.08, 0.08, 0.02)], p=0.65),
                        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                    ]
                )
            except ImportError as e:
                raise RuntimeError("Image augmentation requires torchvision.") from e

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["image_path"]).convert("RGB").resize((self.img_size, self.img_size))
        if self._aug is not None:
            img = self._aug(img)

        cap = None
        if self.captions_by_id:
            cap = self.captions_by_id.get(str(row["id"]))

        prompt = build_prompt_ft(
            row,
            answer_prefix=self.answer_prefix,
            metadata_fields=self.ft_metadata_fields,
            caption=cap,
            use_baseline_prompt=self.use_baseline_prompt,
        )
        ans_idx = int(row["answer"])
        completion = f" {CHOICE_LETTERS[ans_idx]}"
        full_text = prompt + completion

        n_completion = len(self.processor.tokenizer.encode(completion, add_special_tokens=False))

        enc = self.processor(text=full_text, images=img, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        seq_len = int(labels.shape[0])
        n_sup = min(n_completion, seq_len)
        labels[:] = -100
        labels[-n_sup:] = enc["input_ids"][-n_sup:]

        if "attention_mask" in enc:
            labels[enc["attention_mask"] == 0] = -100

        if "pixel_values" in enc and enc["pixel_values"].dtype == torch.float32:
            enc["pixel_values"] = enc["pixel_values"].to(torch.float16)

        enc["labels"] = labels
        return enc


def _permute_mcq_row_choice_order(row: pd.Series, *, rng: random_mod.Random) -> pd.Series:
    """Randomly reorder choice lines (A,B,…) together with texts; remap ``answer`` to the new letter index."""
    ans_idx = int(row["answer"])
    n_raw = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n_raw, len(CHOICE_LETTERS))
    choices = row["choices"]
    if not isinstance(choices, (list, tuple)):
        choices = list(choices)
    choices = list(choices[:n])

    perm = list(range(n))
    rng.shuffle(perm)
    new_choices = [choices[o] for o in perm]

    inverse = [0] * n
    for new_slot, old_idx in enumerate(perm):
        inverse[old_idx] = new_slot

    row2 = row.copy()
    row2["choices"] = new_choices
    row2["answer"] = inverse[ans_idx]
    row2["num_choices"] = n
    return row2


class MCQContrastiveMultimodalDataset(torch.utils.data.Dataset):
    """Multimodal MCQ with one forward pass worth of options per example (for contrastive CE loss)."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        img_dir: Path,
        processor,
        img_size: int,
        captions_by_id: dict[str, str] | None = None,
        augment: bool = False,
        ft_metadata_fields: frozenset[str] | None = None,
        answer_prefix: str = "Answer:",
        use_baseline_prompt: bool = False,
        completion_mode: str = "letter",
        choice_max_chars: int = 160,
        shuffle_choice_order_train: bool = False,
        shuffle_choice_order_seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.img_size = img_size
        self.captions_by_id = captions_by_id or {}
        self.ft_metadata_fields = frozenset() if ft_metadata_fields is None else frozenset(ft_metadata_fields)
        self.answer_prefix = answer_prefix
        self.use_baseline_prompt = bool(use_baseline_prompt)
        self.completion_mode = completion_mode
        self.choice_max_chars = int(choice_max_chars)
        self.shuffle_choice_order_train = bool(shuffle_choice_order_train)
        self.shuffle_choice_order_seed = int(shuffle_choice_order_seed)
        self._aug = None
        if augment:
            try:
                from torchvision import transforms

                self._aug = transforms.Compose(
                    [
                        transforms.RandomApply([transforms.ColorJitter(0.08, 0.08, 0.08, 0.02)], p=0.65),
                        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                    ]
                )
            except ImportError as e:
                raise RuntimeError("Image augmentation requires torchvision.") from e

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        if self.shuffle_choice_order_train:
            rng = random_mod.Random(int(self.shuffle_choice_order_seed) + int(idx) * (2**17))
            row = _permute_mcq_row_choice_order(row, rng=rng)
        img = Image.open(self.img_dir / row["image_path"]).convert("RGB").resize((self.img_size, self.img_size))
        if self._aug is not None:
            img = self._aug(img)

        cap = None
        if self.captions_by_id:
            cap = self.captions_by_id.get(str(row["id"]))

        prompt = build_prompt_ft(
            row,
            answer_prefix=self.answer_prefix,
            metadata_fields=self.ft_metadata_fields,
            caption=cap,
            use_baseline_prompt=self.use_baseline_prompt,
        )
        ans_idx = int(row["answer"])
        n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
        n = min(n, len(CHOICE_LETTERS))

        completions = [
            mcq_completion_suffix(row, i, self.completion_mode, choice_max_chars=self.choice_max_chars)
            for i in range(n)
        ]
        full_texts = [prompt + c for c in completions]

        enc_opts: list[dict[str, torch.Tensor]] = []
        for ft in full_texts:
            enc = self.processor(text=ft, images=img, return_tensors="pt")
            enc = {k: v.squeeze(0) for k, v in enc.items()}
            enc_opts.append(enc)

        prompt_enc = self.processor(text=prompt, images=img, return_tensors="pt")
        prompt_len = int(prompt_enc["input_ids"].shape[1])

        pv = enc_opts[0]["pixel_values"]
        if pv.dtype == torch.float32:
            pv = pv.to(torch.float16)

        return {
            "pixel_values": pv,
            "option_input_ids": [e["input_ids"] for e in enc_opts],
            "option_attention_mask": [e["attention_mask"] for e in enc_opts],
            "prompt_len": prompt_len,
            "gold_idx": ans_idx,
            "num_choices": n,
        }


class MCQContrastiveCollator:
    def __init__(self, *, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_n = max(int(f["num_choices"]) for f in features)
        max_len = 0
        for f in features:
            for ids in f["option_input_ids"]:
                max_len = max(max_len, int(ids.shape[0]))

        bsz = len(features)
        input_ids = torch.full((bsz, max_n, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_n, max_len), dtype=torch.long)
        option_valid = torch.zeros((bsz, max_n), dtype=torch.bool)

        pixel_values: list[torch.Tensor] = []
        prompt_lens: list[int] = []
        gold_idx: list[int] = []

        for b, f in enumerate(features):
            pixel_values.append(f["pixel_values"])
            prompt_lens.append(int(f["prompt_len"]))
            gold_idx.append(int(f["gold_idx"]))
            n = int(f["num_choices"])
            for o in range(n):
                ids = f["option_input_ids"][o]
                am = f["option_attention_mask"][o]
                seq_len = int(ids.shape[0])
                input_ids[b, o, :seq_len] = ids
                attention_mask[b, o, :seq_len] = am
                option_valid[b, o] = True

        return {
            "mcq_contrastive": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "option_valid": option_valid,
                "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
                "gold_idx": torch.tensor(gold_idx, dtype=torch.long),
                "pixel_values": torch.stack(pixel_values),
            }
        }


class MCQContrastiveTrainer(Trainer):
    """Cross-entropy over option log-likelihood scores (matches multimodal MCQ inference)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        mcq = inputs["mcq_contrastive"]
        bsz, max_n, seq_len = mcq["input_ids"].shape

        flat_ids = mcq["input_ids"].reshape(bsz * max_n, seq_len)
        flat_attn = mcq["attention_mask"].reshape(bsz * max_n, seq_len)
        pv = mcq["pixel_values"]
        flat_pv = pv.unsqueeze(1).expand(bsz, max_n, *pv.shape[1:]).reshape(bsz * max_n, *pv.shape[1:])

        outputs = model(input_ids=flat_ids, attention_mask=flat_attn, pixel_values=flat_pv)
        logits = outputs.logits.view(bsz, max_n, seq_len, -1)

        losses: list[torch.Tensor] = []
        for b in range(bsz):
            pl = int(mcq["prompt_lens"][b].item())
            valid = mcq["option_valid"][b]
            logits_b = logits[b]
            ids_b = mcq["input_ids"][b]
            attn_b = mcq["attention_mask"][b]
            scores = mcq_mean_completion_logprobs_options(logits_b, ids_b, attn_b, prompt_len=pl)
            scores = scores.masked_fill(~valid.to(scores.device), -1e9)
            gold = mcq["gold_idx"][b : b + 1].to(scores.device)
            losses.append(F.cross_entropy(scores.unsqueeze(0).float(), gold))

        loss = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "mcq_contrastive" not in inputs:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


class TextOnlyMCQDataset(torch.utils.data.Dataset):
    """Text-only fine-tuning: no ``<image>`` token, no pixel_values."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        processor,
        text_only_fields: frozenset[str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.text_only_fields = frozenset() if text_only_fields is None else frozenset(text_only_fields)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.text_only_fields:
            prompt = build_prompt_text_only_mcq(row, fields=self.text_only_fields)
        else:
            prompt = build_prompt_ft(row, answer_prefix="Answer:").replace("<image>\n", "")

        ans_idx = int(row["answer"])
        completion = f" {CHOICE_LETTERS[ans_idx]}"
        full_text = prompt + completion

        n_completion = len(self.processor.tokenizer.encode(completion, add_special_tokens=False))

        enc = self.processor(text=full_text, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        labels = enc["input_ids"].clone()
        seq_len = int(labels.shape[0])
        n_sup = min(n_completion, seq_len)
        labels[:] = -100
        labels[-n_sup:] = enc["input_ids"][-n_sup:]

        if "attention_mask" in enc:
            labels[enc["attention_mask"] == 0] = -100

        enc["labels"] = labels
        return enc


@dataclass
class Collator:
    pad_to_multiple_of: int = 8
    pad_id: int = 0

    def __call__(self, features):
        batch = {}

        def _1d_long(x: torch.Tensor) -> torch.Tensor:
            x = x.squeeze()
            if x.dim() != 1:
                raise ValueError(f"expected 1D, got {tuple(x.shape)}")
            return x.long()

        input_ids = [_1d_long(f["input_ids"]) for f in features]
        attn = [_1d_long(f["attention_mask"]) for f in features]
        labels = [_1d_long(f["labels"]) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        labels = labels.masked_fill(attention_mask == 0, -100)

        if self.pad_to_multiple_of:
            m = input_ids.shape[1] % self.pad_to_multiple_of
            if m:
                extra = self.pad_to_multiple_of - m
                input_ids = torch.nn.functional.pad(input_ids, (0, extra), value=self.pad_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, extra), value=0)
                labels = torch.nn.functional.pad(labels, (0, extra), value=-100)

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels

        if "pixel_values" in features[0]:
            pv = torch.stack(
                [f["pixel_values"].squeeze(0) if f["pixel_values"].dim() == 4 else f["pixel_values"] for f in features]
            )
            batch["pixel_values"] = pv.to(dtype=torch.float16) if pv.dtype == torch.float32 else pv

        return batch


def _latest_checkpoint_dir(d: Path) -> Path | None:
    cands = []
    for p in d.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        suf = p.name.split("checkpoint-")[-1]
        try:
            step = int(suf)
        except ValueError:
            continue
        cands.append((step, p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def train_lora_adapter(
    lora_cfg_name: str,
    *,
    adapter_out_dir: Path,
    run_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    paths: Paths,
    seed: int,
    model_id: str = MODEL_ID_DEFAULT,
    train_n: int = 3000,
    val_n: int = 500,
    num_train_epochs: float = 1.0,
    ft_img_size: int = 192,
    text_only: bool = False,
    vision_only: bool = False,
    full_finetune: bool = False,
    make_submission: bool = True,
    run_validation_after_train: bool = True,
    save_steps: int = 25,
    eval_steps: int = 200,
    no_eval: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    resume: Literal["auto", "none"] = "auto",
    text_only_fields: frozenset[str] | None = None,
    train_captions_csv: Path | None = None,
    infer_captions_csv: Path | None = None,
    image_augment: bool = False,
    ft_metadata_fields: frozenset[str] | None = None,
    answer_prefix: str = "Answer:",
    submission_img_size: int = 224,
    ft_use_baseline_prompt: bool = True,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    mcq_contrastive: bool = False,
    mcq_completion_mode: str = "letter",
    mcq_choice_max_chars: int = 160,
    mcq_shuffle_choice_order: bool = False,
    mcq_shuffle_choice_seed: int = 42,
) -> Path:
    """Train with LoRA/DoRA (default) or full fine-tuning (--full-finetune).

    Saves:
    - Trainer checkpoints under data/runs/<run_name>/checkpoints/
    - Final weights under adapter_out_dir/ (PEFT adapter or full model dir)
    - run_meta.json under adapter_out_dir/
    - data/submission_<run_name>.csv (if make_submission)
    """
    if text_only and vision_only:
        raise SystemExit("Choose only one: --text-only or --vision-only")
    if mcq_contrastive and text_only:
        raise SystemExit("--mcq-contrastive requires multimodal training (omit --text-only).")
    if mcq_contrastive and mcq_completion_mode not in MCQ_COMPLETION_MODES:
        raise ValueError(f"Bad mcq_completion_mode {mcq_completion_mode!r}; allowed {MCQ_COMPLETION_MODES}")
    if mcq_shuffle_choice_order and not mcq_contrastive:
        raise SystemExit("--mcq-shuffle-choice-order applies only with --mcq-contrastive.")
    if full_finetune and vision_only:
        raise SystemExit("--vision-only uses LoRA on the vision tower; omit it when using --full-finetune.")
    if text_only_fields and not text_only:
        raise SystemExit("--text-only-fields requires --text-only")
    if text_only:
        if train_captions_csv is not None:
            raise SystemExit("--train-captions-csv applies only to multimodal training.")
        if image_augment:
            raise SystemExit("--image-augment applies only to multimodal training.")
        if ft_metadata_fields:
            raise SystemExit("--ft-metadata-fields applies only to multimodal training.")
        if answer_prefix.strip() != "Answer:":
            raise SystemExit("--answer-prefix applies only to multimodal training (text-only uses fixed cue).")

    caps_train = load_captions_csv(train_captions_csv) if train_captions_csv is not None else {}
    infer_caps_path = infer_captions_csv if infer_captions_csv is not None and infer_captions_csv.exists() else None

    torch.backends.cudnn.benchmark = True

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    cfg_from_model = AutoConfig.from_pretrained(model_id)
    num_layers = text_num_hidden_layers_from_config(cfg_from_model)
    recipe = None if full_finetune else make_cfg(lora_cfg_name, num_layers=num_layers)

    if full_finetune:
        ft_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            torch_dtype=ft_dtype,
            low_cpu_mem_usage=True,
        )
        print("Training mode: FULL FINE-TUNE (no LoRA/DoRA; all weights trainable).")
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            quantization_config=bnb,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)

        if vision_only:
            from peft import LoraConfig

            vision_cfg = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                exclude_modules=["model.text_model", "text_model", "lm_head"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, vision_cfg)
            assert recipe is not None
            recipe = type(recipe)(name=f"{lora_cfg_name}_vision_only", cfg=vision_cfg)
            print("LoRA target: VISION-ONLY modules")
        else:
            assert recipe is not None
            model = get_peft_model(model, recipe.cfg)

        model.is_loaded_in_8bit = True

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not full_finetune:
        assert trainable <= 5_000_000, f"Trainable params {trainable} exceeds 5M cap"
    print(f"Trainable params: {trainable:,}")

    run_dir = paths.data_dir / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_from_ckpt: Path | None = None
    if resume == "auto":
        resume_from_ckpt = _latest_checkpoint_dir(ckpt_dir)
        if resume_from_ckpt is not None:
            print("Resuming from checkpoint:", resume_from_ckpt.resolve())

    eval_strategy = "no" if no_eval else "steps"

    use_bf16 = bool(full_finetune and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=int(per_device_train_batch_size),
        per_device_eval_batch_size=int(per_device_eval_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        gradient_checkpointing=True,
        learning_rate=float(learning_rate),
        num_train_epochs=num_train_epochs,
        logging_steps=25,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if not no_eval else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        remove_unused_columns=False,
    )

    # Use full splits when train_n/val_n <= 0, otherwise sample for speed.
    if int(train_n) <= 0 or int(train_n) >= len(train_df):
        train_subset = train_df.reset_index(drop=True)
    else:
        train_subset = train_df.sample(n=int(train_n), random_state=seed).reset_index(drop=True)

    if int(val_n) <= 0 or int(val_n) >= len(val_df):
        val_subset = val_df.reset_index(drop=True)
    else:
        val_subset = val_df.sample(n=int(val_n), random_state=seed).reset_index(drop=True)

    if text_only:
        ds_train = TextOnlyMCQDataset(train_subset, processor=processor, text_only_fields=text_only_fields)
        ds_val = TextOnlyMCQDataset(val_subset, processor=processor, text_only_fields=text_only_fields)
        print("Training mode: TEXT-ONLY (no images).")
        if text_only_fields:
            print("  Text-only extra fields:", ",".join(sorted(text_only_fields)))
    else:
        if mcq_contrastive:
            ds_train = MCQContrastiveMultimodalDataset(
                train_subset,
                img_dir=paths.img_dir,
                processor=processor,
                img_size=ft_img_size,
                captions_by_id=caps_train if caps_train else None,
                augment=image_augment,
                ft_metadata_fields=ft_metadata_fields if ft_metadata_fields else None,
                answer_prefix=answer_prefix,
                use_baseline_prompt=ft_use_baseline_prompt,
                completion_mode=mcq_completion_mode,
                choice_max_chars=mcq_choice_max_chars,
                shuffle_choice_order_train=mcq_shuffle_choice_order,
                shuffle_choice_order_seed=mcq_shuffle_choice_seed,
            )
            ds_val = MCQContrastiveMultimodalDataset(
                val_subset,
                img_dir=paths.img_dir,
                processor=processor,
                img_size=ft_img_size,
                captions_by_id=caps_train if caps_train else None,
                augment=False,
                ft_metadata_fields=ft_metadata_fields if ft_metadata_fields else None,
                answer_prefix=answer_prefix,
                use_baseline_prompt=ft_use_baseline_prompt,
                completion_mode=mcq_completion_mode,
                choice_max_chars=mcq_choice_max_chars,
                shuffle_choice_order_train=False,
            )
            print(f"Training mode: MULTIMODAL MCQ CONTRASTIVE (ft_img_size={ft_img_size}).")
            if caps_train:
                print(f"  Train captions loaded: {len(caps_train)} ids from {train_captions_csv}")
            if image_augment:
                print("  Image augmentation: on (train only).")
            if ft_metadata_fields:
                print("  FT metadata fields:", ",".join(sorted(ft_metadata_fields)))
            if answer_prefix.strip() != "Answer:":
                print(f"  Answer cue: {answer_prefix!r}")
            print(f"  FT prompt: {'baseline(build_prompt)' if ft_use_baseline_prompt else 'short(build_prompt_ft)'}")
            print(f"  MCQ completion mode: {mcq_completion_mode!r} (must match validation scorer)")
            if mcq_shuffle_choice_order:
                print(f"  MCQ choice shuffle: on (train only; rng seed offset {mcq_shuffle_choice_seed})")
            if vision_only:
                print("Adapter mode: VISION-ONLY (text tower frozen except LoRA exclusion).")
        else:
            ds_train = LoRAMCQDataset(
                train_subset,
                img_dir=paths.img_dir,
                processor=processor,
                img_size=ft_img_size,
                captions_by_id=caps_train if caps_train else None,
                augment=image_augment,
                ft_metadata_fields=ft_metadata_fields if ft_metadata_fields else None,
                answer_prefix=answer_prefix,
                use_baseline_prompt=ft_use_baseline_prompt,
            )
            ds_val = LoRAMCQDataset(
                val_subset,
                img_dir=paths.img_dir,
                processor=processor,
                img_size=ft_img_size,
                captions_by_id=caps_train if caps_train else None,
                augment=False,
                ft_metadata_fields=ft_metadata_fields if ft_metadata_fields else None,
                answer_prefix=answer_prefix,
                use_baseline_prompt=ft_use_baseline_prompt,
            )
            print(f"Training mode: MULTIMODAL (ft_img_size={ft_img_size}).")
            if caps_train:
                print(f"  Train captions loaded: {len(caps_train)} ids from {train_captions_csv}")
            if image_augment:
                print("  Image augmentation: on (train only).")
            if ft_metadata_fields:
                print("  FT metadata fields:", ",".join(sorted(ft_metadata_fields)))
            if answer_prefix.strip() != "Answer:":
                print(f"  Answer cue: {answer_prefix!r}")
            print(f"  FT prompt: {'baseline(build_prompt)' if ft_use_baseline_prompt else 'short(build_prompt_ft)'}")
            if vision_only:
                print("Adapter mode: VISION-ONLY (text tower frozen except LoRA exclusion).")

    if mcq_contrastive and not text_only:
        collator = MCQContrastiveCollator(pad_token_id=int(processor.tokenizer.pad_token_id))
        trainer = MCQContrastiveTrainer(
            model=model,
            args=targs,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
        )
    else:
        collator = Collator(pad_to_multiple_of=8, pad_id=int(processor.tokenizer.pad_token_id))
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=collator,
        )
    train_result = trainer.train(resume_from_checkpoint=str(resume_from_ckpt) if resume_from_ckpt else None)

    adapter_out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_out_dir)

    meta: dict[str, Any] = {
        "run_name": run_name,
        "model_id": model_id,
        "seed": int(seed),
        "ft_img_size": int(ft_img_size) if not text_only else None,
        "train_n": int(len(train_subset)),
        "val_n": int(len(val_subset)),
        "num_train_epochs": float(num_train_epochs),
        "trainable_params": int(trainable),
        "checkpoint_dir": str(ckpt_dir),
        "cfg_name": recipe.name if recipe is not None else "full_finetune",
        "full_finetune": bool(full_finetune),
        "text_only": bool(text_only),
        "text_only_fields": sorted(text_only_fields) if text_only_fields else [],
        "vision_only": bool(vision_only),
        "multimodal_prompt": {
            "answer_prefix": answer_prefix if not text_only else None,
            "ft_metadata_fields": sorted(ft_metadata_fields) if ft_metadata_fields else [],
            "train_captions_csv": str(train_captions_csv) if train_captions_csv is not None else None,
            "infer_captions_csv": str(infer_captions_csv) if infer_captions_csv is not None else None,
            "image_augment": bool(image_augment) if not text_only else False,
            "submission_img_size": int(submission_img_size),
            "mcq_contrastive": bool(mcq_contrastive and not text_only),
            "mcq_completion_mode": (mcq_completion_mode if (mcq_contrastive and not text_only) else None),
            "mcq_choice_max_chars": int(mcq_choice_max_chars) if (mcq_contrastive and not text_only) else None,
            "mcq_shuffle_choice_order": bool(mcq_shuffle_choice_order) if (mcq_contrastive and not text_only) else False,
            "mcq_shuffle_choice_seed": int(mcq_shuffle_choice_seed) if (mcq_shuffle_choice_order and mcq_contrastive) else None,
        },
        "lora": (
            {
                "r": int(recipe.cfg.r),
                "lora_alpha": int(getattr(recipe.cfg, "lora_alpha", 0) or 0),
                "use_dora": bool(getattr(recipe.cfg, "use_dora", False)),
                "target_modules": list(recipe.cfg.target_modules) if hasattr(recipe.cfg, "target_modules") else None,
                "layers_to_transform": (
                    list(recipe.cfg.layers_to_transform)
                    if getattr(recipe.cfg, "layers_to_transform", None) is not None
                    else None
                ),
                "layers_pattern": getattr(recipe.cfg, "layers_pattern", None),
            }
            if recipe is not None
            else None
        ),
        "train": {
            "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
            "train_loss": float(getattr(train_result, "training_loss", float("nan"))),
        },
    }
    save_json(adapter_out_dir / "run_meta.json", meta)

    # --- Run validation after training (faster than full submission) ---
    if run_validation_after_train:
        try:
            from .eval_adapters import evaluate_adapter_dir

            print("Running validation after training...")
            # Free training-time GPU state before re-loading a fresh inference model
            del trainer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            infer_mode = mcq_completion_mode if (mcq_contrastive and not text_only) else "letter"

            val_eval_meta = evaluate_adapter_dir(
                adapter_dir=adapter_out_dir,
                run_name=run_name,
                val_df=val_df,
                paths=paths,
                model_id=model_id,
                val_n=val_n,
                img_size=submission_img_size,
                captions_csv=infer_caps_path,
                shuffle_seed=seed,
                mcq_completion_mode=infer_mode,
                mcq_choice_max_chars=mcq_choice_max_chars,
            )
            meta["val_eval_after_train"] = val_eval_meta
            save_json(adapter_out_dir / "run_meta.json", meta)
            if "val_acc" in val_eval_meta:
                print(f"Validation accuracy after training: {val_eval_meta['val_acc']:.4f}")
        except Exception as e:
            print("Warning: post-train validation failed:", repr(e))

    if make_submission:
        try:
            from .make_submission import make_submission_from_adapter

            print("Making full submission...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            infer_mode = mcq_completion_mode if (mcq_contrastive and not text_only) else "letter"

            submission_path = make_submission_from_adapter(
                run_name=run_name,
                adapter_dir=adapter_out_dir,
                test_df=test_df,
                paths=paths,
                model_id=model_id,
                img_size=submission_img_size,
                captions_csv=infer_caps_path,
                mcq_completion_mode=infer_mode,
                mcq_choice_max_chars=mcq_choice_max_chars,
            )
            meta["submission_csv"] = str(submission_path)
            save_json(adapter_out_dir / "run_meta.json", meta)
            print("Saved submission to", submission_path.resolve())
        except Exception as e:
            print("Warning: submission generation failed:", repr(e))

    print("Saved adapter to", adapter_out_dir.resolve())
    print("Saved run metadata to", (adapter_out_dir / "run_meta.json").resolve())
    print("Saved checkpoints under", ckpt_dir.resolve())

    return adapter_out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-id", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--adapter-out", type=Path, required=True)
    ap.add_argument(
        "--full-finetune",
        action="store_true",
        default=False,
        help="Train all weights (no LoRA/DoRA). Loads bf16/fp16 SmolVLM; needs much more GPU memory than QLoRA.",
    )
    ap.add_argument(
        "--cfg",
        type=str,
        choices=[
            "section9_lora",
            "section10_dora",
            "section12_lora",
            "attn_only_lora",
            "attn_mlp_lora",
            "attn_mlp_dora",
            "mlp_only_lora",
            "attn_mlp_lora_r2",
            "attn_mlp_lora_r1_last16",
            "attn_mlp_lora_r2_last16",
            "attn_mlp_lora_r4_last16",
            "attn_mlp_lora_r2_last24",
            "attn_mlp_dora_r2_last16",
            "attn_mlp_dora_r2_last24",
        ],
        default=None,
        help="LoRA recipe (required unless --full-finetune). Ignored when --full-finetune.",
    )

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--train-n", type=int, default=3000)
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--ft-img-size", type=int, default=192)
    ap.add_argument("--text-only", action="store_true", default=False, help="Fine-tune on text only (no images).")
    ap.add_argument(
        "--text-only-fields",
        type=str,
        default=None,
        help="With --text-only only: comma-separated context fields from CSV (hint, grade, lecture). Example: hint,grade",
    )
    ap.add_argument(
        "--vision-only",
        action="store_true",
        default=False,
        help="Fine-tune only vision tower modules with LoRA.",
    )
    ap.add_argument("--make-submission", action="store_true", default=True)
    ap.add_argument("--no-make-submission", action="store_false", dest="make_submission")
    ap.add_argument(
        "--run-validation-after-train",
        action="store_true",
        default=True,
        help="Run validation immediately after training to see score.",
    )
    ap.add_argument(
        "--no-run-validation-after-train",
        action="store_false",
        dest="run_validation_after_train",
        help="Skip validation after training (faster for long runs).",
    )
    ap.add_argument("--save-steps", type=int, default=25, help="Checkpoint save frequency (steps).")
    ap.add_argument("--eval-steps", type=int, default=200, help="Eval frequency (steps).")
    ap.add_argument("--no-eval", action="store_true", default=False, help="Disable eval during training (faster).")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (speed up preprocessing).")
    ap.add_argument("--pin-memory", action="store_true", default=True)
    ap.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    ap.add_argument(
        "--resume",
        type=str,
        default="auto",
        choices=["auto", "none"],
        help="Resume training. 'auto' resumes from latest checkpoint if present.",
    )
    ap.add_argument(
        "--train-captions-csv",
        type=Path,
        default=None,
        help="Optional id,caption CSV; injects Image description during multimodal training.",
    )
    ap.add_argument(
        "--infer-captions-csv",
        type=Path,
        default=None,
        help="Optional id,caption CSV for post-train val + submission (matches scoring.py).",
    )
    ap.add_argument(
        "--image-augment",
        action="store_true",
        default=False,
        help="Light ColorJitter + RandomAffine on train images (multimodal only).",
    )
    ap.add_argument(
        "--ft-metadata-fields",
        type=str,
        default=None,
        help="Multimodal FT only: comma-separated grade,subject,topic,hint,lecture injected into short prompt.",
    )
    ap.add_argument(
        "--answer-prefix",
        type=str,
        default="Answer:",
        help='Answer cue line before the completion (multimodal FT only), e.g. "The correct answer is:"',
    )
    ap.add_argument(
        "--submission-img-size",
        type=int,
        default=224,
        help="Resize for post-train validation + submission inference.",
    )
    ap.add_argument(
        "--ft-use-baseline-prompt",
        action="store_true",
        default=True,
        help="Multimodal FT: train with common.build_prompt (lecture+hint context-first) to match eval.",
    )
    ap.add_argument(
        "--ft-use-short-prompt",
        action="store_false",
        dest="ft_use_baseline_prompt",
        help="Multimodal FT: train with the short prompt (build_prompt_ft).",
    )
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--train-bsz", type=int, default=1, help="per-device train batch size")
    ap.add_argument("--eval-bsz", type=int, default=1, help="per-device eval batch size")
    ap.add_argument("--grad-accum", type=int, default=16, help="gradient accumulation steps")
    ap.add_argument(
        "--mcq-contrastive",
        action="store_true",
        default=False,
        help="Multimodal only: train with CE over per-option log-likelihood scores (matches MCQ inference).",
    )
    ap.add_argument(
        "--mcq-completion-mode",
        type=str,
        default="letter",
        choices=sorted(MCQ_COMPLETION_MODES),
        help="Suffix after Answer: for scoring during train/eval (letter matches default build_prompt inference).",
    )
    ap.add_argument(
        "--mcq-choice-max-chars",
        type=int,
        default=160,
        help="Truncate choice text for letter_choice / choice_text completion modes.",
    )
    ap.add_argument(
        "--mcq-shuffle-choice-order",
        action="store_true",
        default=False,
        help="With --mcq-contrastive: randomly permute A,B,… choice rows each train sample (train only).",
    )
    ap.add_argument(
        "--mcq-shuffle-choice-seed",
        type=int,
        default=42,
        help="Base seed mixed with sample index for deterministic choice-order shuffles per worker.",
    )

    args = ap.parse_args()

    paths = get_paths(args.root)
    set_seed(args.seed)

    train_df, val_df, test_df = load_csvs(paths.data_dir)

    try:
        text_only_fields_parsed = parse_text_only_fields(args.text_only_fields)
    except ValueError as e:
        raise SystemExit(str(e))
    if text_only_fields_parsed and not args.text_only:
        raise SystemExit("--text-only-fields requires --text-only")

    try:
        ft_meta_parsed = parse_ft_metadata_fields(args.ft_metadata_fields)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.full_finetune and args.vision_only:
        raise SystemExit("Cannot combine --full-finetune with --vision-only.")
    if not args.full_finetune and args.cfg is None:
        raise SystemExit("--cfg is required unless --full-finetune.")

    train_lora_adapter(
        lora_cfg_name=args.cfg or "full_finetune",
        adapter_out_dir=args.adapter_out,
        run_name=args.run_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        paths=paths,
        seed=args.seed,
        model_id=args.model_id,
        train_n=args.train_n,
        val_n=args.val_n,
        num_train_epochs=args.epochs,
        ft_img_size=args.ft_img_size,
        text_only=args.text_only,
        vision_only=args.vision_only,
        full_finetune=args.full_finetune,
        make_submission=args.make_submission,
        run_validation_after_train=args.run_validation_after_train,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        no_eval=args.no_eval,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        resume=args.resume,
        text_only_fields=text_only_fields_parsed if text_only_fields_parsed else None,
        train_captions_csv=args.train_captions_csv,
        infer_captions_csv=args.infer_captions_csv,
        image_augment=args.image_augment,
        ft_metadata_fields=ft_meta_parsed if ft_meta_parsed else None,
        answer_prefix=args.answer_prefix,
        submission_img_size=args.submission_img_size,
        ft_use_baseline_prompt=bool(args.ft_use_baseline_prompt),
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.train_bsz),
        per_device_eval_batch_size=int(args.eval_bsz),
        gradient_accumulation_steps=int(args.grad_accum),
        mcq_contrastive=bool(args.mcq_contrastive),
        mcq_completion_mode=str(args.mcq_completion_mode),
        mcq_choice_max_chars=int(args.mcq_choice_max_chars),
        mcq_shuffle_choice_order=bool(args.mcq_shuffle_choice_order),
        mcq_shuffle_choice_seed=int(args.mcq_shuffle_choice_seed),
    )


if __name__ == "__main__":
    main()

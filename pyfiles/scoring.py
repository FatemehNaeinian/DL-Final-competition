from __future__ import annotations

import pandas as pd
import torch
from PIL import Image

from .common import CHOICE_LETTERS, build_prompt, mcq_completion_suffix


def mcq_mean_completion_logprobs_options(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prompt_len: int,
) -> torch.Tensor:
    """Mean token log-probability over completion tokens per option row.

    Matches :func:`predict_mcq_index` (same ``start_pos`` / ``end_pos`` slicing).
    ``logits``: ``(num_options, seq_len, vocab)``; returns shape ``(num_options,)``.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    attn_tail = attention_mask[:, 1:].to(dtype=shift_logits.dtype)

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * attn_tail

    start_pos = max(prompt_len - 1, 0)
    device = token_logps.device
    dtype = token_logps.dtype
    seq_lens = attention_mask.sum(dim=1).long()

    scores: list[torch.Tensor] = []
    for i in range(input_ids.shape[0]):
        sl = int(seq_lens[i].item())
        if sl <= prompt_len:
            scores.append(torch.tensor(-1e9, device=device, dtype=dtype))
            continue
        slice_lp = token_logps[i, start_pos : sl - 1]
        denom = slice_lp.numel() if slice_lp.numel() > 0 else 1
        scores.append(slice_lp.sum() / float(denom))
    return torch.stack(scores, dim=0)


@torch.inference_mode()
def score_mcq_options(
    model,
    processor,
    row: pd.Series,
    image: Image.Image,
    *,
    caption: str | None = None,
    completion_mode: str = "letter",
    choice_max_chars: int = 160,
) -> torch.Tensor:
    """Return per-option mean completion logprobs (higher is better), shape (num_options,)."""
    prompt = build_prompt(row, include_answer=False)
    if caption:
        head = "<image>\n"
        if prompt.startswith(head):
            rest = prompt[len(head) :]
            prompt = f"{head}Image description: {caption}\n{rest}"
        else:
            prompt = f"Image description: {caption}\n{prompt}"

    n = int(row["num_choices"]) if "num_choices" in row else len(row["choices"])
    n = min(n, len(CHOICE_LETTERS))

    completions = [
        mcq_completion_suffix(row, i, completion_mode, choice_max_chars=choice_max_chars) for i in range(n)
    ]
    full_texts = [prompt + c for c in completions]
    images = [image] * n

    inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    inp_prompt = processor(text=prompt, images=image, return_tensors="pt")
    prompt_len = int(inp_prompt["input_ids"].shape[1])

    outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids)

    return mcq_mean_completion_logprobs_options(logits, input_ids, attn_mask, prompt_len=prompt_len)


@torch.inference_mode()
def predict_mcq_index(
    model,
    processor,
    row: pd.Series,
    image: Image.Image,
    *,
    caption: str | None = None,
    completion_mode: str = "letter",
    choice_max_chars: int = 160,
) -> int:
    """Batched log-likelihood scoring. Returns 0-indexed answer index.

    If `caption` is provided, it is injected into the prompt as an "Image description" line.
    """
    scores_t = score_mcq_options(
        model,
        processor,
        row,
        image,
        caption=caption,
        completion_mode=completion_mode,
        choice_max_chars=choice_max_chars,
    )
    return int(scores_t.argmax().item())


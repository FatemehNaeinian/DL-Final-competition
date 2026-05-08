#!/usr/bin/env python3
"""Verify conda deps for QLoRA training (run on a GPU node to confirm CUDA)."""

from __future__ import annotations

import sys


def main() -> int:
    print("Python:", sys.version.split()[0])

    import torch

    print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))

    import transformers

    print("transformers:", transformers.__version__)

    import peft

    print("peft:", peft.__version__)

    import accelerate

    print("accelerate:", accelerate.__version__)

    # bitsandbytes can take a long time on first import without a GPU; still verify when CUDA exists.
    if torch.cuda.is_available():
        import bitsandbytes as bnb

        print("bitsandbytes:", getattr(bnb, "__version__", "unknown"))
    else:
        print("bitsandbytes: skipped (run this script on a GPU node to validate import/init)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

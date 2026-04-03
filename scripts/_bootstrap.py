from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WANDB_PROJECT = "simclr_hard_positive"
WANDB_ENTITY = "zihengkang_astribot"
WANDB_MODE = "online"
WANDB_API_KEY = "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"
WANDB_DIR = REPO_ROOT / "wandb"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
os.environ.setdefault("WANDB_MODE", WANDB_MODE)
os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY)
os.environ.setdefault("WANDB_DIR", str(WANDB_DIR))

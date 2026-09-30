"""复现性工具：配置读取、全局种子、PROVENANCE.json。每个分析 stage 都调用。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "config" / "run.yaml") as fh:
        return yaml.safe_load(fh)


def config_sha() -> str:
    return hashlib.sha256((ROOT / "config" / "run.yaml").read_bytes()).hexdigest()[:16]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def write_provenance(stage: str, inputs: list[Path], outputs: list[Path], seed: int,
                     extra: dict | None = None) -> Path:
    out_dir = ROOT / "results" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "torch_geometric", "scanpy"):
        try:
            pkgs[name] = __import__(name).__version__
        except Exception:
            pass
    rec = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "config_sha": config_sha(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "inputs": {str(Path(p).relative_to(ROOT) if str(p).startswith(str(ROOT)) else p): md5(Path(p))
                   for p in inputs if Path(p).is_file()},
        "outputs": {str(Path(p).relative_to(ROOT)): md5(Path(p)) for p in outputs if Path(p).is_file()},
        "extra": extra or {},
    }
    path = out_dir / "PROVENANCE.json"
    path.write_text(json.dumps(rec, indent=2))
    return path

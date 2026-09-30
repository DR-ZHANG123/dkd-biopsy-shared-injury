"""冻结校验：冻结提交时已存在的每个 config 顶层段必须与当前完全一致；冻结后新增的段不影响冻结设计。"""
from __future__ import annotations

import json
import subprocess

import yaml

from lib.repro import ROOT

FREEZE_COMMIT = "f3b8611"


def check_frozen() -> dict:
    frozen = json.loads((ROOT / "results/15_model/FROZEN_DESIGN.json").read_text())
    cur = yaml.safe_load((ROOT / "config/run.yaml").read_text())
    try:  # 私有开发仓库中可用冻结提交做逐段比对；公开仓库无该提交时只比对冻结设计本身
        old = yaml.safe_load(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{FREEZE_COMMIT}:config/run.yaml"],
                                                     stderr=subprocess.DEVNULL))
    except (subprocess.CalledProcessError, FileNotFoundError):
        old = None
    if old is not None:
        changed = [k for k in old if cur.get(k) != old[k]]
        if changed:
            raise SystemExit(f"冻结后改动了已冻结的 config 段：{changed}，拒绝运行")
    if cur["model2"]["defaults"] != frozen["design"]:
        raise SystemExit("model2.defaults 与 FROZEN_DESIGN.json 不一致，拒绝运行")
    return frozen

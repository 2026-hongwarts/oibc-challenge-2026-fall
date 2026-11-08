"""2차 배치: 시드 잡음 측정(관문 보정) + F5 탐색에서 찾은 새 축(X17 달력, X18 패널 오염, X19 눈 덮임).

    python analysis_lab/X_추가_배치.py         (저장소 맨 위에서, bio_proj 환경. 1차 배치가 끝난 뒤)

N1x2·N2x2 = 기준 모델 E7-00x2 와 설정이 같고 시드만 다름(2·3, 4·5). 기준 대비 차이가 곧 '아무것도 안 바꿨을 때의 흔들림'.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
from compare_years import penalty  # noqa: E402

PY = sys.executable
BASE = "E7-00x2"
CUTS = [None, "2023-06-01", "2022-06-01"]
LOG = ROOT / "outputs" / "work" / "xbatch"
LOG.mkdir(parents=True, exist_ok=True)
V4 = "features_v4:"
EXPS: list[tuple[str, list[str]]] = [
    ("N1x2", ["--seed", "2"]),
    ("N2x2", ["--seed", "4"]),
    ("X17x2", ["--add", V4 + "add_calendar"]),
    ("X19x2", ["--add", V4 + "add_snow_persist"]),
    ("X18x2", ["--add", V4 + "add_days_since_rain"]),
]
t0 = time.time()


def out_exists(name, cut):
    return (ROOT / "outputs" / "preds" / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")).exists()


def main() -> None:
    for name, args in EXPS:
        for c in CUTS:
            if out_exists(name, c):
                continue
            cmd = [PY, "src_tools/train_site_model.py", "--exp", name, "--base", BASE, *args] + (["--cut", c] if c else [])
            with open(LOG / f"{name}_{c[:4] if c else 'valid'}.log", "w", encoding="utf-8") as f:
                r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if r.returncode != 0:
                print(f"  !! 실패 {name} {c}", flush=True)
        vals = []
        for c in CUTS:
            v, b = penalty(name, c), penalty(BASE, c)
            vals.append("   -   " if v is None or b is None else f"{(v[0] / b[0] - 1) * 100:+.2f}%")
        print(f"[{time.time() - t0:5.0f}s] {name:8s} {' / '.join(vals)}", flush=True)


if __name__ == "__main__":
    main()

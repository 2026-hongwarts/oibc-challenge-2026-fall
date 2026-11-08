"""피처 엔지니어링 계획서 v4 의 미시험(⬜) 항목을 한 번에 시험한다 (X01 ~ X16).

    python analysis_lab/X_미시험_배치.py              (저장소 맨 위에서, bio_proj 환경)

각 실험 = 기준 모델 E7-00x2 + 변경 하나, 시드 2개, 세 해(2024-25 / 2023-24 / 2022-23).
이미 결과 파일이 있는 실행은 건너뛴다(중간에 끊겨도 다시 돌리면 이어서 함).
로그: outputs/work/xbatch/*.log, 비교표: outputs/work/xbatch/summary.md
"""
from __future__ import annotations

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
V3 = "features_v3:"

EXPS: list[tuple[str, list[str]]] = [
    ("X16x2", ["--add", V3 + "add_mos"]),
    ("X16bx2", ["--add", V3 + "add_mos_bin"]),
    ("X16cx2", ["--add", V3 + "add_mos_daily"]),
    ("X15sx2", ["--fit-where", "wa_cloud_r17 >= 90"]),          # 흐림 전용 모델 (단독 채점은 의미 없음)
    ("X11x2", ["--add", V3 + "add_irr_celltemp"]),
    ("X10x2", ["--add", V3 + "add_clearsky_phys"]),
    ("X02ax2", ["--reset-add", "--add", V3 + "daily_no_today"]),
    ("X02bx2", ["--reset-add", "--add", V3 + "daily_no_prev"]),
    ("X02cx2", ["--reset-add", "--add", V3 + "daily_no_am"]),
    ("X14x2", ["--target", "csi"]),
    ("X01x2", ["--add", V3 + "add_raw_extra"]),
    ("X03x2", ["--add", V3 + "add_hour_neighbors"]),
    ("X04x2", ["--add", V3 + "add_prev2_delta"]),
    ("X05x2", ["--add", V3 + "add_night_snow"]),
    ("X06x2", ["--add", V3 + "add_day_humid_wind"]),
    ("X07x2", ["--add", V3 + "add_fleet"]),
    ("X08x2", ["--add", V3 + "add_near_stations"]),
    ("X09x2", ["--add", V3 + "add_round_prov_hourly"]),
    ("X12x2", ["--add", V3 + "add_static_more"]),
    ("X13ax2", ["--drop-dead"]),
    ("X13bx2", ["--add", V3 + "mark_partial_outage", "--drop-dead"]),
]
REGIME = [("X15x2", 0.5), ("X15fx2", 1.0)]      # 흐림 구간에서 전용 모델 비중
WHERE = "wa_cloud_r17 >= 90"

t0 = time.time()


def out_exists(name: str, cut: str | None) -> bool:
    return (ROOT / "outputs" / "preds" / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")).exists()


def call(cmd: list[str], logname: str) -> None:
    with open(LOG / f"{logname}.log", "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        print(f"  !! 실패 {logname} (로그 {LOG / (logname + '.log')})", flush=True)


def train(name: str, args: list[str], cut: str | None, base_run: bool = False) -> None:
    if out_exists(name, cut):
        return
    cmd = [PY, "src_tools/train_site_model.py"] + (["--base", BASE] if base_run else ["--exp", name, "--base", BASE] + args)
    if cut:
        cmd += ["--cut", cut]
    call(cmd, f"{name}_{cut[:4] if cut else 'valid'}")


def line(name: str) -> str:
    out = []
    for c in CUTS:
        v, b = penalty(name, c), penalty(BASE, c)
        out.append("   -   " if v is None or b is None else f"{(v[0] / b[0] - 1) * 100:+.2f}%")
    return " / ".join(out)


def main() -> None:
    for c in CUTS[1:]:
        train(BASE, [], c, base_run=True)
    print(f"[{time.time() - t0:5.0f}s] 기준 {BASE} 세 해 준비", flush=True)
    for name, args in EXPS:
        for c in CUTS:
            train(name, args, c)
        if name == "X15sx2":
            for out, w in REGIME:
                for c in CUTS:
                    if out_exists(out, c):
                        continue
                    cmd = [PY, "src_tools/blend_regime.py", "--base", BASE, "--spec", name, "--where", WHERE,
                           "--w", str(w), "--out", out] + (["--cut", c] if c else [])
                    call(cmd, f"{out}_{c[:4] if c else 'valid'}")
                print(f"[{time.time() - t0:5.0f}s] {out:8s} {line(out)}", flush=True)
        else:
            print(f"[{time.time() - t0:5.0f}s] {name:8s} {line(name)}", flush=True)
    names = [BASE] + [n for n, _ in EXPS if n != "X15sx2"] + [o for o, _ in REGIME]
    subprocess.run([PY, "src_tools/compare_years.py", *names, "--md", str(LOG / "summary.md")], cwd=ROOT,
                   env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    main()

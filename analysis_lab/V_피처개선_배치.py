"""피처 개선 배치 (11/9 새벽) — 기준 E9-00x2(LightGBM, A 기압 포함) 위에 변경 하나씩, 시드 2개, 세 해.

    python analysis_lab/V_피처개선_배치.py --set stage1      F9 와 무관하게 정한 변형 (범주형, 운고)
    python analysis_lab/V_피처개선_배치.py --set stage2      F9 결과로 정한 변형 (빼기)

판정(시험 전에 고정): 세 해 모두 E9-00x2 보다 벌금이 낮으면 1차 통과 → CSI-XGBoost 에도 적용해 섞은 결과로 최종 비교,
다른 시드(2·3)로 재현. 이미 결과 파일이 있으면 건너뛴다.
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
BASE = "E9-00x2"
CUTS = [None, "2023-06-01", "2022-06-01"]
LOG = ROOT / "outputs" / "work" / "xbatch"
LOG.mkdir(parents=True, exist_ok=True)
V05 = ["wa_wind_speed", "wb_wind_speed", "wb_vis", "wb_total_liq", "wb_precip_prob",
       "wx_prev_cloud", "wx_prev_rain", "wx_prev_trange", "wx_day_rain", "wx_am_vis_min", "wx_day_trange"]
V06 = ["cosz", "wb_dew_point_r10", "wb_temp_r10", "wb_rel_hum_r10", "wa_ground_press_r10", "wa_temp_r10"]
SETS: dict[str, list[tuple[str, list[str]]]] = {
    "stage1": [
        ("V01x2", ["--cat", "hour"]),                                   # 시각 범주형
        ("V04x2", ["--add", "features_v4:add_ceiling"]),               # 운고 단독 추가
        ("V02x2", ["--cat", "site_id"]),                                # 발전소 ID 범주형
        ("V03x2", ["--cat", "wa_station", "--cat", "wb_station"]),     # 예보 지점 ID 범주형
    ],
    # F9 결과로 정함 (시험 전에 고정한 기준)
    #   V05 기여 없음: 세 fold 평균 순열 중요도 ≤ 0.1% 이고 최소 ≤ 0 인 묶음 11개
    #   V06 중복: 순위 상관 ≥ 0.995 (cosz 는 elev 와 순위 동일, 10시 발행분 5개는 17시와 거의 같음)
    "stage2": [
        ("V05x2", sum([["--drop", d] for d in V05], [])),
        ("V06x2", sum([["--drop", d] for d in V06], [])),
        ("V07x2", sum([["--drop", d] for d in V05 + V06], [])),
    ],
}
t0 = time.time()


def out_exists(name, cut):
    return (ROOT / "outputs" / "preds" / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")).exists()


def main() -> None:
    which = sys.argv[sys.argv.index("--set") + 1]
    for name, args in SETS[which]:
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

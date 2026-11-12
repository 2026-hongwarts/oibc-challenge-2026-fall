"""GFS 0.25° 1시간 일사량 상자(gfs25_fetch.py) → 시각별 고정 지역 피처 outputs/work/gfs25_kr.parquet.

    python src_tools/gfs25_decode.py

1. 공개 시각이 실행일 08:00 UTC(= 대상일 전날 17:00 KST) 보다 늦은 파일은 빈칸(규정 §6, manifest_*.csv).
2. GFS DSWRF 는 6시간마다 다시 시작하는 구간 평균이다(f019 = 18 ~ 19시 평균, f020 = 18 ~ 20시 평균 … f025 = 24 ~ 25시 …).
   구간 시작 s(18, 24, 30), 예보 시각 k 의 1시간 평균 = (k − s)·A_k − (k − 1 − s)·A_{k−1}  (k − 1 = s 이면 A_s = 0).
3. ts = 실행일 00 UTC + k 시간(그 1시간의 끝) 을 KST 로. f019 ~ f035 = 대상일 KST 04 ~ 20시.
4. 피처(모든 발전소에 같은 값, §7): g25_sw_r{위도}_{경도} 1° 지역 42곳 평균, g25_sw_kr 상자 평균.
   이름에 'dswrf' 를 넣지 않은 것은 features_gfs 의 3시간 → 1시간 cos 배분을 타지 않게 하려는 것(이미 1시간 값).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CROP = ROOT / "outputs" / "work" / "gfs25_crop"
OUT = ROOT / "outputs" / "work" / "gfs25_kr.parquet"


def main() -> None:
    man = pd.concat([pd.read_csv(f, dtype={"run_date": str}) for f in sorted(glob.glob(str(CROP / "manifest_*.csv")))
                     if Path(f).stat().st_size > 100],                           # 받는 중이라 아직 빈 파일은 건너뜀
                    ignore_index=True).drop_duplicates(["run_date", "fhour"], keep="last")
    lm = pd.to_datetime(man.last_modified_utc, utc=True)
    deadline = pd.to_datetime(man.run_date, format="%Y%m%d").dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    late = set(zip(man.run_date[lm > deadline], man.fhour[lm > deadline]))
    print(f"마감 뒤 공개되어 뺀 (실행일, 예보 시각) {len(late)}개")
    files = sorted(CROP.glob("*.npz"))
    z0 = np.load(files[0])
    la, lo = z0["lat"], z0["lon"]
    rlat = np.minimum(np.floor(la), 38).astype(int)
    rlon = np.minimum(np.floor(lo), 130).astype(int)
    regions = sorted(set(zip(rlat.ravel(), rlon.ravel())))
    masks = {r: (rlat == r[0]) & (rlon == r[1]) for r in regions}
    rows = []
    for f in files:
        day = f.stem
        try:
            z = np.load(f)
            fh, A = list(z["fh"]), z["v"].astype("float64")
        except Exception as e:                       # 쓰다 만 파일
            print(f"  읽기 실패 {f.name}: {e}")
            continue
        for j, k in enumerate(fh):
            if (day, k) in late:
                A[j] = np.nan
        R = pd.Timestamp(day, tz="UTC")
        for j, k in enumerate(fh):
            s = 6 * ((k - 1) // 6)                                # 구간 시작(18, 24, 30)
            if k - 1 == s:
                h = A[j]
            else:
                if j == 0 or fh[j - 1] != k - 1:
                    continue
                h = (k - s) * A[j] - (k - 1 - s) * A[j - 1]
            h = np.clip(h, 0, None)
            feats = {f"g25_sw_r{r[0]}_{r[1]}": float(np.nanmean(h[m])) if np.isfinite(h[m]).any() else np.nan
                     for r, m in masks.items()}
            feats["g25_sw_kr"] = float(np.nanmean(h)) if np.isfinite(h).any() else np.nan
            rows.append({"ts": (R + pd.Timedelta(hours=int(k))).tz_convert("Asia/Seoul"), **feats})
    t = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    t["ts"] = t.ts.dt.as_unit("ns")
    t.to_parquet(OUT, index=False)
    print(f"저장 {OUT.relative_to(ROOT)}: {len(t):,}시각 × {t.shape[1] - 1}피처, {t.ts.min()} ~ {t.ts.max()}, 빈칸 비율 {t.g25_sw_kr.isna().mean():.2%}")


if __name__ == "__main__":
    main()

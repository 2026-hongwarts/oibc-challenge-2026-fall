"""GFS 원본 메시지(gfs_fetch.py) → 시각별 고정 지역 피처 표 outputs/work/gfs_kr.parquet.  ※ gfs 환경(pygrib)에서 실행

    conda run -n gfs python src_tools/gfs_decode.py

1. 한반도 남부 고정 상자 북위 33 ~ 39°, 동경 124 ~ 131° (0.5° 격자 13 × 15 = 195칸)만 꺼낸다.
   상자와 지역 경계는 발전소 위치·용량과 무관하게 위경도 1° 눈금으로 미리 정했다(규정 §7).
2. GFS 의 일사량·운량은 6시간마다 다시 시작하는 '구간 평균'이다.
   f021 = 18 ~ 21시 평균, f024 = 18 ~ 24시 평균 → 21 ~ 24시 평균 = 2 × f024 − f021. (24 ~ 30, 30 ~ 36 도 같은 방식)
   → 3시간 블록 6개: UTC 실행일 R 기준 예보 18 ~ 21, 21 ~ 24, 24 ~ 27, 27 ~ 30, 30 ~ 33, 33 ~ 36시
3. 블록을 대상일 D = R + 1 의 한국 시각(ts = 그 1시간의 끝)에 붙인다.
   예: 예보 24 ~ 27시 = UTC (R+1) 00 ~ 03시 = KST D 09 ~ 12시 → ts 10, 11, 12시.
4. 피처 (모든 발전소에 같은 값):
   gfs_{dswrf|tcdc}_r{위도}_{경도}  1° 지역(그 칸의 0.5° 격자 평균) 42곳
   gfs_{dswrf|tcdc}_kr               상자 전체 평균
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pygrib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "work" / "gfs_raw"
OUT = ROOT / "outputs" / "work" / "gfs_kr.parquet"
BOX = dict(lat1=33.0, lat2=39.0, lon1=124.0, lon2=131.0)
BLOCKS = [(18, 21, None), (21, 24, 21), (24, 27, None), (27, 30, 27), (30, 33, None), (33, 36, 33)]


def read(path: Path):
    g = pygrib.fromstring(path.read_bytes())          # 경로에 한글이 있으면 pygrib.open 이 실패하므로 메모리에서 해석
    v, la, lo = g.data(**BOX)
    return np.asarray(v, dtype="float64"), la, lo, g.stepRange


def main() -> None:
    files = sorted(RAW.glob("*/*.grb2"))
    pat = re.compile(r"(\d{8})_f(\d{3})_(DSWRF|TCDC)\.grb2")
    store: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    la = lo = None
    bad = 0
    for p in files:
        m = pat.match(p.name)
        if not m:
            continue
        day, fh, var = m.group(1), int(m.group(2)), m.group(3)
        try:
            v, la, lo, step = read(p)
        except Exception:
            bad += 1
            continue
        a, b = map(int, step.split("-"))
        expect = {21: (18, 21), 24: (18, 24), 27: (24, 27), 30: (24, 30), 33: (30, 33), 36: (30, 36)}[fh]
        if (a, b) != expect:
            raise ValueError(f"{p.name}: 평균 구간 {step}, 기대 {expect}")
        store.setdefault((day, var), {})[fh] = v
    print(f"읽은 파일 {len(files) - bad}, 실패 {bad}")

    # 1° 지역: 위도·경도의 내림 값으로 묶음 (39.0, 131.0 경계는 38, 130 칸에 넣음)
    rlat = np.minimum(np.floor(la), 38).astype(int)
    rlon = np.minimum(np.floor(lo), 130).astype(int)
    regions = sorted(set(zip(rlat.ravel(), rlon.ravel())))
    masks = {r: (rlat == r[0]) & (rlon == r[1]) for r in regions}
    print(f"상자 격자 {la.shape}, 1° 지역 {len(regions)}곳")

    rows = []
    for (day, var), fhs in store.items():
        R = pd.Timestamp(day)
        for s, e, short in BLOCKS:
            if e not in fhs or (short is not None and short not in fhs):
                continue
            blk = fhs[e] if short is None else 2 * fhs[e] - fhs[short]      # 6시간 평균에서 앞 3시간 평균을 빼 뒤 3시간 평균
            if var == "DSWRF":
                blk = np.clip(blk, 0, None)
            else:
                blk = np.clip(blk, 0, 100)
            feats = {f"gfs_{var.lower()}_r{r[0]}_{r[1]}": float(blk[mk].mean()) for r, mk in masks.items()}
            feats[f"gfs_{var.lower()}_kr"] = float(blk.mean())
            start_utc = R + pd.Timedelta(hours=s)
            for k in range(1, 4):                                            # 블록 안 3시간 (ts = 그 1시간의 끝)
                ts = (start_utc + pd.Timedelta(hours=k)).tz_localize("UTC").tz_convert("Asia/Seoul")
                rows.append({"ts": ts, "var": var, **feats})
    long = pd.DataFrame(rows)
    parts = []
    for v in ("DSWRF", "TCDC"):
        cols = ["ts"] + [c for c in long.columns if c.startswith(f"gfs_{v.lower()}_")]
        parts.append(long.loc[long["var"] == v, cols])
    wide = parts[0].merge(parts[1], on="ts", how="outer")
    wide = wide.sort_values("ts").reset_index(drop=True)
    wide["ts"] = wide.ts.dt.as_unit("ns")
    wide.to_parquet(OUT, index=False)
    print(f"저장 {OUT} : {len(wide):,}시각 × {wide.shape[1] - 1}피처, 기간 {wide.ts.min()} ~ {wide.ts.max()}")


if __name__ == "__main__":
    main()

"""F11 — 기압 정의 변경 같은 '체계적 원인'이 더 있는지 찾는다 (2026-11-11, docs/피처_근거_분석.md §8).

A. 수준 치우침: 검증 세 해에서 현재 앙상블의 예측 합 ÷ 실적 합(전체, SMP·k 가중, 맑은 한낮)이 해마다 같은 방향·커지는 방향인가
B. 맑은 한낮(A 운량 ≤ 10%, 11 ~ 15시)으로 날씨 오차를 줄인 뒤 치우침을 평일/휴일, 봄/그 밖, 지역, 본/처음 보는 발전소로 나눔
   (출력제어·설비 노화처럼 날씨가 아닌 원인은 맑은 날에 드러난다)
C. 묶음 배율 α 를 해마다 따로 최적화하면 벌금이 얼마나 주는가 = 수준 치우침의 크기(참조 구성)
D. 출처 어긋남: 출처 전용 모델끼리(GFS 전용 S2T1, GEFS 전용 S3, 지역 예보 전용 V07T1) 하루 예측 비율의 월별 중앙값을
   검증 세 해와 평가 기간에서 비교. 한 출처의 값 정의가 바뀌면 그 출처만 쓴 모델이 따로 움직인다.
   양성 대조: A 기압 정의 변경(2026-01 ~)은 V07T1 에만 들어가므로 2026-02 ~ 05 에 S2T1 / V07T1 이 커져야 한다.

    python analysis_lab/F11_체계적원인_탐색.py           (참조 구성, 학습 기간 SMP. 평가 기간 자료는 D 의 예측 비교에만 쓰고 무엇도 정하지 않음)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from evaluate import evaluate_preds
from features_v4 import HOLIDAYS
from oibc_common import PREDS, PREDS_TEST, WORK, load_smp

ENS = "E11-10T1x2"
FOLDS = [("2024-25", ""), ("2023-24", "_cut2023"), ("2022-23", "_cut2022")]
COLS = ["site_id", "ts", "energy", "capacity_kw", "lat", "lon", "start", "hour", "month", "date", "elev", "wa_cloud_r17", "split"]


def fold_rows(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    p = pd.read_parquet(PREDS / f"{ENS}{tag}.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
    d = p.merge(base, on=["site_id", "ts"], how="inner")
    s0 = d.ts.min()
    seen = set(base.loc[(base.ts < s0) & (base.ts >= s0 - pd.DateOffset(years=2)), "site_id"].unique())
    d["seen"] = d.site_id.isin(seen)
    return d


def ratio(d: pd.DataFrame, w: str | None = None) -> float:
    if w is None:
        return d.pred_kwh.sum() / d.energy.sum()
    return (d.pred_kwh * d[w]).sum() / (d.energy * d[w]).sum()


def main() -> None:
    base = pd.read_parquet(WORK / "base.parquet", columns=COLS)
    base["site_id"] = base.site_id.astype(str)
    smp = load_smp("train").set_index("ts").w
    hol = pd.DatetimeIndex(HOLIDAYS.astype("datetime64[ns]"))
    print(f"기준 앙상블 {ENS}. 비율 = 예측 합 ÷ 실적 합 (1 보다 크면 과대 예측)\n")

    print("A·B. 수준 치우침")
    rows = []
    for y, tag in FOLDS:
        d = fold_rows(base, tag)
        d["w"] = d.ts.map(smp).to_numpy()
        day = d.date.dt.normalize()
        d["off"] = (day.dt.dayofweek >= 5) | day.isin(hol)
        clear = d[(d.hour >= 11) & (d.hour <= 15) & (d.wa_cloud_r17 <= 10)]
        sw = (clear.lat < 36.0) & (clear.lon < 127.6)
        spring = clear.month.isin([3, 4, 5])
        r = {"해": y, "전체": ratio(d), "SMP·k 가중": ratio(d, "w"), "맑은 한낮": ratio(clear),
             "맑은 한낮 평일": ratio(clear[~clear.off]), "맑은 한낮 휴일": ratio(clear[clear.off]),
             "맑은 한낮 봄 평일": ratio(clear[spring & ~clear.off]), "맑은 한낮 봄 휴일": ratio(clear[spring & clear.off]),
             "맑은 한낮 남서": ratio(clear[sw]), "맑은 한낮 그 밖": ratio(clear[~sw]),
             "본 발전소": ratio(d[d.seen]), "처음 보는 발전소": ratio(d[~d.seen]),
             "맑은 한낮 행 비율": len(clear) / len(d[(d.hour >= 11) & (d.hour <= 15)])}
        rows.append(r)
        m = d.groupby(d.ts.dt.month).apply(lambda g: g.pred_kwh.sum() / g.energy.sum())
        print(f"  {y} 월별: " + " ".join(f"{k}월 {v:.3f}" for k, v in m.items()))
    t = pd.DataFrame(rows).set_index("해").T
    print(t.map(lambda v: f"{v:.3f}").to_string())

    print("\nC. 해마다 따로 고른 배율 α (참조 구성 벌금, 낙관치 = 수준 치우침을 알았다면 줄일 수 있는 상한)")
    for y, tag in FOLDS:
        p = pd.read_parquet(PREDS / f"{ENS}{tag}.parquet")
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        start = p.ts.min()
        a = base[(base.ts >= start) & (base.ts <= p.ts.max())][["site_id", "ts", "energy"]].reset_index(drop=True)
        res = {}
        for al in np.round(np.arange(0.94, 1.041, 0.01), 2):
            res[al] = evaluate_preds(p.assign(pred_kwh=p.pred_kwh * al), actual=a)["penalty"]
        best = min(res, key=res.get)
        print(f"  {y}: 최적 α {best:.2f}, 벌금 {res[1.0]:,.0f} → {res[best]:,.0f} ({(res[best] / res[1.0] - 1) * 100:+.2f}%)")

    print("\nD. 출처 어긋남: 하루 예측 합 비율의 월별 중앙값 (11 ~ 15시, 세 모델 모두 하루 합 > 0 인 날)")
    pairs = [("S2T1x2", "V07T1x2", "GFS 전용 ÷ 지역 예보 전용"), ("S3x2", "V07T1x2", "GEFS 전용 ÷ 지역 예보 전용"),
             ("S3x2", "S2T1x2", "GEFS 전용 ÷ GFS 전용")]
    names = sorted({n for a_, b_, _ in pairs for n in (a_, b_)})

    def daily(name: str, path: Path) -> pd.Series:
        p = pd.read_parquet(path, columns=["ts", "pred_kwh"])
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul")
        p = p[(p.ts.dt.hour >= 11) & (p.ts.dt.hour <= 15)]
        return p.groupby(p.ts.dt.normalize()).pred_kwh.sum().rename(name)

    sets = {y: pd.concat([daily(n, PREDS / f"{n}{tag}.parquet") for n in names], axis=1) for y, tag in FOLDS}
    sets["평가 2025-26"] = pd.concat([daily(n, PREDS_TEST / f"{n}.parquet") for n in names], axis=1)
    for a_, b_, label in pairs:
        tab = {}
        for y, dd in sets.items():
            dd = dd[(dd > 0).all(axis=1)]
            r = (dd[a_] / dd[b_]).groupby(dd.index.month).median()
            tab[y] = r
        tab = pd.DataFrame(tab).reindex([6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5])
        v = tab[[y for y, _ in FOLDS]]
        z = (tab["평가 2025-26"] - v.mean(axis=1)) / v.std(axis=1)
        out = tab.map(lambda x: f"{x:.3f}")
        out["평가 − 검증 평균(%)"] = ((tab["평가 2025-26"] / v.mean(axis=1) - 1) * 100).map(lambda x: f"{x:+.1f}")
        out["검증 범위 밖"] = np.where((tab["평가 2025-26"] > v.max(axis=1)) | (tab["평가 2025-26"] < v.min(axis=1)), "●", "")
        out["z"] = z.map(lambda x: f"{x:+.1f}")
        print(f"\n  {label}")
        print(out.to_string())


if __name__ == "__main__":
    main()

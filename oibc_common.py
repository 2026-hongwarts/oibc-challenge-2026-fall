"""OIBC 2026 공통 유틸 — 경로·로더·태양 기하·점수식·고장일 탐지·구성 탐색.

EDA 노트북과 이후 모델링/구성 최적화 코드가 같은 정의를 쓰도록 한곳에 모은다.
시각은 모두 KST(tz-aware), 1시간 우측닫힘(ts=13:00 은 12:00~13:00 구간).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "01_participant_dataset"
WORK = ROOT / "work"        # 공통 테이블 (build_base.py)
PREDS = ROOT / "preds"      # 가상 평가연도 예측 (실험번호.parquet)
TZ = "Asia/Seoul"

TOTAL_CAP_KW = 42_655.4
CAP_MIN_KW = 20_000.0                      # 집합별 설비용량 하한
CAP_MIN_FRAC = CAP_MIN_KW / TOTAL_CAP_KW   # 부분 집합으로 시뮬레이션할 때 쓰는 비율 하한(≈0.469)
PENALTY_RATE = 0.10

# 가상 평가연도: 실제 평가(2025-06-01 01:00 ~ 2026-06-01 00:00)와 같은 6월 시작 1년
VALID_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)
VALID_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)
TEST_FROM = pd.Timestamp("2025-06-01 01:00", tz=TZ)   # 실제 평가 기간 시작


# ── 시간 ────────────────────────────────────────────────────────────────
def k_weight(ts) -> np.ndarray:
    """시가 11~15 이면 3, 그 외 1 (README 정의)."""
    h = pd.DatetimeIndex(ts).hour
    return np.where((h >= 11) & (h <= 15), 3, 1).astype("int8")


def day_of(ts) -> pd.Series:
    """우측닫힘 시각이 속한 날짜(tz 없는 자정 datetime64). ts=다음날 00:00 은 전날 23~24시 구간이다."""
    s = pd.Series(pd.DatetimeIndex(ts)) - pd.Timedelta(hours=1)
    return s.dt.tz_localize(None).dt.normalize()


def solar(ts, lat, lon):
    """태양 고도·방위·cos(천정각) (NOAA 근사, baseline.py 와 동일). 구간 중앙(-30분) 기준."""
    ts = pd.DatetimeIndex(ts)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                       - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    decl = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
            - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
            - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    tst = (hour - 0.5) * 60 + eqtime + 4 * (np.asarray(lon) - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(np.asarray(lat))
    cosz = np.clip(np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha), -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha),
                               np.tan(decl)*np.cos(la) - np.sin(la)*np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


# ── 로더 ────────────────────────────────────────────────────────────────
def load_sites() -> pd.DataFrame:
    return pd.read_csv(DATA / "sites.csv")


def load_generation(sites: pd.DataFrame | None = None) -> pd.DataFrame:
    """발전량 + 설비용량·좌표·관측지점 + cf·태양기하·날짜."""
    sites = load_sites() if sites is None else sites
    g = pd.read_parquet(DATA / "train" / "generation.parquet")
    g["ts"] = pd.to_datetime(g.ts).dt.as_unit("ns")
    g = g.merge(sites[["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]],
                on="site_id", how="left")
    g["cf"] = (g.energy / g.capacity_kw).astype("float32")
    g["elev"], g["azim"], g["cosz"] = solar(g.ts, g.lat.to_numpy(), g.lon.to_numpy())
    g["date"] = day_of(g.ts).to_numpy()
    return g


def load_smp(split: str = "test") -> pd.DataFrame:
    """split='test' 는 채점용 smp.csv(k 포함), 'train' 은 학습 구간(k 를 시각에서 파생)."""
    path = DATA / ("smp.csv" if split == "test" else "train/smp.csv")
    s = pd.read_csv(path)
    s["ts"] = pd.to_datetime(s.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    if "k" not in s:
        s["k"] = k_weight(s.ts)
    s["w"] = s.smp * s.k                   # 오차 1 kWh 의 페널티 = 0.10 × w
    return s


def load_forecast(provider: str, split: str = "train", cols=None) -> pd.DataFrame:
    """provider='wa'|'wb'. cols 를 주면 키(station, round, ts)와 함께 그 열만 읽는다."""
    key = [f"{provider}_station", "round", "ts"]
    f = pd.read_parquet(DATA / split / f"forecast_{provider}.parquet",
                        columns=None if cols is None else key + list(cols))
    f["ts"] = pd.to_datetime(f.ts).dt.as_unit("ns")
    return f


def fill_round_gaps(f: pd.DataFrame, provider: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """(지점, 라운드, 시각) 격자에서 빠진 행을 같은 지점·시각의 다른 라운드 값으로 채운다."""
    st = f"{provider}_station"
    full = pd.MultiIndex.from_product([f[st].unique(), sorted(f["round"].unique()), index],
                                      names=[st, "round", "ts"])
    g = f.set_index([st, "round", "ts"]).reindex(full)
    other = g.groupby(level=[st, "ts"]).transform("first")   # 같은 지점·시각의 비결측 값
    return g.fillna(other).reset_index()


# ── 데이터 품질 ──────────────────────────────────────────────────────────
def dead_days(gen: pd.DataFrame, frac: float = 0.01, min_elev: float = 5.0) -> pd.DataFrame:
    """해 뜬 시간(elev>min_elev) 일 발전량이 설비용량×frac 이하인 (site_id, date) — 고장·정지 추정."""
    d = gen[gen.elev > min_elev].groupby(["site_id", "date"]).agg(
        energy=("energy", "sum"), cap=("capacity_kw", "first"), hours=("energy", "size"))
    return d[d.energy <= frac * d.cap].reset_index()[["site_id", "date", "energy", "hours"]]


def drop_dead(df: pd.DataFrame, dead: pd.DataFrame) -> pd.DataFrame:
    """학습용: 정지·고장 추정일의 행을 뺀다. 검증 실적에는 쓰지 않는다(평가에서도 고장은 일어난다)."""
    m = df.merge(dead[["site_id", "date"]].assign(_dead=True), on=["site_id", "date"], how="left")
    return m[m._dead.isna()].drop(columns="_dead")


# ── 점수 ────────────────────────────────────────────────────────────────
def score(actual: pd.DataFrame, forecast: pd.DataFrame, smp: pd.DataFrame) -> dict:
    """대회 점수식. actual/forecast 는 long 형식 [group, ts, kwh], smp 는 [ts, smp, k].

    S = Σ SMP·실적 − 0.10 × Σ SMP·k·|예측 − 실적|  (G1, G2 합)
    """
    a = actual.merge(forecast, on=["group", "ts"], suffixes=("_a", "_f")).merge(smp, on="ts")
    revenue = float((a.smp * a.kwh_a).sum())
    penalty = PENALTY_RATE * float((a.smp * a.k * (a.kwh_f - a.kwh_a).abs()).sum())
    return {"score": revenue - penalty, "revenue": revenue, "penalty": penalty,
            "penalty_pct": penalty / revenue * 100}


def group_penalty(R: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    """R: (시각×발전소) 오차(예측−실적, kWh), x: G1 여부(bool), w: SMP·k. 0.10 을 곱하기 전 값."""
    return float(w @ (np.abs(R[:, x].sum(1)) + np.abs(R[:, ~x].sum(1))))


def local_search(R: np.ndarray, w: np.ndarray, x0: np.ndarray, cap: np.ndarray,
                 min_frac: float = CAP_MIN_FRAC, max_iter: int = 500) -> np.ndarray:
    """발전소 1개를 반대 집합으로 옮기는 이동 중 최선을 반복 적용(용량 하한 유지)."""
    x = x0.copy()
    e1, e2 = R[:, x].sum(1), R[:, ~x].sum(1)
    cur = w @ (np.abs(e1) + np.abs(e2))
    for _ in range(max_iter):
        s = np.where(x, -1.0, 1.0)                       # G2→G1 이면 +1
        cand = w @ (np.abs(e1[:, None] + R * s) + np.abs(e2[:, None] - R * s))
        frac = (cap[x].sum() + s * cap) / cap.sum()
        cand[(frac < min_frac) | (frac > 1 - min_frac)] = np.inf
        i = int(np.argmin(cand))
        if cand[i] >= cur - 1e-9:
            break
        x[i] = ~x[i]
        e1, e2, cur = e1 + R[:, i] * s[i], e2 - R[:, i] * s[i], cand[i]
    return x

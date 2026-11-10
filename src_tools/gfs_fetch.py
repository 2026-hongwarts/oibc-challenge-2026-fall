"""GFS(미국 기상청 전 지구 예보) 0.5° 일사량·운량 내려받기 — 외부 공개 자료 (대회 규정 §5 ~ §7).

    python src_tools/gfs_fetch.py                     2021-06-01 ~ 2026-05-30 실행 전부 (이어받기 가능)
    python src_tools/gfs_fetch.py --start 2024-06-01 --end 2024-06-03     일부만

출처 (규정 §5: 공개 · 상업 이용 가능 · 같은 주소에서 다시 받을 수 있음)
    NOAA Open Data Dissemination(NODD) 프로그램의 AWS 공개 저장소 noaa-gfs-bdp-pds (로그인 없음)
    https://registry.opendata.aws/noaa-gfs-bdp-pds/
    파일: https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{YYYYMMDD}/00/atmos/gfs.t00z.pgrb2.0p50.f{FFF}
    색인: 같은 주소 + .idx (각 변수의 바이트 위치) → 필요한 변수만 바이트 범위 요청으로 받는다.

쓰는 실행과 공개 시각 (규정 §4·§6: 전날 17:00 KST 전에 공개된 실행만)
    대상일 D(KST)의 예측에는 D 전날(UTC 날짜 D-1)의 00시 UTC 실행만 쓴다 = 한국 시각 D-1 09:00 실행.
    이 실행의 파일은 같은 날 약 03:40 UTC(= D-1 12:40 KST)에 저장소에 올라온다 → 마감 17:00 KST 보다 4시간 이상 앞섬.
    파일마다 저장소의 LastModified(공개 시각)를 outputs/work/gfs_raw/manifest.csv 에 남겨 근거로 삼는다.
    06시 UTC 실행(D-1 15:00 KST)은 공개가 17:00 KST 이후라 쓰지 않는다.

GEFS(--source gefs): 같은 NODD 의 noaa-gefs-pds (https://registry.opendata.aws/noaa-gefs/), 00시 UTC 실행,
    pgrb2ap5/geavg(앙상블 평균)·gespr(앙상블 표준편차) 0.5°. 공개 약 03:55 ~ 04:02 UTC(= D-1 13시 KST) → 마감 전.

받는 것: 예보 시각 f021, f024, f027, f030, f033, f036 (= 대상일 KST 03 ~ 21시를 덮는 3시간 평균 블록)의
    DSWRF:surface(지표 하향 단파 일사, 평균) · TCDC:entire atmosphere(전체 운량, 평균) GRIB2 메시지.
    원본 메시지를 outputs/work/gfs_raw/{YYYY}/{YYYYMMDD}_f{FFF}_{VAR}.grb2 로 저장(전 지구, 해석은 gfs_decode.py).
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "work" / "gfs_raw"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
FHOURS = [21, 24, 27, 30, 33, 36]
RES = "0p50"
ALL_VARS = {"DSWRF": re.compile(r":DSWRF:surface:\d+-\d+ hour ave fcst:"),
            "TCDC": re.compile(r":TCDC:entire atmosphere:\d+-\d+ hour ave fcst:"),
            "LCDC": re.compile(r":LCDC:low cloud layer:\d+-\d+ hour ave fcst:"),
            "MCDC": re.compile(r":MCDC:middle cloud layer:\d+-\d+ hour ave fcst:"),
            "HCDC": re.compile(r":HCDC:high cloud layer:\d+-\d+ hour ave fcst:")}
VARS = {k: ALL_VARS[k] for k in ("DSWRF", "TCDC")}
# GEFS(NOAA 전 지구 앙상블 예보, 31개 멤버): 앙상블 평균(geavg)·표준편차(gespr) 0.5°. 파일 이름의 변수 끝에 M(평균)/S(표준편차)
SOURCE = "gfs"
GEFS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_PRODS = {"M": "geavg", "S": "gespr"}


def get(url: str, rng: tuple[int, int] | None = None, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def _list(bucket: str, prefix: str) -> dict[str, str]:
    x = get(f"{bucket}/?list-type=2&prefix={prefix}&max-keys=1000").decode()
    return dict(zip(re.findall(r"<Key>([^<]+)</Key>", x), re.findall(r"<LastModified>([^<]+)</LastModified>", x)))


def list_cycle(day: str) -> dict[str, str]:
    """그날 00시 실행의 파일 LastModified."""
    if SOURCE == "gefs":
        out = {}
        for prod in GEFS_PRODS.values():
            out.update(_list(GEFS_BUCKET, f"gefs.{day}/00/atmos/pgrb2ap5/{prod}.t00z.pgrb2a.0p50.f0"))
        return out
    return _list(BUCKET, f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.{RES}.f0")


def _targets(day: str, fh: int) -> list[tuple[str, str, str]]:
    """(파일 변수 이름, 저장소 주소, 키) 목록."""
    if SOURCE == "gefs":
        return [(v + s, GEFS_BUCKET, f"gefs.{day}/00/atmos/pgrb2ap5/{prod}.t00z.pgrb2a.0p50.f{fh:03d}")
                for s, prod in GEFS_PRODS.items() for v in VARS]
    return [(v, BUCKET, f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.{RES}.f{fh:03d}") for v in VARS]


def fetch_one(day: str, fh: int, mods: dict[str, str]) -> list[list]:
    rows = []
    need = [t for t in _targets(day, fh) if not (OUT / day[:4] / f"{day}_f{fh:03d}_{t[0]}.grb2").exists()]
    idx: dict[str, list[str]] = {}
    for name, bucket, key in need:
        if key not in idx:
            idx[key] = get(f"{bucket}/{key}.idx").decode().splitlines()
        lines = idx[key]
        offs = [int(l.split(":")[1]) for l in lines]
        v = name if SOURCE != "gefs" else name[:-1]
        i = next(j for j, l in enumerate(lines) if VARS[v].search(l))
        end = offs[i + 1] - 1 if i + 1 < len(offs) else None
        data = get(f"{bucket}/{key}", (offs[i], end))
        p = OUT / day[:4] / f"{day}_f{fh:03d}_{name}.grb2"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append([day, fh, name, f"{bucket}/{key}", mods.get(key, ""), lines[i], len(data)])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="GFS 0.5° 일사량·운량 내려받기")
    ap.add_argument("--start", default="2021-06-01", help="첫 실행 날짜(UTC) = 첫 대상일 - 1일")
    ap.add_argument("--end", default="2026-05-30", help="마지막 실행 날짜(UTC) = 마지막 대상일 - 1일")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--res", default="0p50", choices=["0p50", "1p00"], help="해상도 (1p00 은 outputs/work/gfs_raw_1p00 에)")
    ap.add_argument("--vars", nargs="+", default=["DSWRF", "TCDC"], choices=list(ALL_VARS))
    ap.add_argument("--source", default="gfs", choices=["gfs", "gefs"],
                    help="gefs: 앙상블 평균·표준편차 0.5° (outputs/work/gefs_raw 에, 변수 이름 끝 M/S)")
    args = ap.parse_args()
    global OUT, RES, VARS, SOURCE
    RES = args.res
    VARS = {k: ALL_VARS[k] for k in args.vars}
    SOURCE = args.source
    if SOURCE == "gefs":
        OUT = ROOT / "outputs" / "work" / "gefs_raw"
    elif RES != "0p50":
        OUT = ROOT / "outputs" / "work" / f"gfs_raw_{RES}"
    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [(d0 + timedelta(n)).strftime("%Y%m%d") for n in range((d1 - d0).days + 1)]
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.csv"
    new = not man.exists()
    if not new:      # 중간에 끊겨 공개 시각 기록 없이 남은 파일은 지우고 다시 받는다(규정 §6 근거를 모든 파일에 남기기 위해)
        with open(man, encoding="utf-8") as f:
            have = {(r[0], int(r[1]), r[2]) for r in csv.reader(f) if r and r[0] != "run_date"}
        orphan = [p for p in OUT.glob("*/*.grb2")
                  if (p.stem.split("_")[0], int(p.stem.split("_")[1][1:]), p.stem.split("_", 2)[2]) not in have]
        for p in orphan:
            p.unlink()
        print(f"공개 시각 기록이 없는 파일 {len(orphan)}개 지움 → 다시 받음", flush=True)
    t0, done = time.time(), 0
    with open(man, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["run_date", "fhour", "var", "url", "last_modified_utc", "idx_line", "bytes"])
        for i in range(0, len(days), 20):                     # 20일씩 묶어 진행
            chunk = days[i:i + 20]
            mods = {}
            for d in chunk:
                try:
                    mods.update(list_cycle(d))
                except Exception as e:
                    print(f"  목록 실패 {d}: {e}", flush=True)
            futs = [ex.submit(fetch_one, d, fh, mods) for d in chunk for fh in FHOURS]
            for fu in as_completed(futs):
                try:
                    for r in fu.result():
                        w.writerow(r)
                except Exception as e:
                    print(f"  실패: {e}", flush=True)
            f.flush()
            done += len(chunk)
            el = time.time() - t0
            print(f"[{el:6.0f}s] {done}/{len(days)}일 ({chunk[-1]}), 남은 시간 약 {el / done * (len(days) - done) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    main()

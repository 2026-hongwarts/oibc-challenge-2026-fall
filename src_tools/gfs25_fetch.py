"""GFS 0.25° 1시간 일사량(DSWRF) 받기 + 한반도 상자 잘라 저장 (11/12, 판 뒤집기 검토 ③) — gfs 환경(pygrib)에서 실행.

    conda run -n gfs python src_tools/gfs25_fetch.py                  2021-06-01 ~ 2026-05-30 (이어받기 가능)
    → outputs/work/gfs25_crop/{YYYYMMDD}.npz (예보 시각 19 ~ 35 의 상자 격자), outputs/work/gfs25_crop/manifest.csv

출처(규정 §5): NOAA NODD AWS noaa-gfs-bdp-pds, gfs.{R}/00/atmos/gfs.t00z.pgrb2.0p25.f{FFF} (0.25°, 1시간 간격 f001 ~ f120).
공개 시각(§6): 파일마다 LastModified 를 manifest 에 남기고, 해석에서 R 08:00 UTC(= 대상일 전날 17:00 KST) 보다 늦은 것은 뺀다.
변수: DSWRF:surface 'S-k hour ave'(6시간마다 다시 시작하는 구간 평균) → 해석에서 시간별로 풀기(gfs25_decode).
f019 ~ f035 = 대상일 KST 04 ~ 20시(ts = 그 1시간의 끝). 원본(약 0.8MB × 17 × 1825일)은 저장하지 않고 상자만 남긴다.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pygrib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "work" / "gfs25_crop"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
FH = list(range(19, 36))
BOX = dict(lat1=33.0, lat2=39.0, lon1=124.0, lon2=131.0)
PAT = re.compile(r":DSWRF:surface:\d+-\d+ hour ave fcst:")


def get(url, rng=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def listing(day):
    x = get(f"{BUCKET}/?list-type=2&prefix=gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f0&max-keys=1000").decode()
    return dict(zip(re.findall(r"<Key>([^<]+)</Key>", x), re.findall(r"<LastModified>([^<]+)</LastModified>", x)))


def one(day, fh, mods):
    key = f"gfs.{day}/00/atmos/gfs.t00z.pgrb2.0p25.f{fh:03d}"
    lines = get(f"{BUCKET}/{key}.idx").decode().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    i = next(j for j, l in enumerate(lines) if PAT.search(l))
    end = offs[i + 1] - 1 if i + 1 < len(offs) else None
    data = get(f"{BUCKET}/{key}", (offs[i], end))
    g = pygrib.fromstring(data)
    v, la, lo = g.data(**BOX)
    return fh, np.asarray(v, dtype="float32"), la, lo, [day, fh, "DSWRF", f"{BUCKET}/{key}", mods.get(key, ""), lines[i], len(data)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--end", default="2026-05-30")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=10, help="한꺼번에 받을 날 수")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / f"manifest_{a.start}.csv"          # 프로세스마다 따로(날짜 구간을 나눠 동시에 돌릴 때)
    new = not man.exists()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [(d0 + timedelta(n)).strftime("%Y%m%d") for n in range((d1 - d0).days + 1)]
    todo = [d for d in days if not (OUT / f"{d}.npz").exists()]
    print(f"남은 날 {len(todo)}/{len(days)}", flush=True)
    t0 = time.time()
    with open(man, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["run_date", "fhour", "var", "url", "last_modified_utc", "idx_line", "bytes"])
        done = 0
        for c in range(0, len(todo), a.chunk):          # 여러 날을 한꺼번에(지연 시간이 병목이라)
            chunk = todo[c:c + a.chunk]
            mods = {}
            for day in chunk:
                try:
                    mods.update(listing(day))
                except Exception as e:
                    print(f"  목록 실패 {day}: {e}", flush=True)
            futs = {(day, fh): ex.submit(one, day, fh, mods) for day in chunk for fh in FH}
            for day in chunk:
                try:
                    res = [futs[(day, fh)].result() for fh in FH]
                except Exception as e:
                    print(f"  실패 {day}: {type(e).__name__} {e}", flush=True)
                    continue
                arr = np.stack([r[1] for r in res])
                np.savez_compressed(OUT / f"{day}.npz", v=arr, fh=np.array(FH), lat=res[0][2].astype("float32"), lon=res[0][3].astype("float32"))
                for r in res:
                    w.writerow(r[4])
            f.flush()
            done += len(chunk)
            el = time.time() - t0
            print(f"[{el:6.0f}s] {done}/{len(todo)}일 ({chunk[-1]}), 남은 시간 약 {el / done * (len(todo) - done) / 60:.0f}분", flush=True)


if __name__ == "__main__":
    main()

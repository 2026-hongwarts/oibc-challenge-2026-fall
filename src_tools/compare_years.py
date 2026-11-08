"""세 해 비교표: 실험마다 2024-25(가상 평가연도), 2023-24(_cut2023), 2022-23(_cut2022) 벌금과 기준 대비 %.

    python src_tools/compare_years.py E7-00x2 X01x2 X02ax2 ...      첫 이름이 기준
    python src_tools/compare_years.py E7-00x2 X01x2 --md out.md     표를 마크다운으로도 저장

파일이 없는 해는 빈칸. 채택 관문: 세 해 모두 기준보다 낮음.
참조 구성(sites.csv current_group)으로 채점한다.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import pandas as pd

from evaluate import evaluate_preds, read_preds
from oibc_common import PREDS, WORK

YEARS = [("2024-25", None), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]


@lru_cache(maxsize=None)
def actual(cut: str | None) -> pd.DataFrame:
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "split"])
    if cut is None:
        a = a[a.split == "valid"]
    else:
        from train_site_model import cut_window
        s, e = cut_window(cut)
        a = a[(a.ts >= s) & (a.ts < e)]
    a = a.drop(columns="split").reset_index(drop=True)
    a["site_id"] = a.site_id.astype(str)
    return a


def penalty(name: str, cut: str | None) -> tuple[float, float] | None:
    p = PREDS / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")
    if not p.exists():
        return None
    r = evaluate_preds(read_preds(p), actual=actual(cut))
    return r["penalty"], r["peak"]


def main() -> None:
    ap = argparse.ArgumentParser(description="세 해 비교표")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    ref = {y: penalty(args.names[0], c) for y, c in YEARS}
    rows = []
    for n in args.names:
        row = {"실험": n}
        ok = n != args.names[0]
        for y, c in YEARS:
            v = penalty(n, c)
            if v is None or ref[y] is None:
                row[y] = "" if v is None else f"{v[0]:,.0f}"
                ok = False
                continue
            d = (v[0] / ref[y][0] - 1) * 100
            row[y] = f"{v[0]:,.0f}" if n == args.names[0] else f"{d:+.2f}%"
            ok = ok and d < 0
        row["관문"] = "" if n == args.names[0] else ("통과" if ok else "")
        rows.append(row)
        print(" | ".join(f"{row[k]:>14}" for k in row), flush=True)
    if args.md:
        keys = list(rows[0])
        lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(r[k] for k in keys) + " |" for r in rows]
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""후보마다 '자기 구성'으로 세 해 벌금을 매겨 첫 후보 대비 감소액·% 를 낸다.

    python src_tools/compare_own_comp.py E10-08x2 E11-10x2 E11-10T1x2
        → 각 이름의 outputs/compositions/{이름}.csv 로 채점 (없으면 --comp 이름=경로)
    python src_tools/compare_own_comp.py E10-08x2 X=outputs/compositions/E11-10x2.csv

이름=경로 로 주면 그 구성 파일을 쓴다. 첫 이름이 기준(보통 마지막 제출).
"""
from __future__ import annotations

import sys

from compare_years import YEARS, actual
from evaluate import evaluate_preds, read_preds
from oibc_common import OUTPUTS, PREDS


def pen(name: str, comp: str) -> list[float]:
    out = []
    for _, c in YEARS:
        f = PREDS / (f"{name}_cut{c[:4]}.parquet" if c else f"{name}.parquet")
        out.append(evaluate_preds(read_preds(f), comp, actual(c))["penalty"])
    return out


def main() -> None:
    items = []
    for a in sys.argv[1:]:
        n, _, comp = a.partition("=")
        items.append((n, comp or str(OUTPUTS / "compositions" / f"{n}.csv")))
    res = {n: pen(n, c) for n, c in items}
    head = f"{'':12s}" + "".join(f"{y:>15s}" for y, _ in YEARS) + f"{'합':>15s}"
    print(head)
    base = res[items[0][0]]
    for n, v in res.items():
        print(f"{n:12s}" + "".join(f"{x:15,.0f}" for x in v) + f"{sum(v):15,.0f}")
    for n, v in list(res.items())[1:]:
        print(f"{n + ' 감소':12s}" + "".join(f"{y - x:15,.0f}" for x, y in zip(base, v)) + f"{sum(v) - sum(base):15,.0f}")
        print(f"{n + ' %':12s}" + "".join(f"{(y / x - 1) * 100:+14.2f}%" for x, y in zip(base, v))
              + f"{(sum(v) / sum(base) - 1) * 100:+14.2f}%")


if __name__ == "__main__":
    main()

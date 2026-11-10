"""11/11 밤샘 자동 진행 (사용자 승인) — 미리 정한 규칙으로만 판단하고, 제출은 하지 않는다(제출은 사용자).

    python analysis_lab/overnight.py >> outputs/work/overnight.log 2>&1      (다시 실행하면 끝난 학습은 건너뜀)

1. 기다림: G8w(run_smpw.log "SMP 가중 끝"), MLP(run_mlp.log "MLP 끝"), S4(run_gfs18.log "GFS 이전 실행 끝"),
   에어로졸(run_fe_gpu.log "GPU 피처 걸러내기 끝"). 각자 최대 11/11 06:00 까지 — 그때까지 안 끝난 것은 불통과로 본다.
2. 판정(세 해 모두 개선, 참조 구성):
     G8w   G8wx2       vs G5bT1x2     → 통과하면 G5bT1x2 자리를 G8wx2 로
     에어로졸 E13-11agx2 vs E11-11gx2   → 통과하면 E11-11x2 자리를 E13-11agx2 로
     S4    E13-11x2    vs E12-10cx2   → 통과하면 S4x2 추가
     MLP   E13-12x2    vs E12-10cx2   → 통과하면 M1x2 추가
3. 후보 E14x2 = E12-10c 에 통과한 변경을 모두 적용(같은 비중). E12-10cx2 대비 세 해 모두 개선이 아니면
   통과한 변경 하나씩만 적용한 판 중 세 해 모두 개선 + 가장 나쁜 해가 가장 덜 나쁜 것(maximin).
4. 구성: optimize_composition (1개 옮기기) + --swap. 맞바꾸기는 세 검증 모두 1개 옮기기판보다 낮을 때만.
5. 최고 제출 E11-10T1x2(자기 구성) 대비 세 해 합 −3% 이하이면
   시드 재현(새 구성원과 S3 를 시드 2·3 판으로, 기준도 같은 자리를 시드 2·3 판으로) → 세 해 모두 개선이면 제출 파일까지 만든다.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
from compare_years import YEARS, penalty          # noqa: E402

PY = sys.executable
W = ROOT / "outputs" / "work"
PREDS = ROOT / "outputs" / "preds"
DEADLINE = datetime(2026, 11, 11, 6, 0)
BASE = ["E11-12x2", "E11-11x2", "G5bT1x2", "S3x2", "V07T1x2"]
CUTS = [None, "2023-06-01", "2022-06-01"]


def log(m: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M}] {m}", flush=True)


def run(args: list[str], out: Path | None = None) -> int:
    with open(out, "w", encoding="utf-8") if out else open(W / "overnight_sub.log", "a", encoding="utf-8") as f:
        return subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode


def wait_for(logname: str, text: str) -> bool:
    p = W / logname
    while datetime.now() < DEADLINE:
        if p.exists() and text in p.read_text(encoding="utf-8", errors="ignore"):
            return True
        time.sleep(60)
    return False


def pens(name: str) -> list[float] | None:
    out = []
    for _, c in YEARS:
        r = penalty(name, c)
        if r is None:
            return None
        out.append(r[0])
    return out


def judge(base: str, cand: str) -> tuple[bool, list[float]]:
    b, c = pens(base), pens(cand)
    if b is None or c is None:
        return False, []
    r = [y / x - 1 for x, y in zip(b, c)]
    return all(v < 0 for v in r), r


def fmt(r: list[float]) -> str:
    return " / ".join(f"{v * 100:+.2f}%" for v in r) if r else "결과 없음"


def blend(out: str, members: list[str], mode: str = "valid") -> None:
    if mode == "test":
        run(["src_tools/blend.py", "--out", out, *members, "--mode", "test"])
        return
    for c in CUTS:
        run(["src_tools/blend.py", "--out", out, *members, *(["--cut", c] if c else [])])


def train(name: str, base: str, extra: list[str]) -> None:
    for c in CUTS:
        f = PREDS / (f"{name}_cut{c[:4]}.parquet" if c else f"{name}.parquet")
        if f.exists():
            continue
        log(f"  학습 {name} {c or '2024-25'}")
        run(["src_tools/train_site_model.py", "--exp", name, "--base", base, *extra, *(["--cut", c] if c else [])],
            W / "xbatch" / f"{name}_{(c or '')[:4]}.log")


def main() -> None:
    log("밤샘 자동 진행 시작")
    done = {k: wait_for(f, t) for k, (f, t) in {
        "g8w": ("run_smpw.log", "SMP 가중 끝"), "mlp": ("run_mlp.log", "MLP 끝"),
        "s4": ("run_gfs18.log", "GFS 이전 실행 끝"), "aer": ("run_fe_gpu.log", "GPU 피처 걸러내기 끝")}.items()}
    log(f"끝난 실험: {done}")
    tests = {"g8w": ("G5bT1x2", "G8wx2"), "aer": ("E11-11gx2", "E13-11agx2"),
             "s4": ("E12-10cx2", "E13-11x2"), "mlp": ("E12-10cx2", "E13-12x2")}
    ok = {}
    for k, (b, c) in tests.items():
        ok[k], r = judge(b, c) if done[k] else (False, [])
        log(f"  {k}: {c} vs {b} = {fmt(r)} → {'통과' if ok[k] else '불통과'}")

    def members_for(keys: set[str]) -> list[str]:
        m = list(BASE)
        if "g8w" in keys:
            m[m.index("G5bT1x2")] = "G8wx2"
        if "aer" in keys:
            m[m.index("E11-11x2")] = "E13-11agx2"
        if "s4" in keys:
            m.append("S4x2")
        if "mlp" in keys:
            m.append("M1x2")
        return m

    passed = {k for k, v in ok.items() if v}
    if not passed:
        log("통과한 변경 없음 → 끝 (후보 E12-10c 유지)")
        return
    cand, members = "E14x2", members_for(passed)
    blend(cand, members)
    good, r = judge("E12-10cx2", cand)
    log(f"E14x2 {members} vs E12-10cx2 = {fmt(r)} → {'통과' if good else '불통과'}")
    if not good and len(passed) > 1:
        best = None
        for k in sorted(passed):
            name = f"E14{k}x2"
            blend(name, members_for({k}))
            g, rr = judge("E12-10cx2", name)
            log(f"  {name} vs E12-10cx2 = {fmt(rr)} → {'통과' if g else '불통과'}")
            if g and (best is None or max(rr) < best[1]):
                best = (name, max(rr), members_for({k}))
        if best is None:
            log("하나씩 적용한 판도 통과 못 함 → 끝")
            return
        cand, members = best[0], best[2]
    elif not good:
        log("끝 (후보 E12-10c 유지)")
        return
    log(f"후보 {cand}: {members}")

    run(["src_tools/optimize_composition.py", "--preds", cand], W / f"overnight_comp_{cand}.log")
    run(["src_tools/optimize_composition.py", "--preds", cand, "--swap"], W / f"overnight_swap_{cand}.log")
    sw = re.findall(r"1개 옮기기판 대비 ([+-]\d+\.\d+)%p", (W / f"overnight_swap_{cand}.log").read_text(encoding="utf-8"))
    use_swap = len(sw) == 3 and all(float(v) < 0 for v in sw)
    comp = ROOT / "outputs" / "compositions" / (f"{cand}_swap.csv" if use_swap else f"{cand}.csv")
    log(f"구성: 맞바꾸기 {sw} → {'맞바꾸기판' if use_swap else '1개 옮기기판'} ({comp.name})")

    from compare_own_comp import pen
    b = pen("E11-10T1x2", str(ROOT / "outputs" / "compositions" / "E11-10T1x2.csv"))
    c = pen(cand, str(comp))
    pct = (sum(c) / sum(b) - 1) * 100
    log(f"최고 제출 E11-10T1 대비(자기 구성): " + " / ".join(f"{(y / x - 1) * 100:+.2f}%" for x, y in zip(b, c))
        + f", 세 해 합 {pct:+.2f}% ({sum(c) - sum(b):+,.0f}원)")
    if pct > -3:
        log("3% 기준 미달 → 제출 파일 안 만듦. 끝")
        return

    log("3% 기준 넘음 → 시드 재현")
    s2 = {"G8wx2": ("G8ws2x2", "G8wx2"), "S4x2": ("S4s2x2", "S4x2"), "M1x2": ("M1s2x2", "M1x2"),
          "E13-11agx2": ("E13-11ags2x2", "E13-11agx2"), "S3x2": ("S3s2x2", None)}
    rep = []
    for m in members:
        if m in s2:
            name, base = s2[m]
            if base:
                train(name, base, ["--seed", "2"])
            rep.append(name)
        else:
            rep.append(m)
    blend(f"{cand[:-2]}s2x2", rep)
    # 재현 기준: E12-10c 에서 S3 → S3s2, 자리를 바꾼 구성원은 그 기준 모델도 시드 2·3 판으로(같은 조건 비교)
    brep = ["E11-12x2", "E11-11x2", "G5bT1x2", "S3s2x2", "V07T1x2"]
    if "G8wx2" in members:
        brep[2] = "G5bT1s2x2"
    if "E13-11agx2" in members:
        train("E11-11gs2x2", "E11-11gx2", ["--seed", "2"])
        brep[1] = "E11-11gs2x2"
    blend("E12-10cs2Bx2", brep)
    g, rr = judge("E12-10cs2Bx2", f"{cand[:-2]}s2x2")
    log(f"재현 {cand[:-2]}s2x2 {rep} vs E12-10cs2Bx2 {brep} = {fmt(rr)} → {'통과' if g else '불통과'}")
    if not g:
        log("재현 실패 → 제출 파일 안 만듦. 끝")
        return

    log("제출 파일 만들기")
    for m in members:
        if not (ROOT / "outputs" / "preds_test" / f"{m}.parquet").exists():
            log(f"  {m} 평가 기간 학습")
            run(["src_tools/train_site_model.py", "--base", m, "--mode", "test"], W / "xbatch" / f"{m}_test.log")
    blend(cand, members, mode="test")
    rc = run(["src_tools/make_submission.py", str(ROOT / "outputs" / "preds_test" / f"{cand}.parquet"),
              "--composition", str(comp), "--name", f"{cand}_comp"], W / f"overnight_submit_{cand}.log")
    if rc == 0:
        (ROOT / "outputs" / "submissions" / f"{cand}_comp" / ".done").touch()
        log(f"제출 파일 끝: outputs/submissions/{cand}_comp (제출은 사용자)")
    else:
        log("  !! 제출 파일 실패")
    log("끝")


if __name__ == "__main__":
    main()

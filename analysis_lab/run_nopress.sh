#!/usr/bin/env bash
# 1번: A 기압 판단 자료 (11/11 사용자: 2번 파이프라인 점검 다음에). **채택 여부는 사용자가 정한다**(11/8 결정이 규정 §6 해석에 걸려 있음).
# A 기압 정의가 평가 기간 중간(2026-02)에 바뀐 것은 11/8 에 평가 기간 분포를 보다 알게 됨 → 내부 규칙으로 판정하지 않고 자료만 만든다.
#   - 기압을 쓰는 구성원 4개(E11-12, E11-11, G5bT1, V07T1)를 기압만 빼고 다시 학습(CPU, 같은 장치·같은 설정). S3 는 기압을 안 씀.
#   - E15NPx2 = 기압 뺀 판 앙상블 vs E12-10cx2 세 해(참조 구성), 각자 구성 E11-10T1 대비
#   - 평가 기간 예측(시험 모드)도 만들어 E12-10cfx2 와 달별 합계 차이(%)를 표로 — 평가 기간 '예측'끼리의 비교이며 실측은 없음
#   bash analysis_lab/run_nopress.sh >> outputs/work/run_nopress.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
run() {
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
for m in V07T1x2 G5bT1x2 E11-11x2 E11-12x2; do
  run "${m%x2}npx2" "$m" --drop wa_ground_press
done
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E15NPx2 E11-12npx2 E11-11npx2 G5bT1npx2 S3x2 V07T1npx2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E15NPx2 | tail -2
for m in V07T1npx2 G5bT1npx2 E11-11npx2 E11-12npx2; do
  [ -f "outputs/preds_test/${m}.parquet" ] && continue
  log "$m 평가 기간 학습"
  $PY src_tools/train_site_model.py --base "$m" --mode test > "outputs/work/xbatch/${m}_test.log" 2>&1 || log "  !! 실패 $m"
done
$PY src_tools/blend.py --out E15NPx2 E11-12npx2 E11-11npx2 G5bT1npx2 S3x2 V07T1npx2 --mode test > /dev/null 2>&1
$PY - <<'PY'
import pandas as pd
a = pd.read_parquet("outputs/preds_test/E12-10cfx2.parquet"); b = pd.read_parquet("outputs/preds_test/E15NPx2.parquet")
m = a.merge(b, on=["site_id", "ts"], suffixes=("_p", "_np"))
mon = pd.to_datetime(m.ts).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m")
t = m.groupby(mon)[["pred_kwh_p", "pred_kwh_np"]].sum()
print("평가 기간 달별 예측 합계: 기압 뺀 판 / 기압 포함 판 − 1 (%)")
print(((t.pred_kwh_np / t.pred_kwh_p - 1) * 100).round(2).to_string())
print(f"연간 {(t.pred_kwh_np.sum() / t.pred_kwh_p.sum() - 1) * 100:+.2f}%")
PY
log "기압 자료 끝"

#!/usr/bin/env bash
# 제출 파일 (11/10 사용자 결정: E11-10T2 결과를 보고 만든다)
#   E11-10T2 가 E11-10T1 보다 세 해 모두 나으면(compare_years "통과") T2, 아니면 T1.
#   T2: E11-12T2 + E11-11T2 + G5bT1 + S2T1 + V07T1   T1: E11-12 + E11-11 + G5bT1 + S2T1 + V07T1   (같은 비중)
# 순서: run_tune_xgb_apply.sh 의 "XGBoost 적용 끝" 기다림 → 시드 재현 단계 잠시 멈춤 → 평가 기간 학습·섞기·구성 → 제출 파일
#       → 재현 단계 다시 실행(이어서). 끝난 평가 기간 예측은 건너뜀.
#   bash analysis_lab/run_submit_T.sh >> outputs/work/run_submit_T.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "XGBoost 적용 끝" outputs/work/run_tune_xgb_apply.log 2>/dev/null; do sleep 60; done
if [ ! -f outputs/work/submit_T.choice ]; then
  if grep -aq "E11-10T2x2.*통과" outputs/work/run_tune_xgb_apply.log; then echo T2 > outputs/work/submit_T.choice; else echo T1 > outputs/work/submit_T.choice; fi
fi
C=$(cat outputs/work/submit_T.choice)
if [ "$C" = T2 ]; then M=(E11-12T2x2 E11-11T2x2 G5bT1x2 S2T1x2 V07T1x2); else M=(E11-12x2 E11-11x2 G5bT1x2 S2T1x2 V07T1x2); fi
log "제출 후보 E11-10${C}x2: ${M[*]}"
if [ ! -f "outputs/submissions/E11-10${C}x2_comp/.done" ]; then
  log "시드 재현 단계 잠시 멈춤"
  powershell -NoProfile -ExecutionPolicy Bypass -File analysis_lab/stop_repro.ps1
  for name in "${M[@]}"; do
    [ -f "outputs/preds_test/${name}.parquet" ] && continue
    log "$name 평가 기간 학습"
    $PY src_tools/train_site_model.py --base "$name" --mode test > "outputs/work/xbatch/${name}_test.log" 2>&1 || log "  !! 실패 $name"
  done
  log "섞기"
  $PY src_tools/blend.py --out "E11-10${C}x2" "${M[@]}" --mode test
  log "제출 파일"
  $PY src_tools/make_submission.py "outputs/preds_test/E11-10${C}x2.parquet" --composition "outputs/compositions/E11-10${C}x2.csv" \
    --name "E11-10${C}x2_comp" && touch "outputs/submissions/E11-10${C}x2_comp/.done"
  log "제출 파일 끝"
fi
log "시드 재현 다시 시작"
bash analysis_lab/run_tune_xgb_apply.sh >> outputs/work/run_tune_xgb_apply.log 2>&1
log "재현 끝"

#!/usr/bin/env bash
# E9 — A 기압을 다시 넣은 후보 (2026-11-08 결정: 규정 §6 "평가 기간 전체를 보고 규칙을 정하면 안 된다").
# 기압 제외(E4-07)는 평가 기간 예보 분포를 보고 정한 것이라 되돌린다. 나머지 설정은 E8-04x2 와 같다.
#
#   bash analysis_lab/E9_기압포함.sh            (저장소 맨 위에서. PY 로 bio_proj 파이썬 지정 가능)
#
# E9-00x2 = LightGBM, 하루 요약 + 앞뒤 시각(X03), 계절 피처만 제외   ← 새 단일 모델 기준
# E9-11x2 = CatBoost, 같은 피처
# E9-12x2 = XGBoost, 같은 피처, CSI 타깃
# E9-04x2 = E9-11x2 + E9-12x2 반반 → 구성 최적화 → 제출 파일 outputs/submissions/E9-04x2_comp
set -euo pipefail
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
KEEP=(--reset-drop --drop doy_sin --drop doy_cos)
log() { echo "[$(date +%H:%M)] $*"; }

for spec in "E9-00x2 X03x2" "E9-11x2 X03catx2" "E9-12x2 X20xgbx2"; do
  set -- $spec
  for cut in "" 2023-06-01 2022-06-01; do
    name=$1; [ -n "$cut" ] && name="$1_cut${cut:0:4}"
    if [ -f "outputs/preds/$name.parquet" ]; then continue; fi
    log "$1 (base $2) ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$1" --base "$2" "${KEEP[@]}" ${cut:+--cut $cut} > "outputs/work/xbatch/${name}.log" 2>&1
  done
done

log "섞기 E9-04x2 (세 해)"
for cut in "" 2023-06-01 2022-06-01; do
  $PY src_tools/blend.py --out E9-04x2 E9-11x2 E9-12x2 ${cut:+--cut $cut} > /dev/null 2>&1
done

log "기압 제외판과 비교 (같은 시드)"
$PY src_tools/compare_years.py X03x2 E9-00x2
$PY src_tools/compare_years.py X03catx2 E9-11x2
$PY src_tools/compare_years.py X20xgbx2 E9-12x2
$PY src_tools/compare_years.py E8-04x2 E9-04x2

log "구성 최적화"
$PY src_tools/optimize_composition.py --preds E9-04x2

log "평가 기간 예측"
$PY src_tools/train_site_model.py --base E9-11x2 --mode test > outputs/work/xbatch/E9-11x2_test.log 2>&1
$PY src_tools/train_site_model.py --base E9-12x2 --mode test > outputs/work/xbatch/E9-12x2_test.log 2>&1
$PY src_tools/blend.py --out E9-04x2 E9-11x2 E9-12x2 --mode test
$PY src_tools/make_submission.py outputs/preds_test/E9-04x2.parquet --composition outputs/compositions/E9-04x2.csv --name E9-04x2_comp
log "완료"

#!/usr/bin/env bash
# 제출 후보 E11-10: 다섯 모델 같은 비중 + E11-10 오차로 정한 구성
#   E11-12x2 XGBoost CSI · E11-11x2 XGBoost 발전율 · G5bx2 LightGBM 발전율  (V07 + GFS 지역 + 구름 높이별 운량)
#   S2x2  GFS 만 쓰는 LightGBM · V07x2 지역 예보만 쓰는 LightGBM          (HEFTCom2024 1위의 예보 출처별 모델)
# 평가 기간 예측(2023-06 ~ 2025-05 학습) → 섞기 → 구성 → 제출 파일. 이미 있으면 건너뜀.
#   bash analysis_lab/run_submit_E11-10.sh > outputs/work/run_submit_E11-10.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for name in G5bx2 S2x2 V07x2 E11-11x2 E11-12x2; do
  [ -f "outputs/preds_test/${name}.parquet" ] && continue
  log "$name 평가 기간 학습"
  $PY src_tools/train_site_model.py --base "$name" --mode test > "outputs/work/xbatch/${name}_test.log" 2>&1 || log "  !! 실패 $name"
done
log "섞기"
$PY src_tools/blend.py --out E11-10x2 E11-12x2 E11-11x2 G5bx2 S2x2 V07x2 --mode test
log "제출 파일"
$PY src_tools/make_submission.py outputs/preds_test/E11-10x2.parquet --composition outputs/compositions/E11-10x2.csv --name E11-10x2_comp
log "끝"

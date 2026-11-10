#!/usr/bin/env bash
# 제출 파일 E12-10cx2_comp — E11-10T1 의 GFS 전용 모델(S2T1) 자리를 GEFS 전용 모델(S3)로 바꾼 다섯 모델 같은 비중
# + 이 앙상블 오차로 정한 구성(outputs/compositions/E12-10cx2.csv). 나머지 네 모델의 평가 기간 예측은 E11-10T1 제출 때 만든 것(CPU) 그대로.
#   bash analysis_lab/run_submit_E12-10c.sh >> outputs/work/run_submit_E12-10c.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
M=(E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2)
log "제출 후보 E12-10cx2: ${M[*]}"
for name in "${M[@]}"; do
  [ -f "outputs/preds_test/${name}.parquet" ] && continue
  log "$name 평가 기간 학습"
  $PY src_tools/train_site_model.py --base "$name" --mode test > "outputs/work/xbatch/${name}_test.log" 2>&1 || log "  !! 실패 $name"
done
log "섞기"
$PY src_tools/blend.py --out E12-10cx2 "${M[@]}" --mode test
log "제출 파일"
$PY src_tools/make_submission.py outputs/preds_test/E12-10cx2.parquet --composition outputs/compositions/E12-10cx2.csv \
  --name E12-10cx2_comp && touch outputs/submissions/E12-10cx2_comp/.done
log "제출 파일 끝"

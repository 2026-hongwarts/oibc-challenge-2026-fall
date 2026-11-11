#!/usr/bin/env bash
# MLP 두 번째(M2) 시드 재현 + 제출 파일 (11/11 사용자 GPU 허락).
# 미리 정한 판정: E13-13s2x2 = E12-10c + M2s2x2(시드 2·3) vs E12-10cx2, 세 해 모두 개선 → 통과하면
#   평가 기간 M2 학습 → E13-13 섞기(빠짐 처리) → 일출·일몰 → 추세 → 제출 파일 E13-13ftrx2_comp(구성 E12-10cx2.csv) + 공식 형식 검사
#   bash analysis_lab/run_mlp2_repro.sh >> outputs/work/run_mlp2_repro.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/M2s2x2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "M2s2x2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp M2s2x2 --base M2x2 --seed 2 ${cut:+--cut $cut} \
    > "outputs/work/xbatch/M2s2x2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-13s2x2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 M2s2x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
R=$($PY src_tools/compare_years.py E12-10cx2 E13-13s2x2 | tail -1); echo "$R"
if ! echo "$R" | grep -q "통과"; then log "M2 재현 불통과 → 끝"; log "M2 재현 끝"; exit 0; fi
[ -f outputs/preds_test/M2x2.parquet ] || { log "M2x2 평가 기간 학습"; $PY src_tools/train_site_model.py --base M2x2 --mode test > outputs/work/xbatch/M2x2_test.log 2>&1; }
$PY src_tools/blend.py --out E13-13fx2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 M2x2 --mode test 2>&1 | grep -a "빠짐\|저장"
$PY src_tools/twilight_fix.py --preds E13-13fx2 --out E13-13ftx2 2>&1 | grep -a "preds_test"
$PY src_tools/trend_fix.py --preds E13-13ftx2 --out E13-13ftrx2 2>&1 | grep -a "test"
$PY src_tools/make_submission.py outputs/preds_test/E13-13ftrx2.parquet --composition outputs/compositions/E12-10cx2.csv --name E13-13ftrx2_comp 2>&1 | tail -2
touch outputs/submissions/E13-13ftrx2_comp/.done
$PY outputs/work/validate_submission.py --composition outputs/submissions/E13-13ftrx2_comp/composition.csv \
  --forecast outputs/submissions/E13-13ftrx2_comp/forecast.csv --data 01_participant_dataset | head -1
log "M2 재현 끝"

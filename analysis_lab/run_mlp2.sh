#!/usr/bin/env bash
# MLP 두 번째 시도(첫 시도 M1 은 G5bT1 보다 +25% — 레시피 부족). 조사 권장대로 발전소 임베딩(--cat site_id, 처음 보는 발전소는
# 학습된 임베딩 평균) + 학습량 늘림(60에폭, 은닉 1024, 배치 2048, 학습률 1e-3).
# 미리 정한 판정: E13-13x2 = E12-10c + M2x2 (여섯 같은 비중) vs E12-10cx2, 세 해 모두 개선.
#   bash analysis_lab/run_mlp2.sh >> outputs/work/run_mlp2.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
# GPU 는 사용자 허락 뒤에만(11/11 사용자): outputs/work/gpu.ok 가 생길 때까지 기다린다
[ -f outputs/work/gpu.ok ] || log "GPU 허락 기다림 (outputs/work/gpu.ok)"
until [ -f outputs/work/gpu.ok ]; do sleep 60; done
until grep -aq "D1 끝" outputs/work/run_recheck.log 2>/dev/null; do sleep 60; done   # GPU 하나씩
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/M2x2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "M2x2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp M2x2 --base M1x2 --cat site_id \
    --params '{"epochs": 60, "hidden": 1024, "batch": 2048, "lr": 0.001}' ${cut:+--cut $cut} \
    > "outputs/work/xbatch/M2x2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py G5bT1x2 M1x2 M2x2 | tail -2
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-13x2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 M2x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E13-13x2 | tail -1
log "MLP2 끝"

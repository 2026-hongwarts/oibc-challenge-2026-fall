#!/usr/bin/env bash
# MLP 보조 모델 (11/11 사용자 승인) — G5bT1x2 와 같은 피처(247개)를 PyTorch MLP(GPU)로. src_tools/mlp_model.py
# 미리 정한 판정: E13-12x2 = E12-10c 다섯 모델 + M1x2 (여섯 같은 비중, MLP 비중 1/6) 를 E12-10cx2 와 참조 구성으로 비교 → 세 해 모두 개선이면 통과.
#   (M1x2 단독은 G5bT1x2 와 비교만 해 둔다 — 혼자서는 GBM 보다 못해도 됨)
# 메모리: CPU 학습(G8w)이 끝난 뒤 시작.   bash analysis_lab/run_mlp.sh >> outputs/work/run_mlp.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "SMP 가중 끝" outputs/work/run_smpw.log 2>/dev/null; do sleep 60; done
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/M1x2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "M1x2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp M1x2 --base G5bT1x2 --model mlp ${cut:+--cut $cut} \
    > "outputs/work/xbatch/M1x2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py G5bT1x2 M1x2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-12x2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 M1x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E13-12x2 | tail -1
log "MLP 끝"

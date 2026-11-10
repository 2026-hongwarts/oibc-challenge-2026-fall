#!/usr/bin/env bash
# GFS 이전 실행(D-2 18시 UTC) 전용 모델 S4 — 출처별 전용 모델 확장(S2 = GFS 00시, S3 = GEFS, S4 = GFS 18시 이전 실행).
# 받기 → 해석 → (CPU 가 비면) S4x2 = S2T1x2 에서 GFS 대신 GFS 이전 실행 → E13-11x2 = E12-10c + S4 (여섯 같은 비중).
# 미리 정한 판정: E13-11x2 를 E12-10cx2 와 참조 구성(compare_years)으로 비교 → 세 해 모두 개선이면 통과.
#   bash analysis_lab/run_gfs18.sh >> outputs/work/run_gfs18.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
GPY=/c/Users/109so/miniconda3/envs/gfs/python.exe
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
if ! grep -aq "1825/1825일" outputs/work/gfs18_fetch.log 2>/dev/null; then
  log "받기"
  $PY -u src_tools/gfs_fetch.py --source gfs18 --workers 16 >> outputs/work/gfs18_fetch.log 2>&1
fi
grep -a -c "실패" outputs/work/gfs18_fetch.log
if [ ! -f outputs/work/gfs18_kr.parquet ]; then
  log "해석"
  $GPY src_tools/gfs_decode.py --raw outputs/work/gfs18_raw --step-shift 6 --prefix g18 \
    --out outputs/work/gfs18_kr.parquet > outputs/work/gfs18_decode.log 2>&1 || log "  !! 해석 실패"
  tail -2 outputs/work/gfs18_decode.log
  cp outputs/work/gfs18_raw/manifest.csv outputs/gfs18_manifest.csv
fi
until grep -aq "SMP 가중 끝" outputs/work/run_smpw.log 2>/dev/null; do sleep 60; done   # CPU 학습은 하나씩
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/S4x2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "S4x2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp S4x2 --base S2T1x2 --reset-add --add features_gfs:add_gfs18_regions ${cut:+--cut $cut} \
    > "outputs/work/xbatch/S4x2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py S2T1x2 S4x2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-11x2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 S4x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E13-11x2 | tail -1
log "GFS 이전 실행 끝"

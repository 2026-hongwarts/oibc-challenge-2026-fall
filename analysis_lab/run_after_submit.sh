#!/usr/bin/env bash
# 제출 파일 끝 → T2 시드 재현 멈춤(T2 기각이라 필요 없음) → GPU 확인 → GEFS 해석 끝 기다림 → run_gefs.sh
# 학습 하나가 메모리를 16~21GB 쓰므로(전체 32GB) 학습은 하나씩 차례로 돌린다.
# GPU 확인: E11-11x2 를 --device cuda 로 2024-25 만 다시 학습 → CPU 결과(E11-11x2)와 벌금 비교·걸린 시간.
#   bash analysis_lab/run_after_submit.sh >> outputs/work/run_after_submit.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "제출 파일 끝" outputs/work/run_submit_T.log; do sleep 30; done
for i in 1 2 3; do
  powershell -NoProfile -ExecutionPolicy Bypass -File analysis_lab/stop_repro.ps1
  sleep 20
done
log "T2 시드 재현 멈춤"
if [ ! -f outputs/preds/E11-11gx2.parquet ]; then
  log "GPU 확인: E11-11gx2 2024-25 (--device cuda)"
  t=$(date +%s)
  $PY src_tools/train_site_model.py --exp E11-11gx2 --base E11-11x2 --device cuda \
    > outputs/work/xbatch/E11-11gx2_.log 2>&1 || log "  !! 실패 GPU 확인"
  log "  걸린 시간 $(( ($(date +%s) - t) / 60 ))분 (CPU 는 E11-11T2x2 2024-25 기준 약 49분)"
  $PY src_tools/compare_years.py E11-11x2 E11-11gx2 | tail -2
fi
# 해석 로그는 출력이 버퍼에 있다가 끝날 때 한꺼번에 써지므로, 로그가 비어 있지 않으면 해석이 끝난 것
until [ -s outputs/work/gefs_decode.log ]; do sleep 60; done
tail -3 outputs/work/gefs_decode.log
[ -f outputs/gefs_ens_manifest.csv ] || cp outputs/work/gefs_raw/manifest.csv outputs/gefs_ens_manifest.csv
log "GEFS 실험 시작"
bash analysis_lab/run_gefs.sh >> outputs/work/run_gefs.log 2>&1
log "끝"

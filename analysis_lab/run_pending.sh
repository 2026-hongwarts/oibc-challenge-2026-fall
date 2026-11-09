#!/usr/bin/env bash
# 남은 1단계 실험을 순서대로 실행 (중간에 끊겨도 다시 실행하면 끝난 결과는 건너뛰고 이어서 함).
#   bash analysis_lab/run_pending.sh > outputs/work/run_pending.log 2>&1
# 1) GFS 해석(마감 뒤 공개 파일 제외) → 2) G1·G2 GFS 피처 → 3) A1 2단계 보조 모델 → 4) T1 시간대 전용 모델 + 섞기
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
G=${G:-/c/Users/109so/miniconda3/envs/gfs/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

if [ ! -f outputs/work/gfs_kr.done ] || [ outputs/work/gfs_raw/manifest.csv -nt outputs/work/gfs_kr.done ]; then
  log "GFS 해석"
  $G -u src_tools/gfs_decode.py > outputs/work/gfs_decode.log 2>&1 && touch outputs/work/gfs_kr.done
  tail -2 outputs/work/gfs_decode.log
fi
log "G1·G2 (GFS)";            $PY -u analysis_lab/V_피처개선_배치.py --set gfs >> outputs/work/vbatch5.log 2>&1; grep "x2 " outputs/work/vbatch5.log | tail -2
log "A1 (2단계 보조 모델)";   $PY -u analysis_lab/V_피처개선_배치.py --set aux >> outputs/work/vbatch6.log 2>&1; grep "x2 " outputs/work/vbatch6.log | tail -1
log "T1 (시간대 전용 모델)";  $PY -u analysis_lab/V_피처개선_배치.py --set seg >> outputs/work/vbatch7.log 2>&1
for w in 0.5 1.0; do
  n=$([ "$w" = "0.5" ] && echo T1x2 || echo T1fx2)
  for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
    $PY src_tools/blend_regime.py --base E9-00x2 --spec T1sx2 --where "hour >= 11 and hour <= 15" --w $w --out $n $c > /dev/null 2>&1
  done
done
$PY src_tools/compare_years.py E9-00x2 T1x2 T1fx2
log "1단계 끝"

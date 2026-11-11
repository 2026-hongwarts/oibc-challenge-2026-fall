#!/usr/bin/env bash
# 수상 해법 근거 GFS 피처 걸러내기 — GPU XGBoost(E11-11gx2, XGB 발전율) 위에서 빠르게 (11/10 사용자 승인).
# 미리 정한 판정: 각 실험을 E11-11gx2 대비 세 해 비교 → 세 해 모두 개선이면 통과(그다음 LightGBM·앙상블에서 확인).
#   E13-11dgx2  + add_gfs_daily      GFS 하루 합·평균 (AMS 2013-14 1·3위)
#   E13-11ngx2  + add_gfs_neighbors  GFS 3시간 전·뒤 값 (HEFTCom2024 1위 SVK·4위 GEB)
#   E13-11pgx2  + add_gfs_pwat       GFS 가강수량 (AMS 2013-14 2위)        ← PWAT 받기 끝난 뒤
#   E13-11agx2  + add_aer_regions    GEFS-Aerosols 광학두께                ← 에어로졸 받기 끝난 뒤
# 메모리: GPU 학습 하나 + CPU(run_gefs.sh) 학습 하나까지만. 끝난 것은 건너뜀.
#   bash analysis_lab/run_fe_gpu.sh >> outputs/work/run_fe_gpu.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
GPY=/c/Users/109so/miniconda3/envs/gfs/python.exe
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
# GPU 는 사용자 허락 뒤에만(11/11 사용자): outputs/work/gpu.ok 가 생길 때까지 기다린다
[ -f outputs/work/gpu.ok ] || log "GPU 허락 기다림 (outputs/work/gpu.ok)"
until [ -f outputs/work/gpu.ok ]; do sleep 60; done
run() {
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
  $PY src_tools/compare_years.py E11-11gx2 "$name" | tail -1
}
until grep -aq "GEFS XGBoost 끝" outputs/work/run_gefs_xgb.log 2>/dev/null; do sleep 60; done
run E13-11dgx2 E11-11gx2 --add features_gfs:add_gfs_daily
run E13-11ngx2 E11-11gx2 --add features_gfs:add_gfs_neighbors
log "GFS 하루 요약·앞뒤 블록 끝"
until grep -aq "1825/1825일" outputs/work/pwat_fetch.log 2>/dev/null; do sleep 60; done
if [ ! -f outputs/work/gfs_pwat.parquet ]; then
  $GPY src_tools/gfs_decode.py --raw outputs/work/gfs_raw --vars PWAT --inst PWAT \
    --out outputs/work/gfs_pwat.parquet > outputs/work/gfs_pwat_decode.log 2>&1 || log "  !! PWAT 해석 실패"
  tail -1 outputs/work/gfs_pwat_decode.log
  cp outputs/work/gfs_raw/manifest.csv outputs/gfs_manifest.csv
fi
run E13-11pgx2 E11-11gx2 --add features_gfs:add_gfs_pwat
log "PWAT 끝"
until grep -aq "1825/1825일" outputs/work/chem_fetch.log 2>/dev/null; do sleep 60; done
if [ ! -f outputs/work/aer_kr.parquet ]; then
  $GPY src_tools/gfs_decode.py --raw outputs/work/chem_raw --vars AOD DUAOD --inst AOD DUAOD --prefix aer \
    --out outputs/work/aer_kr.parquet > outputs/work/aer_decode.log 2>&1 || log "  !! 에어로졸 해석 실패"
  tail -1 outputs/work/aer_decode.log
  cp outputs/work/chem_raw/manifest.csv outputs/aer_manifest.csv
fi
run E13-11agx2 E11-11gx2 --add features_gfs:add_aer_regions
log "GPU 피처 걸러내기 끝"

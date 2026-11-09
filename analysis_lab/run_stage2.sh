#!/usr/bin/env bash
# 2단계: 통과한 변경을 합친 E10 계열 — V07(기여 없는 11개 + 중복 6개 빼기) + G2(GFS 1° 지역 42곳 + 전체 평균).
# 모델 × 타깃 2×2 와 재현(시드 2·3)을 세 해로. 이미 결과가 있으면 건너뜀(다시 실행하면 이어서).
#   bash analysis_lab/run_stage2.sh > outputs/work/run_stage2.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
V07=(wa_wind_speed wb_wind_speed wb_vis wb_total_liq wb_precip_prob wx_prev_cloud wx_prev_rain wx_prev_trange
     wx_day_rain wx_am_vis_min wx_day_trange cosz wb_dew_point_r10 wb_temp_r10 wb_rel_hum_r10 wa_ground_press_r10 wa_temp_r10)
COMMON=(--add features_gfs:add_gfs_regions)
for d in "${V07[@]}"; do COMMON+=(--drop "$d"); done

run() {   # run 이름 base 추가옵션...
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "${COMMON[@]}" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}

run E10-00x2   E9-00x2                     # LightGBM, 발전율
run E10-00s2x2 E9-00x2 --seed 2            # 재현 (비교: E9-00s2x2)
run E10-12x2   E9-12x2                     # XGBoost, CSI
run E10-02x2   E9-00x2 --target csi        # LightGBM, CSI
run E10-11x2   E9-12x2 --target cf         # XGBoost, 발전율
log "비교"
$PY src_tools/compare_years.py E9-00x2 E10-00x2
$PY src_tools/compare_years.py E9-00s2x2 E10-00s2x2
$PY src_tools/compare_years.py E9-12x2 E10-12x2
$PY src_tools/compare_years.py E10-00x2 E10-02x2 E10-11x2 E10-12x2
log "2단계 끝"

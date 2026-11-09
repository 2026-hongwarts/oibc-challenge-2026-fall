#!/usr/bin/env bash
# 최종 후보 E10-08: XGBoost CSI(E10-12) + XGBoost 발전율(E10-11) + LightGBM 발전율(E10-00) 1/3 씩
#   피처 = 기준(하루 요약 + 앞뒤 시각 + A 기압) − V07(기여 없는 11개 + 중복 6개) + G2(GFS 1° 지역 42곳 + 평균)
# 평가 기간 예측 → 섞기 → 구성(outputs/compositions/E10-08x2.csv) → 제출 파일. 이미 있으면 건너뜀.
#   bash analysis_lab/run_final_E10-08.sh > outputs/work/run_final.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

# 다른 곳에서 돌고 있는 평가 기간 학습이 있으면 끝날 때까지 기다림
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--mode test*' }) { exit 0 } else { exit 1 }"; do sleep 20; done

for name in E10-00x2 E10-12x2 E10-11x2; do
  [ -f "outputs/preds_test/${name}.parquet" ] && continue
  log "$name 평가 기간 학습"
  $PY src_tools/train_site_model.py --base "$name" --mode test > "outputs/work/xbatch/${name}_test.log" 2>&1 || log "  !! 실패 $name"
done
log "섞기"
$PY src_tools/blend.py --out E10-08x2 E10-12x2 E10-11x2 E10-00x2 --mode test
until [ -f outputs/compositions/E10-08x2.csv ]; do sleep 20; done
log "제출 파일"
$PY src_tools/make_submission.py outputs/preds_test/E10-08x2.parquet --composition outputs/compositions/E10-08x2.csv --name E10-08x2_comp
log "끝"

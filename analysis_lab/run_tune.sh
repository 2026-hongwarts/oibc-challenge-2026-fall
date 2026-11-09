#!/usr/bin/env bash
# 설정값 탐색: LightGBM(G5bx2 설정) 30회 → XGBoost(E11-11x2 설정) 20회.
# 컴퓨터가 꺼져도 다시 실행하면 이어서 함(시도마다 outputs/work/tune_{설정}.db 에 저장, 끝난 시도는 건너뜀).
#   bash analysis_lab/run_tune.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
$PY -u src_tools/tune.py --base G5bx2 --trials 30 >> outputs/work/tune_lgb.log 2>&1
$PY -u src_tools/tune.py --base E11-11x2 --trials 20 >> outputs/work/tune_xgb.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] 탐색 끝" >> outputs/work/tune_lgb.log

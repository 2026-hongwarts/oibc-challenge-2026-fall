#!/usr/bin/env bash
# 11/11 사용자 결정: A 기압 빼기 + 추세 보정 적용. 제출 파일을 먼저 만든다(평가 기간용 학습만), 세 해 검증은 그 뒤.
#   기압을 쓰는 구성원 4개를 기압만 빼고 같은 설정·같은 장치(CPU)로 평가 기간 학습 → E15NPx2 = [E11-12np, E11-11np, G5bT1np, S3, V07T1np]
#   → 빠짐 처리(blend 기본) → 일출·일몰 보정 → (추세 보정) → 구성은 E12-10cx2.csv 그대로(기압 효과만 따로 보이게)
#   제출 파일: E15NPftx2_comp(기압 뺌 + 일출·일몰), E15NPftrx2_comp(+ 추세 보정), E12-10cftrx2_comp(지금 최고 + 추세 보정만)
#   그다음: 세 해 검증(run_nopress.sh 앞부분과 같음) → 부분 정지 빼기 → 에어로졸 → MLP 두 번째
#   bash analysis_lab/run_np_submit.sh >> outputs/work/run_np_submit.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
sub() {   # 예측 이름, 제출 이름
  [ -f "outputs/submissions/$2/.done" ] && return
  $PY src_tools/make_submission.py "outputs/preds_test/$1.parquet" --composition outputs/compositions/E12-10cx2.csv --name "$2" \
    2>&1 | tail -3 && touch "outputs/submissions/$2/.done"
}
# 0) 지금 최고 + 추세 보정 (재학습 없음)
$PY src_tools/trend_fix.py --preds E12-10cftx2 --out E12-10cftrx2 2>&1 | grep -av Warning | grep -a "test"
sub E12-10cftrx2 E12-10cftrx2_comp
# 1) 기압 뺀 구성원 평가 기간 학습
for m in V07T1x2 G5bT1x2 E11-11x2 E11-12x2; do
  n="${m%x2}npx2"
  [ -f "outputs/preds_test/${n}.parquet" ] && continue
  log "$n 평가 기간 학습"
  $PY src_tools/train_site_model.py --exp "$n" --base "$m" --drop wa_ground_press --mode test \
    > "outputs/work/xbatch/${n}_test.log" 2>&1 || log "  !! 실패 $n"
done
$PY src_tools/blend.py --out E15NPfx2 E11-12npx2 E11-11npx2 G5bT1npx2 S3x2 V07T1npx2 --mode test 2>&1 | grep -av Warning
$PY src_tools/twilight_fix.py --preds E15NPfx2 --out E15NPftx2 2>&1 | grep -a "preds_test"
$PY src_tools/trend_fix.py --preds E15NPftx2 --out E15NPftrx2 2>&1 | grep -av Warning | grep -a "test"
sub E15NPftx2 E15NPftx2_comp
sub E15NPftrx2 E15NPftrx2_comp
$PY - <<'PY'
import pandas as pd
a = pd.read_parquet("outputs/preds_test/E12-10cftx2.parquet"); b = pd.read_parquet("outputs/preds_test/E15NPftx2.parquet")
m = a.merge(b, on=["site_id", "ts"], suffixes=("_p", "_np"))
mon = pd.to_datetime(m.ts).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m")
t = m.groupby(mon)[["pred_kwh_p", "pred_kwh_np"]].sum()
print("평가 기간 달별 예측 합계: 기압 뺀 판 / 기압 포함 판 − 1 (%)")
print(((t.pred_kwh_np / t.pred_kwh_p - 1) * 100).round(2).to_string())
print(f"연간 {(t.pred_kwh_np.sum() / t.pred_kwh_p.sum() - 1) * 100:+.2f}%")
PY
log "기압 제출 파일 끝"
# 2) 세 해 검증(기압 뺀 판) → 3) 나머지 순서
bash analysis_lab/run_nopress.sh >> outputs/work/run_nopress.log 2>&1
bash analysis_lab/run_clean.sh >> outputs/work/run_clean.log 2>&1
bash analysis_lab/run_fe_gpu.sh >> outputs/work/run_fe_gpu.log 2>&1
bash analysis_lab/run_mlp2.sh >> outputs/work/run_mlp2.log 2>&1
log "순서 끝"

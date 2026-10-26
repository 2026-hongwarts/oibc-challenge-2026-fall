# OIBC 2026 태양광 집합자원 — 팀 저장소

2026 POSTECH OIBC Big Data Challenge. 태양광 발전소 253곳을 두 묶음(G1, G2)으로 나누고, 묶음별 시간 발전량을 하루 전에 예측한다. 정산금 S에서 수익은 모든 팀이 같으므로 승부는 예측 오차 벌금이다. 마감 2026-11-13.

- 역할 분담·일정: https://claude.ai/artifact/VqqzYusqpUCfzMm7EiFk4i
- 계획서와 근거: [eda_preprocessing_plan.md](eda_preprocessing_plan.md) · EDA 결과: [eda_phase1.ipynb](eda_phase1.ipynb)

## 준비

1. Python 3.10 이상, `pip install pandas pyarrow lightgbm` (청천 일사 작업은 `pvlib`도)
2. 대회 데이터를 `01_participant_dataset/`에 둔다. 데이터는 git에 올리지 않는다.
3. 공통 테이블 만들기 (약 2분, 메모리 약 6GB)
   ```
   python build_base.py
   ```
4. 기준 모델 예측을 각자 PC에서 한 번 만든다. PC나 라이브러리 버전에 따라 숫자가 조금 다를 수 있으니, 비교는 같은 PC에서 만든 결과끼리 한다.
   ```
   python train_site_model.py --exp E4-00x2 --seeds 2
   ```

## 실험

```
python train_site_model.py --exp E3-01 --drop-dead                 # 탐색용 (시드 1개)
python train_site_model.py --exp E3-01x2 --drop-dead --seeds 2     # 채택 판단용 (시드 2개 평균)
python train_site_model.py --exp E2-02 --add features_weather:add_weather_features
python evaluate.py preds/E3-01x2.parquet                           # 채점만 다시
```

- 결과는 `preds/실험번호.parquet`에 저장되고, `preds/champion.txt`가 가리키는 기준 모델과 바로 비교된다.
- 옵션: `--drop-dead` 정지일 제외 · `--rounds both|10|17|avg` · `--years N` 학습 기간 · `--weight cap|cap_k` · `--objective l1` · `--params '{"num_leaves": 63}'` · `--mode test` 실제 평가 기간 예측
- 피처 함수 규칙은 [train_site_model.py](train_site_model.py) 맨 위 설명에 있다. 발전량으로 만드는 값은 `df[df.is_fit]`에서만 계산한다.

## 규칙

- 실험 번호는 `E역할-순번`(예: `E2-01`), 시드 2개 결과는 끝에 `x2`. 한 실험에 바꾸는 것은 하나.
- 시드만 바꿔도 벌금이 약 0.5% 움직인다. 채택은 `--seeds 2` 결과가 기준 모델보다 벌금 0.5% 이상 낮고 11~15시도 낮을 때.
- 금지: 실측 기상(`observed_*`)을 입력으로 쓰기 · 2024-06 이후 발전량으로 만든 값을 학습에 쓰기 · 가상 평가연도로 조기 종료 · 좌표로 실제 위치 찾기(실격) · 대회 데이터 업로드
- git에는 코드·문서·설정 기록(`preds/*.json`)·`preds/champion.txt`만 올린다. 예측 파일과 공통 테이블은 `.gitignore`로 빠진다.

## 파일

| 파일 | 내용 |
|---|---|
| `oibc_common.py` | 공통 함수: 점수식, 데이터 읽기, 태양 기하, 정지일 탐지, 구성 탐색 |
| `build_base.py` | 공통 테이블 `work/base.parquet`(학습·가상 평가연도), `work/base_test.parquet`(실제 평가 기간) |
| `train_site_model.py` | 발전소 단위 LightGBM 학습·예측. 옵션 없이 돌리면 baseline.py 재현(E4-00) |
| `evaluate.py` | 가상 평가연도(2024-06 ~ 2025-05) 채점기. 대회 점수식 그대로 |
| `eda_phase1.ipynb` | EDA v2 (실행 결과 포함) |
| `eda_preprocessing_plan.md` | 계획서 v2 |
| `대회 목표.txt` | 대회 설명 |

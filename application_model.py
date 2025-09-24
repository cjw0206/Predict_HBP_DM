import pandas as pd
import joblib
import os

# ==============================
# 1) 모델 불러오기
# ==============================
# 기존 모델 (antihypertensives 포함 안 된 버전)
# model_hbp = joblib.load("model_save/mlp_hbp_rank1.pkl")
# model_dm  = joblib.load("model_save/mlp_dm_rank1.pkl")

# model_hbp = joblib.load("model_save/xgb_hbp_patients_rank7.pkl")
# model_dm  = joblib.load("model_save/xgb_dm_patients_rank2.pkl")

# sex, birth_year feature 포함된 모델
model_hbp = joblib.load("model_save/model_hbp.pkl")
model_dm  = joblib.load("model_save/model_dm.pkl")

# print("Model size hbp (bytes):", os.path.getsize("model_save/mlp_hbp_patients_rank1.pkl"))
# print("Model size dm (bytes):", os.path.getsize("model_save/mlp_dm_patients_rank1.pkl"))

# ==============================
# 2) 시나리오 정의
# ==============================
scenarios = {
    # ==========================
    # 정상군
    # ==========================
    "정상 남(20대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 115, "dbp": 75, "spo2": 98, "temp": 36.6,
        "weight": 72, "height": 178, "glucose": 88, "pulse": 72,
        "sex": 1, "birth_year": 2000,
    },
    "정상 여(20대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 110, "dbp": 70, "spo2": 98, "temp": 36.5,
        "weight": 55, "height": 165, "glucose": 85, "pulse": 76,
        "sex": 0, "birth_year": 2002,
    },
    "정상 남(40대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 118, "dbp": 78, "spo2": 98, "temp": 36.6,
        "weight": 75, "height": 174, "glucose": 92, "pulse": 73,
        "sex": 1, "birth_year": 1983,
    },
    "정상 여(40대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 116, "dbp": 76, "spo2": 98, "temp": 36.7,
        "weight": 60, "height": 163, "glucose": 90, "pulse": 75,
        "sex": 0, "birth_year": 1985,
    },
    "정상 남(60대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 122, "dbp": 80, "spo2": 97, "temp": 36.6,
        "weight": 70, "height": 170, "glucose": 95, "pulse": 74,
        "sex": 1, "birth_year": 1965,
    },
    "정상 여(60대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 124, "dbp": 82, "spo2": 97, "temp": 36.6,
        "weight": 62, "height": 160, "glucose": 94, "pulse": 76,
        "sex": 0, "birth_year": 1963,
    },

    # ==========================
    # 고혈압
    # ==========================
    "고혈압 남(40대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 155, "dbp": 98, "spo2": 96, "temp": 36.7,
        "weight": 85, "height": 172, "glucose": 100, "pulse": 80,
        "sex": 1, "birth_year": 1980,
    },
    "고혈압 여(70대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 165, "dbp": 105, "spo2": 95, "temp": 36.5,
        "weight": 70, "height": 160, "glucose": 105, "pulse": 82,
        "sex": 0, "birth_year": 1955,
    },
    "약 복용 + 혈압 정상 남(60대)": {
        "antihypertensives": 1, "fasting": 0,
        "sbp": 125, "dbp": 80, "spo2": 97, "temp": 36.6,
        "weight": 72, "height": 170, "glucose": 110, "pulse": 75,
        "sex": 1, "birth_year": 1962,
    },

    # ==========================
    # 당뇨
    # ==========================
    "식후 고혈당 남(50대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 125, "dbp": 82, "spo2": 97, "temp": 36.7,
        "weight": 78, "height": 174, "glucose": 220, "pulse": 78,
        "sex": 1, "birth_year": 1972,
    },
    "공복 혈당 상승 여(30대)": {
        "antihypertensives": 0, "fasting": 1,
        "sbp": 118, "dbp": 78, "spo2": 97, "temp": 36.7,
        "weight": 54, "height": 162, "glucose": 155, "pulse": 70,
        "sex": 0, "birth_year": 1990,
    },

    # ==========================
    # 체중 이슈
    # ==========================
    "비만+혈압↑+혈당↑ 남(40대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 150, "dbp": 95, "spo2": 95, "temp": 36.8,
        "weight": 110, "height": 170, "glucose": 195, "pulse": 88,
        "sex": 1, "birth_year": 1982,
    },
    "저체중+혈당↑ 여(20대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 115, "dbp": 75, "spo2": 93, "temp": 36.4,
        "weight": 42, "height": 170, "glucose": 180, "pulse": 70,
        "sex": 0, "birth_year": 2001,
    },

    # ==========================
    # 맥박 이상
    # ==========================
    "혈압 < 50 남(70대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 130, "dbp": 85, "spo2": 97, "temp": 36.6,
        "weight": 70, "height": 170, "glucose": 95, "pulse": 45,
        "sex": 1, "birth_year": 1950,
    },
    "혈압 > 120 여(40대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 135, "dbp": 88, "spo2": 96, "temp": 36.8,
        "weight": 65, "height": 165, "glucose": 115, "pulse": 130,
        "sex": 0, "birth_year": 1985,
    },

    # ==========================
    # 산소포화도
    # ==========================
    "저산소증 남(60대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 125, "dbp": 80, "spo2": 85, "temp": 36.6,
        "weight": 75, "height": 172, "glucose": 95, "pulse": 88,
        "sex": 1, "birth_year": 1960,
    },
    "저산소증+고혈압 여(70대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 160, "dbp": 105, "spo2": 82, "temp": 36.6,
        "weight": 70, "height": 160, "glucose": 110, "pulse": 95,
        "sex": 0, "birth_year": 1953,
    },

    "저산소증+고혈압 여(70대)": {
        "antihypertensives": 0, "fasting": 0,
        "sbp": 160, "dbp": 105, "spo2": 82, "temp": 36.6,
        "weight": 70, "height": 160, "glucose": 110, "pulse": 95,
        "sex": 0, "birth_year": 1953,
    },

}


# ==============================
# 3) 시나리오별 예측
# ==============================
results = []
for name, patient_info in scenarios.items():
    # DataFrame 변환
    X_dm = pd.DataFrame([patient_info])  # DM 모델은 antihypertensives 포함
    X_hbp = X_dm.drop(columns=["antihypertensives"])  # HBP 모델은 antihypertensives 제외

    # 예측
    score_hbp = model_hbp.predict_proba(X_hbp)[:, 1][0]
    score_dm = model_dm.predict_proba(X_dm)[:, 1][0]

    results.append({
        "시나리오": name,
        "고혈압 위험 점수": round(score_hbp, 4),
        "당뇨 위험 점수": round(score_dm, 4),
    })

# ==============================
# 4) 결과 정리 출력
# ==============================
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

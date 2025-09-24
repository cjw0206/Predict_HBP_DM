import pandas as pd
import numpy as np

def remove_outliers(df):
    """
    의학적으로 불가능한 값이 있으면 그 행 전체를 제거한다.
    단, 값이 0인 경우는 결측치로 간주하므로 이상치 검사에서 제외한다.
    """
    bounds = {
        "sbp": (50, 250),     # 수축기 혈압
        "dbp": (30, 150),     # 이완기 혈압
        "spo2": (50, 100),    # 산소포화도
        "temp": (30, 43),     # 체온
        "weight": (20, 150),  # 체중
        "height": (100, 200), # 키
        "glucose": (40, 400), # 혈당
        "pulse": (30, 200),   # 맥박
    }

    for col, (low, high) in bounds.items():
        if col in df.columns:
            # 값이 0이 아닌 경우만 검사
            mask_outlier = (df[col] != 0) & ((df[col] < low) | (df[col] > high))
            df = df.loc[~mask_outlier].copy()

    return df

def preprocess_vital_accumulated():
    # -----------------------------
    # 1) CSV 로드
    # -----------------------------
    vitals  = pd.read_csv("Data/vitals_accumulated.csv")

    # 키 컬럼 정리
    vitals["name"] = vitals["name"].astype(str).str.strip()

    # -----------------------------
    # 2) 결측치 처리 준비
    # -----------------------------
    target_cols = ["sbp", "dbp", "spo2", "temp", "weight", "height", "glucose", "pulse"]

    # 숫자 변환
    for col in target_cols:
        if col in vitals.columns:
            vitals[col] = pd.to_numeric(vitals[col], errors="coerce")

    # 의학적 이상치 제거 (평균 계산 전에 수행)
    vitals = remove_outliers(vitals)

    # -----------------------------
    # 3) 결측치(0) 개수 3개 까지 허용
    # -----------------------------
    # 0 또는 NaN 이면 결측으로 취급
    missing_mask = vitals[target_cols].apply(lambda row: ((row == 0) | (row.isna())).sum(), axis=1)
    vitals = vitals[missing_mask <= 3].copy()

    # -----------------------------
    # 4) 전체 평균 (0, NaN 제외)
    # -----------------------------
    global_means = {}
    for col in target_cols:
        if col in vitals.columns:
            gm = vitals.loc[(vitals[col].notna()) & (vitals[col] > 0), col].mean()
            global_means[col] = gm if pd.notna(gm) else 0.0

    # -----------------------------
    # 5-1) 환자별 평균으로 0 대체, 없으면 전체 평균
    # -----------------------------
    # for col in target_cols:
    #     if col not in vitals.columns:
    #         continue
    #
    #     # 환자별 비영 평균
    #     per_patient_mean = (
    #         vitals.loc[(vitals[col].notna()) & (vitals[col] > 0)]
    #         .groupby("name")[col]
    #         .mean()
    #     )
    #
    #     mapped_mean = vitals["name"].map(per_patient_mean)
    #     mask_missing = vitals[col].isna() | (vitals[col] == 0)
    #
    #     fill_values = mapped_mean.where(mapped_mean.notna(), global_means[col])
    #     vitals.loc[mask_missing, col] = fill_values[mask_missing]


    # -----------------------------
    # 5-2) 환자별 평균으로 0 대체, 없으면 제거
    # -----------------------------
    for col in target_cols:
        if col not in vitals.columns:
            continue

        # 환자별 비영 평균 계산
        per_patient_mean = (
            vitals.loc[(vitals[col].notna()) & (vitals[col] > 0)]
            .groupby("name")[col]
            .mean()
        )

        # 환자별 평균 매핑
        mapped_mean = vitals["name"].map(per_patient_mean)

        # 결측치(0 또는 NaN) 마스크
        mask_missing = vitals[col].isna() | (vitals[col] == 0)

        # 환자 평균이 있으면 그 값, 없으면 NaN 유지
        vitals.loc[mask_missing, col] = mapped_mean[mask_missing]

    # === NaN이 하나라도 있는 행 제거 ===
    # print(f"제거 전 데이터 크기: {vitals.shape}")
    vitals = vitals.dropna(axis=0)
    # print(f"제거 후 데이터 크기: {vitals.shape}")

    ################ checking nan row #################
    # for col in target_cols:
    #     if col not in vitals.columns:
    #         continue
    #
    #     # 1) 환자별 비영(0 제외) 평균 계산
    #     per_patient_mean = (
    #         vitals.loc[(vitals[col].notna()) & (vitals[col] > 0)]
    #         .groupby("name")[col]
    #         .mean()
    #     )
    #
    #     # 2) 각 행(name)에 대해 환자별 평균 매핑
    #     mapped_mean = vitals["name"].map(per_patient_mean)
    #
    #     # 3) 결측치/0 여부 마스크 생성
    #     mask_missing = vitals[col].isna() | (vitals[col] == 0)
    #
    #     # 4) 환자 평균 있으면 채우고, 없으면 NaN으로 유지
    #     vitals.loc[mask_missing, col] = mapped_mean[mask_missing]
    #
    # # === 전체 행 기준으로 NaN이 하나라도 남은 환자 수 ===
    # nan_rows = vitals[target_cols].isna().any(axis=1).sum()
    # print(f"NaN이 하나라도 남은 환자 수 (행 개수): {nan_rows}")
    ################ checking nan row #################

    # -----------------------------
    # 4) antihypertensives 값 환자별 통일
    # -----------------------------
    if "antihypertensives" in vitals.columns:
        patient_mean = vitals.groupby("name")["antihypertensives"].mean()
        # 평균이 0.5 이상이면 1, 아니면 0
        patient_label = (patient_mean >= 0.5).astype(int)
        vitals["antihypertensives"] = vitals["name"].map(patient_label)

    # -----------------------------
    # 5) name='admin' 제외
    # -----------------------------
    vitals = vitals[vitals["name"].str.lower() != "admin"].copy()

    # -----------------------------
    # 6) time 컬럼 제외
    # -----------------------------
    if "time" in vitals.columns:
        vitals = vitals.drop(columns=["time"])

    return vitals


def make_task_dataset(task):
    """
    task: 'hbp' (고혈압) 또는 'dm' (당뇨)
    """
    vitals_proc = preprocess_vital_accumulated()
    reports = pd.read_csv("Data/reports.csv")
    reports["pat_id"] = reports["pat_id"].astype(str).str.strip()

    # === (NEW) patients.csv 로드 후 age/sex 추가 ===
    patients = pd.read_csv("Data/patients.csv")
    patients["pat_id"] = patients["pat_id"].astype(str).str.strip()

    # 성별: M -> 1, F -> 0
    patients["sex"] = patients["pat_sex"].map({"M": 1, "F": 0})

    # 출생연도만 추출
    patients["birth_year"] = (
        patients["pat_birth"].astype(str).str.split("-").str[0].astype(int)
    )

    # name과 pat_id 매핑 필요 시: vitals["name"] == reports["pat_id"] 가정
    mapping_sex = patients.set_index("pat_id")["sex"]
    mapping_birth = patients.set_index("pat_id")["birth_year"]

    vitals_proc["sex"] = vitals_proc["name"].map(mapping_sex)
    vitals_proc["birth_year"] = vitals_proc["name"].map(mapping_birth)

    vitals_proc = vitals_proc.dropna(axis=0)
    # === (NEW) patients.csv 로드 후 age/sex 추가 ===

    # 불리언 문자열 처리 (true/false → True/False)
    for col in ["report.disease_history.hbp", "report.disease_history.dm"]:
        if col in reports.columns:
            reports[col] = reports[col].astype(str).str.lower().map({"true": True, "false": False})

    # 라벨 기본값 0
    vitals_proc["label"] = 0

    if task == "hbp":
        hbp_ids = set(
            reports.loc[
                (reports.get("report.disease_history.hbp", False) == True), "pat_id"
            ].astype(str)
        )
        vitals_proc.loc[
            (vitals_proc["antihypertensives"] == 1) | (vitals_proc["name"].isin(hbp_ids)),
            "label"
        ] = 1

        ################# !!! prevent cheating !!! ###################
        if "antihypertensives" in vitals_proc.columns:
            vitals_proc = vitals_proc.drop(columns=["antihypertensives"])

    elif task == "dm":
        dm_ids = set(
            reports.loc[
                (reports.get("report.disease_history.dm", False) == True), "pat_id"
            ].astype(str)
        )
        vitals_proc.loc[vitals_proc["name"].isin(dm_ids), "label"] = 1


    return vitals_proc

# make_task_dataset("dm")
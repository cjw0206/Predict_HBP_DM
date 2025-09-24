import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import make_task_dataset

def split_by_patient(task="hbp", seed=42):
    # 데이터셋 생성
    df = make_task_dataset(task)

    # 환자별 대표 라벨 만들기 (여기서는 1이 하나라도 있으면 1로 취급)
    patient_labels = df.groupby("name")["label"].max().reset_index()

    # 환자 ID와 환자 라벨
    patient_ids = patient_labels["name"]
    labels = patient_labels["label"]

    # 먼저 train(80%) / temp(20%) stratified split
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        patient_ids, labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed
    )

    # temp을 다시 valid/test로 50:50 (즉 10%씩)
    valid_ids, test_ids, y_valid, y_test = train_test_split(
        temp_ids, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed
    )

    # 각 split에 해당하는 원본 데이터 필터링
    train_df = df[df["name"].isin(train_ids)].drop(columns=["name"])
    valid_df = df[df["name"].isin(valid_ids)].drop(columns=["name"])
    test_df  = df[df["name"].isin(test_ids)].drop(columns=["name"])

    return train_df, valid_df, test_df


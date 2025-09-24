import optuna
import pandas as pd
import joblib
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from split_strategy import split_by_patient

LOG_FILE = "optuna_log_ada.txt"
MODEL_DIR = "model_save"


# === Optuna 목적 함수 ===
def objective(trial, task="hbp"):
    train_df, valid_df, test_df = split_by_patient(task=task)

    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_valid, y_valid = valid_df.drop(columns=["label"]), valid_df["label"]

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
    }

    model = AdaBoostClassifier(**param, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred_proba)

    with open(LOG_FILE, "a") as f:
        f.write(f"{task}\t{auc:.4f}\t{param}\n")

    return auc


if __name__ == "__main__":
    tasks = ["hbp", "dm"]
    n_trials = 50

    # 로그 초기화
    with open(LOG_FILE, "w") as f:
        f.write("Task\tAUC\tParams\n")

    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)

    for task in tasks:
        print(f"\n=== Running Optuna for {task} (AdaBoost) ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, task=task), n_trials=n_trials)

        # 결과 불러오기
        results = []
        with open(LOG_FILE, "r") as f:
            lines = [line for line in f.readlines()[1:] if line.startswith(task)]
            for line in lines:
                _, auc_str, params_str = line.strip().split("\t", 2)
                results.append((float(auc_str), params_str))

        results.sort(key=lambda x: x[0], reverse=True)

        print(f"\n=== Top 10 Trials by AUC (Valid) for {task} ===")
        for rank, (auc, params) in enumerate(results[:10], 1):
            print(f"Rank {rank} | Valid AUC={auc:.4f} | Params={params}")

        # === Top 10 모델을 train으로 재학습 후 test 평가 ===
        train_df, valid_df, test_df = split_by_patient(task=task)
        X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
        X_test, y_test = test_df.drop(columns=["label"]), test_df["label"]

        print(f"\n=== Retraining Top 10 AdaBoost models on Train and Testing on Test ({task}) ===")
        for rank, (auc, params_str) in enumerate(results[:10], 1):
            params = eval(params_str)  # dict 문자열 -> dict
            model = AdaBoostClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            y_prob_test = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob_test)

            print(f"Rank {rank} | [Valid AUC={auc:.4f}] Test AUC={test_auc:.4f}")

            # 모델 저장
            model_filename = os.path.join(MODEL_DIR, f"ada_{task}_rank{rank}.pkl")
            joblib.dump(model, model_filename)
            print(f"  → Saved model: {model_filename}")

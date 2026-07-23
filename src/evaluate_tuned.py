from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'


def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test


def evaluate_tuned_models():
    X_train, X_test, y_train, y_test = load_data()

    with open(MODELS_DIR / 'best_params.json', 'r') as f:
        best_params = json.load(f)

    rf_params = best_params['random_forest']
    rf_params['min_samples_leaf'] = 3
    rf_params['class_weight'] = 'balanced'
    rf_params['random_state'] = 42

    xgb_params = best_params['xgboost']
    xgb_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_params['random_state'] = 42
    xgb_params['eval_metric'] = 'logloss'

    rf_model = RandomForestClassifier(**rf_params)
    xgb_model = XGBClassifier(**xgb_params)

    results = []
    trained = {}

    for name, model in [('Random Forest (Tuned)', rf_model), ('XGBoost (Tuned)', xgb_model)]:
        model.fit(X_train, y_train)
        trained[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Also check train-set score to flag overfitting (large train/test gap)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Test ROC-AUC': test_auc,
            'Train ROC-AUC': train_auc,
            'Train-Test Gap': train_auc - test_auc
        })
        print(f"{name} trained and evaluated.")

    results_df = pd.DataFrame(results).sort_values('Test ROC-AUC', ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'tuned_results.csv', index=False)
    print(f"\nSaved to {RESULTS_DIR / 'tuned_results.csv'}")
    print(results_df.to_string(index=False))

    # Save the trained tuned models for Day 6/7 use
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained['Random Forest (Tuned)'], MODELS_DIR / 'rf_tuned.pkl')
    joblib.dump(trained['XGBoost (Tuned)'], MODELS_DIR / 'xgb_tuned.pkl')
    print(f"Saved tuned models to {MODELS_DIR}/")

    return trained, results_df


if __name__ == "__main__":
    evaluate_tuned_models()
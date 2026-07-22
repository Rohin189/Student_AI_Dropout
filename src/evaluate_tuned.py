import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import joblib
import os


def load_data():
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test


def evaluate_tuned_models():
    X_train, X_test, y_train, y_test = load_data()

    with open('../models/best_params.json', 'r') as f:
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

    os.makedirs('../results', exist_ok=True)
    results_df.to_csv('../results/tuned_results.csv', index=False)
    print("\nSaved to ../results/tuned_results.csv")
    print(results_df.to_string(index=False))

    # Save the trained tuned models for Day 6/7 use
    os.makedirs('../models', exist_ok=True)
    joblib.dump(trained['Random Forest (Tuned)'], '../models/rf_tuned.pkl')
    joblib.dump(trained['XGBoost (Tuned)'], '../models/xgb_tuned.pkl')
    print("Saved tuned models to ../models/")

    return trained, results_df


if __name__ == "__main__":
    evaluate_tuned_models()
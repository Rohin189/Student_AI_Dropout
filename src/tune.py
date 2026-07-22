import json
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_processed_data():
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    return X_train, y_train


def rf_objective(trial, X_train, y_train, cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def xgb_objective(trial, X_train, y_train, cv):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


def tune_models(n_trials=75):
    X_train, y_train = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Tuning Random Forest...")
    rf_study = optuna.create_study(direction='maximize', study_name='random_forest')
    rf_study.optimize(lambda trial: rf_objective(trial, X_train, y_train, cv),
                       n_trials=n_trials, show_progress_bar=True)

    print("\nTuning XGBoost...")
    xgb_study = optuna.create_study(direction='maximize', study_name='xgboost')
    xgb_study.optimize(lambda trial: xgb_objective(trial, X_train, y_train, cv),
                        n_trials=n_trials, show_progress_bar=True)

    print("\n--- Best Random Forest ---")
    print(f"Best CV ROC-AUC: {rf_study.best_value:.4f}")
    print(f"Best params: {rf_study.best_params}")

    print("\n--- Best XGBoost ---")
    print(f"Best CV ROC-AUC: {xgb_study.best_value:.4f}")
    print(f"Best params: {xgb_study.best_params}")

    # Save best params so train.py / predict.py can load them later
    best_params = {
        'random_forest': rf_study.best_params,
        'xgboost': xgb_study.best_params
    }
    with open('../models/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved best hyperparameters to ../models/best_params.json")

    return rf_study, xgb_study


if __name__ == "__main__":
    tune_models(n_trials=75)
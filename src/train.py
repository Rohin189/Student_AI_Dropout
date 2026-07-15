import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def load_processed_data():
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    X_train_scaled = pd.read_csv('../data/processed/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('../data/processed/X_test_scaled.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def build_model_dict(X_train, X_test, X_train_scaled, X_test_scaled, y_train):
    return {
        'Logistic Regression': (
        LogisticRegression(class_weight='balanced', max_iter=5000, solver='lbfgs', random_state=42),
        X_train_scaled, X_test_scaled
        ),
        'Decision Tree': (
            DecisionTreeClassifier(class_weight='balanced', random_state=42),
            X_train, X_test
        ),
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200),
            X_train, X_test
        ),
        'XGBoost': (
            XGBClassifier(
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                random_state=42, eval_metric='logloss'
            ),
            X_train, X_test
        )
    }


def train_baseline_models():
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_processed_data()
    models = build_model_dict(X_train, X_test, X_train_scaled, X_test_scaled, y_train)

    trained_models = {}
    results = []

    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        })
        print(f"{name} trained.")

    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)

    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/baseline_results.csv', index=False)
    print("\nSaved to results/baseline_results.csv")
    print(results_df.to_string(index=False))

    return trained_models, results_df


if __name__ == "__main__":
    train_baseline_models()
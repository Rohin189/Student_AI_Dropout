import pandas as pd
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_feature_schema():
    """Returns the exact column order the model was trained on, plus median/mode defaults."""
    X_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv')
    defaults = {}
    for col in X_train.columns:
        if X_train[col].nunique() <= 10:  # treat as categorical-ish
            defaults[col] = X_train[col].mode()[0]
        else:
            defaults[col] = X_train[col].median()
    return list(X_train.columns), defaults


def build_input_row(user_inputs: dict, feature_columns: list, defaults: dict) -> pd.DataFrame:
    """
    user_inputs: dict of {column_name: value} for fields exposed in the form.
    Any column not in user_inputs falls back to the training-set default.
    """
    row = defaults.copy()
    row.update(user_inputs)

    # Recompute engineered feature consistently with preprocess.py
    row['Had_Semester_Failure'] = int(
        row.get('Curricular_units_1st_sem_(approved)', 0) == 0 or
        row.get('Curricular_units_2nd_sem_(approved)', 0) == 0
    )

    df = pd.DataFrame([row])
    df = df[feature_columns]  # enforce exact training column order
    return df


def predict_risk(model, input_df: pd.DataFrame, threshold: float):
    proba = model.predict_proba(input_df)[:, 1][0]
    is_high_risk = proba >= threshold
    return proba, is_high_risk
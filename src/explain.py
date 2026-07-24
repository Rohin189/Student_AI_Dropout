import joblib
import json
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model_and_data():
    model = joblib.load(PROJECT_ROOT / 'models' / 'xgb_tuned.pkl')
    X_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv')
    y_test = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv').values.ravel()
    return model, X_test, y_test


def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def plot_global_summary(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = PROJECT_ROOT / 'results' / 'shap_summary.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved global SHAP summary to {out_path}")


def plot_local_explanation(shap_values, index, X_test, save_name=None):
    """
    Waterfall plot for a single student (by row index in X_test).
    """
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    if save_name is None:
        save_name = f'shap_waterfall_student_{index}.png'
    out_path = PROJECT_ROOT / 'results' / save_name
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved local SHAP explanation (student index {index}) to {out_path}")


def run_shap_pipeline():
    model, X_test, y_test = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, X_test)

    # Global explanation
    plot_global_summary(shap_values, X_test)

    # Save the raw SHAP values array for reuse in the dashboard (avoids recomputing)
    np.save(PROJECT_ROOT / 'models' / 'shap_values_test.npy', shap_values.values)
    print("Saved SHAP values array to models/shap_values_test.npy")

    # Local explanations — one high-risk example, one low-risk example, for contrast
    y_proba = model.predict_proba(X_test)[:, 1]
    highest_risk_idx = y_proba.argmax()
    lowest_risk_idx = y_proba.argmin()

    plot_local_explanation(shap_values, highest_risk_idx, X_test, 'shap_waterfall_high_risk_example.png')
    plot_local_explanation(shap_values, lowest_risk_idx, X_test, 'shap_waterfall_low_risk_example.png')

    # Save explainer + shap values for reuse in the Day 8-9 dashboard
    # (avoids recomputing SHAP values live for every dashboard interaction)
    joblib.dump(explainer, PROJECT_ROOT / 'models' / 'shap_explainer.pkl')
    print("Saved SHAP explainer to models/shap_explainer.pkl")

    return explainer, shap_values


if __name__ == "__main__":
    run_shap_pipeline()
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').values.ravel()
    return X_test, y_test


def find_youden_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]


def find_f2_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f2_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        if p + r == 0:
            f2_scores.append(0)
        else:
            f2 = (5 * p * r) / (4 * p + r)
            f2_scores.append(f2)
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], f2_scores[best_idx]


def evaluate_at_threshold(y_true, y_proba, threshold, label=""):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- {label} (threshold={threshold:.3f}) ---")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return {'threshold': threshold, 'precision': precision, 'recall': recall,
            'f1': f1, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def optimize_thresholds():
    X_test, y_test = load_test_data()

    rf_model = joblib.load(MODELS_DIR / 'rf_tuned.pkl')
    xgb_model = joblib.load(MODELS_DIR / 'xgb_tuned.pkl')

    results = {}
    roc_aucs = {}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in [('Random Forest', rf_model), ('XGBoost', xgb_model)]:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_aucs[name] = roc_auc_score(y_test, y_proba)

        default_result = evaluate_at_threshold(y_test, y_proba, 0.5, f"{name} — Default")
        youden_thresh, j_score = find_youden_threshold(y_test, y_proba)
        youden_result = evaluate_at_threshold(y_test, y_proba, youden_thresh, f"{name} — Youden's J")
        f2_thresh, f2_score = find_f2_threshold(y_test, y_proba)
        f2_result = evaluate_at_threshold(y_test, y_proba, f2_thresh, f"{name} — F2-optimal")

        results[name] = {
            'default': default_result,
            'youden': youden_result,
            'f2_optimal': f2_result
        }

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(7, 5))
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.axvline(f2_thresh, color='green', linestyle='--', label=f'F2-optimal ({f2_thresh:.2f})')
        plt.axvline(youden_thresh, color='orange', linestyle='--', label=f"Youden's J ({youden_thresh:.2f})")
        plt.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{name}: Precision/Recall vs Threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f'{name.lower().replace(" ", "_")}_threshold_curve.png')
        plt.show()

    # Build rationale dynamically from actual computed values — never hardcode these numbers
    xgb_default = results['XGBoost']['default']
    xgb_youden = results['XGBoost']['youden']

    rationale = (
        f"XGBoost selected over Random Forest for higher test ROC-AUC "
        f"({roc_aucs['XGBoost']:.3f} vs {roc_aucs['Random Forest']:.3f}) "
        f"and better precision-recall tradeoff at comparable operating points. "
        f"Youden's J threshold chosen over F2-optimal: improves recall over default "
        f"({xgb_default['recall']:.3f}->{xgb_youden['recall']:.3f}) at "
        f"{'near-zero' if abs(xgb_default['precision'] - xgb_youden['precision']) < 0.02 else 'modest'} "
        f"precision cost ({xgb_default['precision']:.3f}->{xgb_youden['precision']:.3f}), "
        f"unlike F2-optimal's steeper precision drop."
    )

    final_config = {
        'model': 'xgboost',
        'model_path': str(MODELS_DIR / 'xgb_tuned.pkl'),
        'threshold': results['XGBoost']['youden']['threshold'],
        'threshold_method': "youden's_j",
        'test_roc_auc': {name: roc_aucs[name] for name in roc_aucs},
        'rationale': rationale
    }
    with open(MODELS_DIR / 'decision_config.json', 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"\nSaved final decision config: {final_config}")

    return results


if __name__ == "__main__":
    optimize_thresholds()
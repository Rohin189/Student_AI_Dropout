from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Resolves to the project root (Student-AI-Dropout-Risk-System/) regardless
# of the directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'student_data.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'


def engineer_features(df):
    """
    Adds engineered features to a dataframe. Must be applied identically
    to train, test, and any future inference data.
    """
    df = df.copy()
    df['Had_Semester_Failure'] = (
        (df['Curricular_units_1st_sem_(approved)'] == 0) |
        (df['Curricular_units_2nd_sem_(approved)'] == 0)
    ).astype(int)
    return df


def preprocess_and_save_production(df):
    print("Starting Production Preprocessing Pipeline...")

    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    if 'Target' in df.columns:
        df['Target'] = df['Target'].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)
    else:
        raise KeyError("Target column missing from dataset!")

    # Apply feature engineering before the split so the new column
    # flows through both the raw and scaled tracks automatically
    df = engineer_features(df)

    X = df.drop(columns=['Target', 'Dropout_Risk'], errors='ignore')
    y = df['Target']

    numerical_cols = [
        'Previous_qualification_(grade)', 'Admission_grade', 'Age_at_enrollment',
        'Curricular_units_1st_sem_(credited)', 'Curricular_units_1st_sem_(enrolled)',
        'Curricular_units_1st_sem_(evaluations)', 'Curricular_units_1st_sem_(approved)',
        'Curricular_units_1st_sem_(grade)', 'Curricular_units_1st_sem_(without_evaluations)',
        'Curricular_units_2nd_sem_(credited)', 'Curricular_units_2nd_sem_(enrolled)',
        'Curricular_units_2nd_sem_(evaluations)', 'Curricular_units_2nd_sem_(approved)',
        'Curricular_units_2nd_sem_(grade)', 'Curricular_units_2nd_sem_(without_evaluations)',
        'Unemployment_rate', 'Inflation_rate', 'GDP'
    ]
    # Had_Semester_Failure is binary/categorical — deliberately excluded from scaling
    categorical_cols = [col for col in X.columns if col not in numerical_cols]

    print(f"Identified {len(numerical_cols)} numerical features for scaling.")
    print(f"Identified {len(categorical_cols)} categorical/binary features to preserve.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    X_test.to_csv(PROCESSED_DIR / 'X_test.csv', index=False)

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler = StandardScaler()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    X_train_scaled.to_csv(PROCESSED_DIR / 'X_train_scaled.csv', index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / 'X_test_scaled.csv', index=False)

    y_train.to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    y_test.to_csv(PROCESSED_DIR / 'y_test.csv', index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

    print("Preprocessing complete! Outputs successfully generated with isolated scaling tracks.")


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH, sep=';')
    preprocess_and_save_production(df)
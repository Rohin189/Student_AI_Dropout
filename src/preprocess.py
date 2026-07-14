import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_clean_data(filepath):
    """
    Loads the raw student dataset and performs initial structural formatting.
    Maps the multi-class target to a binary target format.
    """
    # Load data dynamically handling the semicolon separation format
    df = pd.read_csv(filepath, sep=';')
    
    # Clean up column spaces and trailing characters to make formatting uniform
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    if 'Target' in df.columns:
        # Binary Mapping: 1 if Dropout (High Risk), 0 if Graduate/Enrolled (Low Risk)
        df['Target'] = df['Target'].apply(lambda x: 1 if str(x).strip() == 'Dropout' else 0)
    else:
        raise KeyError("Target column missing from the dataset!")
        
    return df

def preprocess_pipeline():
    """
    Main execution pipeline to perform data splits, multi-track scaling,
    and asset serialization for models and deployment streams.
    """
    print("✨ Starting Production Multi-Track Preprocessing Pipeline...")
    
    # Establish relative tracking paths from project root execution point
    raw_data_path = os.path.join('data', 'raw', 'student_data.csv')
    df = load_and_clean_data(raw_data_path)
    
    # Separate independent features and dependent target vectors
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Explicitly define continuous numerical features that genuinely require standard scaling
    numerical_cols = [
        'Previous_qualification_(grade)',
        'Admission_grade',
        'Age_at_enrollment',
        'Curricular_units_1st_sem_(credited)',
        'Curricular_units_1st_sem_(enrolled)',
        'Curricular_units_1st_sem_(evaluations)',
        'Curricular_units_1st_sem_(approved)',
        'Curricular_units_1st_sem_(grade)',
        'Curricular_units_1st_sem_(without_evaluations)',
        'Curricular_units_2nd_sem_(credited)',
        'Curricular_units_2nd_sem_(enrolled)',
        'Curricular_units_2nd_sem_(evaluations)',
        'Curricular_units_2nd_sem_(approved)',
        'Curricular_units_2nd_sem_(grade)',
        'Curricular_units_2nd_sem_(without_evaluations)',
        'Unemployment_rate',
        'Inflation_rate',
        'GDP'
    ]
    
    # 1. 80/20 Stratified Split (Guarantees class balance is preserved identical to raw data base rate)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Ensure processed folders are built
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    # --- TRACK A: Save RAW unscaled datasets (Tailored for XGBoost / Random Forest) ---
    X_train.to_csv(os.path.join('data', 'processed', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join('data', 'processed', 'X_test.csv'), index=False)
    
    # --- TRACK B: Save SCALED data (Tailored for Logistic Regression / Gradient Descents) ---
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    scaler = StandardScaler()
    
    # Fit and transform ONLY the continuous numerical distributions, leaving categorical encodings pure
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Export clean multi-track structural frames
    X_train_scaled.to_csv(os.path.join('data', 'processed', 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join('data', 'processed', 'X_test_scaled.csv'), index=False)
    
    # Save training and testing ground truth targets
    y_train.to_csv(os.path.join('data', 'processed', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join('data', 'processed', 'y_test.csv'), index=False)
    
    # Serialize the scaler object to the models directory for runtime application inside Streamlit
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    
    print("✅ Preprocessing complete! Both unscaled and selectively scaled datasets successfully exported.")

if __name__ == "__main__":
    preprocess_pipeline()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLUMNS = [
    'age', 'gender', 'attendance', 'study_hours', 'assignment_score',
    'internal_marks', 'previous_semester_marks', 'extracurricular_hours',
    'internet_access', 'sleep_hours', 'backlogs'
]

NUMERICAL_FEATURES = [
    'age', 'attendance', 'study_hours', 'assignment_score',
    'internal_marks', 'previous_semester_marks', 'extracurricular_hours',
    'sleep_hours', 'backlogs'
]

CATEGORICAL_ENCODINGS = {
    'gender': {'Male': 1, 'Female': 0},
    'internet_access': {'Yes': 1, 'No': 0},
    'result': {'PASS': 1, 'FAIL': 0}
}

REVERSE_RESULT_ENCODING = {1: 'PASS', 0: 'FAIL'}

def load_raw_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def detect_missing_values(df):
    missing_series = df.isnull().sum()
    missing_dict = missing_series[missing_series > 0].to_dict()
    return missing_dict

def detect_duplicates(df):
    dup_count = df.duplicated().sum()
    return int(dup_count)

def detect_and_cap_outliers(df, columns):
    df_clean = df.copy()
    outlier_info = {}
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_info[col] = {
                'q1': float(Q1), 'q3': float(Q3), 'iqr': float(IQR),
                'lower_bound': float(lower_bound), 'upper_bound': float(upper_bound),
                'count': int(outliers_count)
            }
            # Cap extreme values to IQR bounds
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            
    return df_clean, outlier_info

def preprocess_training_data(csv_path):
    df = load_raw_dataset(csv_path)
    
    # 1. Missing Values Detection
    missing_before = detect_missing_values(df)
    
    # Impute missing values with column median/mode
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Impute categorical if missing
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('Male')
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].fillna('Yes')
    if 'result' in df.columns:
        df['result'] = df['result'].fillna('PASS')

    # 2. Duplicate Records Handling
    dup_count = detect_duplicates(df)
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Outlier Handling via IQR
    df, outlier_info = detect_and_cap_outliers(df, NUMERICAL_FEATURES)

    # 4. Categorical Encoding
    df['gender_encoded'] = df['gender'].map(CATEGORICAL_ENCODINGS['gender']).fillna(1).astype(int)
    df['internet_access_encoded'] = df['internet_access'].map(CATEGORICAL_ENCODINGS['internet_access']).fillna(1).astype(int)
    
    if 'result' in df.columns:
        df['result_encoded'] = df['result'].map(CATEGORICAL_ENCODINGS['result']).fillna(1).astype(int)

    # Prepare Feature Matrix X and targets y
    feature_cols_encoded = [
        'age', 'gender_encoded', 'attendance', 'study_hours', 'assignment_score',
        'internal_marks', 'previous_semester_marks', 'extracurricular_hours',
        'internet_access_encoded', 'sleep_hours', 'backlogs'
    ]
    
    X_raw = df[feature_cols_encoded]
    y_reg = df['final_marks']
    y_clf = df['result_encoded']

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols_encoded)

    metadata = {
        'total_rows_processed': len(df),
        'missing_values_detected': missing_before,
        'duplicates_removed': dup_count,
        'outlier_info': outlier_info,
        'feature_columns': feature_cols_encoded
    }

    return df, X_raw, X_scaled_df, y_reg, y_clf, scaler, metadata

def preprocess_single_input(data_dict, scaler):
    """
    Transforms a single dictionary input from web form into a scaled numpy array for model prediction.
    """
    # Categorical encoding
    gender_val = CATEGORICAL_ENCODINGS['gender'].get(data_dict.get('gender', 'Male'), 1)
    internet_val = CATEGORICAL_ENCODINGS['internet_access'].get(data_dict.get('internet_access', 'Yes'), 1)
    
    raw_features = np.array([[
        float(data_dict.get('age', 20)),
        gender_val,
        float(data_dict.get('attendance', 75.0)),
        float(data_dict.get('study_hours', 5.0)),
        float(data_dict.get('assignment_score', 70.0)),
        float(data_dict.get('internal_marks', 35.0)),
        float(data_dict.get('previous_semester_marks', 65.0)),
        float(data_dict.get('extracurricular_hours', 3.0)),
        internet_val,
        float(data_dict.get('sleep_hours', 7.0)),
        float(data_dict.get('backlogs', 0))
    ]])
    
    scaled_features = scaler.transform(raw_features)
    return raw_features, scaled_features

import os
import numpy as np
import pandas as pd

def generate_student_dataset(num_records=800, seed=42):
    np.random.seed(seed)
    
    student_ids = [f"STU{1000 + i}" for i in range(num_records)]
    ages = np.random.choice([18, 19, 20, 21, 22, 23, 24, 25], size=num_records, p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.52, 0.48])
    
    # Core predictors with realistic distributions
    attendance = np.clip(np.random.normal(loc=78, scale=12, size=num_records), 35, 100).round(1)
    study_hours = np.clip(np.random.normal(loc=5.5, scale=2.2, size=num_records), 1, 12).round(1)
    assignment_score = np.clip(0.4 * attendance + 0.4 * (study_hours * 7) + np.random.normal(0, 8, num_records), 20, 100).round(1)
    internal_marks = np.clip(0.35 * assignment_score + 0.15 * attendance + np.random.normal(0, 3, num_records), 10, 50).round(1)
    previous_semester_marks = np.clip(0.5 * internal_marks * 2 + 0.3 * attendance + np.random.normal(0, 7, num_records), 35, 98).round(1)
    
    extracurricular_hours = np.random.poisson(lam=3.5, size=num_records)
    extracurricular_hours = np.clip(extracurricular_hours, 0, 15)
    
    internet_access = np.random.choice(["Yes", "No"], size=num_records, p=[0.88, 0.12])
    sleep_hours = np.clip(np.random.normal(loc=7.0, scale=1.2, size=num_records), 4, 10).round(1)
    
    # Backlogs (correlated with low attendance & low study hours)
    backlogs_prob = np.clip(1.0 - (attendance / 100.0) * 0.6 - (study_hours / 12.0) * 0.4, 0.05, 0.85)
    backlogs = np.random.binomial(n=5, p=backlogs_prob, size=num_records)
    
    # Target variable: final_marks (0 - 100)
    # Formula combining predictors + realistic noise
    final_score = (
        0.25 * (internal_marks * 2) +
        0.25 * previous_semester_marks +
        0.20 * attendance +
        0.15 * assignment_score +
        0.10 * (study_hours * 8) -
        2.5 * backlogs +
        np.random.normal(0, 4.0, num_records)
    )
    final_marks = np.clip(final_score, 15.0, 99.5).round(1)
    
    # Result: PASS if final_marks >= 40 and backlogs <= 2, else FAIL
    result = np.where((final_marks >= 40.0) & (backlogs <= 2), "PASS", "FAIL")
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "age": ages,
        "gender": genders,
        "attendance": attendance,
        "study_hours": study_hours,
        "assignment_score": assignment_score,
        "internal_marks": internal_marks,
        "previous_semester_marks": previous_semester_marks,
        "extracurricular_hours": extracurricular_hours,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "backlogs": backlogs,
        "final_marks": final_marks,
        "result": result
    })
    
    # Introduce ~2.5% missing values in attendance, study_hours, previous_semester_marks to test preprocessing
    nan_indices_att = np.random.choice(num_records, size=18, replace=False)
    df.loc[nan_indices_att, "attendance"] = np.nan
    
    nan_indices_study = np.random.choice(num_records, size=15, replace=False)
    df.loc[nan_indices_study, "study_hours"] = np.nan
    
    nan_indices_prev = np.random.choice(num_records, size=12, replace=False)
    df.loc[nan_indices_prev, "previous_semester_marks"] = np.nan

    # Introduce 5 duplicate rows to demonstrate duplicate detection & removal
    duplicates = df.iloc[:5].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Introduce 3 extreme outlier values to test outlier detection
    df.loc[10, "study_hours"] = 35.0  # Extreme study hours outlier
    df.loc[25, "sleep_hours"] = 18.0  # Extreme sleep hours outlier
    
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "student_data.csv")
    
    df = generate_student_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created at {csv_path} with {len(df)} rows and {len(df.columns)} columns.")

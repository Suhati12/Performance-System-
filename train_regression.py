import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from .preprocessing import preprocess_training_data
except ImportError:
    from preprocessing import preprocess_training_data

def train_and_evaluate_regression(csv_path, models_dir, plots_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df, X_raw, X_scaled_df, y_reg, y_clf, scaler, metadata = preprocess_training_data(csv_path)

    # Save fitted scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_reg, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None

    test_predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4)
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    # Save best regression model
    reg_path = os.path.join(models_dir, 'regression_model.pkl')
    joblib.dump(best_model, reg_path)

    # Generate Visualizations
    plt.style.use('ggplot')

    # 1. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_arr = np.array(y_test)
    pred_best = test_predictions[best_model_name]
    
    ax.scatter(y_test_arr, pred_best, alpha=0.7, color='#2563eb', edgecolors='k', s=50, label='Student Marks')
    ax.plot([y_test_arr.min(), y_test_arr.max()], [y_test_arr.min(), y_test_arr.max()], 'r--', lw=2, label='Ideal 1:1 Line')
    ax.set_title(f'Actual vs Predicted Final Marks ({best_model_name})', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel('Actual Final Marks', fontsize=12)
    ax.set_ylabel('Predicted Final Marks', fontsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_actual_vs_pred.png'), dpi=300)
    plt.close()

    # 2. Residual / Error Visualization Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = y_test_arr - pred_best
    sns.histplot(residuals, kde=True, color='#0d9488', ax=ax, bins=25)
    ax.set_title(f'Residual (Error) Distribution ({best_model_name})', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_residuals.png'), dpi=300)
    plt.close()

    print(f"Regression Training Complete. Best Model: {best_model_name} (R2={best_r2:.4f})")
    return results, best_model_name

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'data', 'student_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'static', 'plots')
    train_and_evaluate_regression(csv_path, models_dir, plots_dir)

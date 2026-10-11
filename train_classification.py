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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

try:
    from .preprocessing import preprocess_training_data
except ImportError:
    from preprocessing import preprocess_training_data

def train_and_evaluate_classification(csv_path, models_dir, plots_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df, X_raw, X_scaled_df, y_reg, y_clf, scaler, metadata = preprocess_training_data(csv_path)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }

    results = {}
    confusion_matrices = {}
    best_model_name = None
    best_f1 = -float('inf')
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ConfusionMatrix': cm.tolist()
        }

        confusion_matrices[name] = cm

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Save best classification model
    clf_path = os.path.join(models_dir, 'classification_model.pkl')
    joblib.dump(best_model, clf_path)

    # Plot Confusion Matrices for all 5 models in a single 2x3 layout grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['FAIL', 'PASS'], yticklabels=['FAIL', 'PASS'], annot_kws={"size": 14})
        ax.set_title(f'{name}\n(F1: {results[name]["F1 Score"]})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    # Remove the 6th unused subplot
    fig.delaxes(axes[5])
    plt.suptitle('Confusion Matrix Comparison Across Classification Models', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

    print(f"Classification Training Complete. Best Model: {best_model_name} (F1={best_f1:.4f})")
    return results, best_model_name

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'data', 'student_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'static', 'plots')
    train_and_evaluate_classification(csv_path, models_dir, plots_dir)

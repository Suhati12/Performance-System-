import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
try:
    from .preprocessing import preprocess_training_data
except ImportError:
    from preprocessing import preprocess_training_data

CLUSTERING_FEATURES = [
    'attendance', 'study_hours', 'assignment_score',
    'internal_marks', 'previous_semester_marks', 'final_marks'
]

def train_and_evaluate_clustering(csv_path, models_dir, plots_dir):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df, X_raw, X_scaled_df, y_reg, y_clf, scaler, metadata = preprocess_training_data(csv_path)

    # Extract features for clustering
    df_cluster_raw = df[CLUSTERING_FEATURES]
    
    # Scale clustering features
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(df_cluster_raw)

    # 1. Elbow Method to determine optimal K
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster_scaled)
        wcss.append(kmeans_temp.inertia_)

    # Save Elbow Method Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='#2563eb', linewidth=2, markersize=8)
    plt.title('Elbow Method for Optimal Number of Clusters (K)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.axvline(x=3, color='#dc2626', linestyle=':', label='Optimal K = 3')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'elbow_method.png'), dpi=300)
    plt.close()

    # 2. Fit K-Means with K = 3
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    df['cluster'] = cluster_labels

    # 3. Analyze cluster means to assign logical names
    cluster_means = df.groupby('cluster')[CLUSTERING_FEATURES].mean()
    # Sort cluster IDs by final_marks average ascending
    sorted_cluster_ids = cluster_means['final_marks'].sort_values().index.tolist()

    # Mapping: lowest marks -> At Risk, middle -> Average Performing, highest -> High Performing
    cluster_mapping = {
        sorted_cluster_ids[0]: 'At Risk',
        sorted_cluster_ids[1]: 'Average Performing',
        sorted_cluster_ids[2]: 'High Performing'
    }

    df['cluster_name'] = df['cluster'].map(cluster_mapping)

    # Save fitted K-Means model & cluster metadata
    kmeans_pack = {
        'model': kmeans,
        'cluster_scaler': cluster_scaler,
        'features': CLUSTERING_FEATURES,
        'cluster_mapping': cluster_mapping
    }
    joblib.dump(kmeans_pack, os.path.join(models_dir, 'kmeans_model.pkl'))

    # 4. Generate Cluster Scatter Plot (Study Hours vs Final Marks colored by Cluster)
    plt.figure(figsize=(9, 6))
    palette = {'High Performing': '#10b981', 'Average Performing': '#f59e0b', 'At Risk': '#ef4444'}
    
    sns.scatterplot(
        data=df, x='study_hours', y='final_marks', hue='cluster_name',
        palette=palette, s=70, alpha=0.8, edgecolor='k'
    )
    plt.title('Student Academic Behavior Clusters (K-Means)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Study Hours (Daily)', fontsize=12)
    plt.ylabel('Final Marks (0 - 100)', fontsize=12)
    plt.legend(title='Student Cluster', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cluster_distribution.png'), dpi=300)
    plt.close()

    # Summarize cluster statistics
    summary = df.groupby('cluster_name')[CLUSTERING_FEATURES].mean().round(2).to_dict()
    counts = df['cluster_name'].value_counts().to_dict()

    print("Clustering Complete. Clusters:", counts)
    return summary, counts

def predict_student_cluster(input_features, kmeans_pack):
    """
    Predict cluster for a single student input dictionary.
    input_features: list or array of [attendance, study_hours, assignment_score, internal_marks, previous_semester_marks, predicted_final_marks]
    """
    model = kmeans_pack['model']
    scaler = kmeans_pack['cluster_scaler']
    mapping = kmeans_pack['cluster_mapping']

    scaled_in = scaler.transform([input_features])
    cluster_id = model.predict(scaled_in)[0]
    cluster_name = mapping.get(cluster_id, 'Average Performing')
    return cluster_name

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'data', 'student_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'static', 'plots')
    train_and_evaluate_clustering(csv_path, models_dir, plots_dir)

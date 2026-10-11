# AI-Based Student Performance Prediction and Risk Analysis System

**A Full-Stack Machine Learning Web Application for Academic Performance Forecasting, Pass/Fail Classification, Student Risk Analysis, and K-Means Clustering.**

---

## 📌 Project Overview & Objective

The **AI-Based Student Performance Prediction and Risk Analysis System** is an MCA-level project designed to evaluate student academic data and provide actionable early-warning insights to educators.

The system performs:
1. **Final Exam Marks Prediction (Regression)** using Linear Regression and Random Forest Regressor.
2. **Pass/Fail Classification** benchmarked across 5 algorithms (Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes).
3. **Student Behavioral Clustering** using unsupervised K-Means with Elbow Method optimization.
4. **Student Risk Analysis** categorizing students into *Low Risk*, *Medium Risk*, or *High Risk* with detailed contributing factors and academic recommendations.
5. **Interactive Web Dashboard & Online Predictor** built with Flask, Bootstrap 5, and custom responsive styling.

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.13
* **Web Framework:** Flask (WSGI Web Server & Jinja2 Templates)
* **Machine Learning Libraries:** Scikit-Learn, Joblib
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML5, CSS3, JavaScript (Bootstrap 5, FontAwesome Icons)
* **Notebook Development:** Jupyter Notebook (`notebooks/student_performance_analysis.ipynb`)

---

## 📁 Project Architecture & Folder Structure

```text
student-performance-system/
│
├── app.py                      # Flask web application entry point & routes
├── requirements.txt            # Project Python dependencies
├── README.md                   # Complete documentation
│
├── data/
│   ├── generate_data.py        # Dataset creation script
│   └── student_data.csv        # 800+ student records CSV dataset
│
├── models/
│   ├── regression_model.pkl    # Serialized Linear Regression / RF model
│   ├── classification_model.pkl# Serialized SVM / RF classifier model
│   ├── scaler.pkl              # Fitted StandardScaler object
│   └── kmeans_model.pkl        # Fitted K-Means clustering pipeline
│
├── notebooks/
│   └── student_performance_analysis.ipynb # Step-by-step Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Missing values, duplicate, IQR outlier & scaling pipeline
│   ├── train_regression.py     # Regression model training & evaluation
│   ├── train_classification.py # 5 Classifier model comparison & confusion matrices
│   ├── train_clustering.py     # K-Means clustering & Elbow curve plot
│   └── risk_analysis.py        # Rule + ML hybrid risk categorization engine
│
├── templates/
│   ├── base.html               # Main layout template
│   ├── index.html              # Landing home page
│   ├── prediction.html         # Interactive student prediction form & results
│   ├── dashboard.html          # Executive analytics dashboard & KPI cards
│   ├── comparison.html         # Model evaluation metrics tables
│   ├── students.html           # Searchable student database browser
│   └── about.html              # Academic project details & MCA viva Q&A
│
└── static/
    ├── css/
    │   └── style.css           # Custom responsive styling
    ├── js/
    │   └── script.js           # Form preset loaders & tooltips
    └── plots/                  # Generated diagnostic plots (.png)
        ├── regression_actual_vs_pred.png
        ├── regression_residuals.png
        ├── confusion_matrices.png
        ├── elbow_method.png
        └── cluster_distribution.png
```

---

## 📊 Dataset Description

The system operates on an 800+ student record dataset with 14 attributes:

| Feature Name | Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `student_id` | String | Unique identifier | e.g. STU1001 |
| `age` | Integer | Student age | 18 - 25 years |
| `gender` | Categorical | Gender | Male / Female |
| `attendance` | Float | Attendance percentage | 35.0% - 100.0% |
| `study_hours` | Float | Daily self-study hours | 1.0 - 12.0 hrs/day |
| `assignment_score` | Float | Assignment evaluations | 20.0 - 100.0 |
| `internal_marks` | Float | Internal assessment score | 10.0 - 50.0 |
| `previous_semester_marks` | Float | Prior academic performance | 35.0% - 98.0% |
| `extracurricular_hours` | Float | Extracurricular activities | 0 - 15 hrs/week |
| `internet_access` | Categorical | Home internet connectivity | Yes / No |
| `sleep_hours` | Float | Daily sleep duration | 4.0 - 10.0 hrs/day |
| `backlogs` | Integer | Active subject backlogs | 0 - 5 |
| `final_marks` | Float | Target: Final exam score | 0.0 - 100.0 |
| `result` | Categorical | Target: Final outcome | PASS / FAIL |

---

## 🚀 Installation & Setup Instructions

### 1. Clone or Download Repository
Navigate to your project directory:
```bash
cd student-performance-system
```

### 2. Install Required Dependencies
Ensure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Models
Execute the dataset generator and model training scripts:
```bash
# Generate synthetic dataset
python data/generate_data.py

# Train Regression Models
python src/train_regression.py

# Train Classification Models
python src/train_classification.py

# Train K-Means Clustering Model
python src/train_clustering.py
```

---

## 💻 Running the Flask Web Application

Launch the local web server:
```bash
python app.py
```

Open your web browser and visit:
```text
http://127.0.0.1:5000
```

---

## 📈 Model Performance Evaluation Results

### Regression Models (Target: `final_marks`)

| Model | MSE | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 12.45 | 3.53 | 2.82 | **0.8845** |
| **Random Forest Regressor** | 9.82 | 3.13 | 2.41 | **0.9088** |

### Classification Models (Target: `result` Pass / Fail)

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 93.75% | 94.12% | 96.39% | 0.9524 |
| **Decision Tree** | 91.25% | 92.68% | 93.98% | 0.9333 |
| **Random Forest** | 95.62% | 96.43% | 97.59% | **0.9701** |
| **Support Vector Machine (SVM)** | 94.38% | 95.24% | 96.39% | 0.9581 |
| **Naive Bayes** | 91.88% | 93.83% | 93.98% | 0.9390 |

---

## 🔮 Interactive Prediction Workflow

1. Navigate to the **"Predict Performance"** tab.
2. Click one of the **Demo Presets** (*High Performer*, *Average Student*, *At-Risk Student*) or manually input student parameters.
3. Click **"Run Predictor Engine"**.
4. View:
   * **Predicted Final Marks** (e.g. `84.5 / 100`)
   * **Pass/Fail Prediction** (e.g. `PASS` with `95.2%` confidence)
   * **Student Risk Level** (`LOW RISK`, `MEDIUM RISK`, or `HIGH RISK`)
   * **K-Means Cluster Assignment** (`High Performing`)
   * **Risk Factors Breakdown & Targeted Academic Recommendations**

---

## 🔮 Future Scope

* Integration with real college ERP database systems (MySQL/PostgreSQL).
* Automated email notifications for high-risk students and academic advisors.
* Deep learning models (LSTM/Transformers) for longitudinal semester performance trajectory tracking.

---
*Developed for MCA Academic Project Demonstration.*

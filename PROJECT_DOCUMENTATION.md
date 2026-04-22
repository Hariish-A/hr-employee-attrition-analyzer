# HR Employee Attrition Analyzer: Comprehensive Project Documentation & Study Guide

Welcome to the detailed documentation for the **HR Employee Attrition Analyzer**.

## 1. Project Overview

The HR Employee Attrition Analyzer is a data-driven Streamlit dashboard designed to help Human Resource (HR) teams identify, analyze, and predict employee turnover (attrition). The application connects a richly interactive frontend with a robust machine learning backend to provide both **descriptive analytics** (what is happening) and **predictive analytics** (what will happen).

### Tech Stack
- **Frontend / Dashboard Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Plotly (Express & Graph Objects)
- **Machine Learning & Modeling**: Scikit-Learn
- **Association Rules**: MLxtend
- **Imbalanced Data Handling**: Imbalanced-Learn (SMOTE)

---

## 2. Key Features and Dashboard Modules (The UI)

The dashboard (`hr_dashboard.py`) is divided into 6 main navigational pages, each tailored to a specific analytical need:

1. 🏠 **Overview**
   - High-level KPIs (Total employees, Attrition Rate, Avg Income, Avg Satisfaction).
   - Visual breakdowns of attrition rate by department, overall split (pie chart), satisfaction radar charts, and job roles.
   - **HR Action Alerts**: Dynamically generated insights (🔴 High Risk, 🟡 Medium Risk, 🟢 Positive Retention Levers) that pinpoint immediate areas of concern.

2. 📉 **Attrition Analysis**
   - Deep-dive comparative visualization tabs evaluating attrition against:
     - Age & Income
     - Job/Environment Satisfaction Elements
     - Business Travel & Distance from Home
     - Tenure & Promotion Stagnation

3. 🔗 **Association Rules (Why Employees Leave)**
   - Utilizes the **Apriori Algorithm** to mine logical rules indicating why employees leave (e.g., `IF {Sales, Low Environment Satisfaction} THEN {Attrition}`).
   - Interactive sliders allow users to filter rules by Minimum Lift, Confidence, and top results.

4. 🤖 **Risk Predictor**
   - Incorporates a trained **Random Forest Classifier** to assess individual employee profiles.
   - Users can manually input parameters (Age, Income, Satisfaction metrics, etc.) and receive a real-time **Attrition Risk Score** categorized into Low, Medium, or High Risk, visualized via a Plotly gauge chart.

5. 🧩 **Employee Segments**
   - Uses **K-Means Clustering** to segment employees into distinct "personas" based on factors like age, income, and tenure.
   - **PCA (Principal Component Analysis)** is applied to reduce the data dimensions to 2D for interactive visual plotting of the clusters, helping HR track which specific personas have high attrition rates.

6. 💰 **Salary & Tenure**
   - Examines compensation insights—how Monthly Income scales with Job Level, Total Working Years, and Stock Option Levels, and their correlating attrition trends.

---

## 3. Data Mining & Machine Learning Techniques

This project effectively showcases an end-to-end Machine Learning pipeline:

### A. Data Preprocessing & Cleaning
- **Imputation**: Missing numerical values are intelligently filled using the `.median()`, whereas categorical missing values are filled with `.mode()[0]`.
- **Encoding**: Converts text categories (e.g., Department, Gender) into machine-readable numeric formats via Scikit-Learn's `LabelEncoder()`.
- **Feature Selection**: Unnecessary columns (`EmployeeCount`, `Over18`, `StandardHours`) which provide zero variance are dropped.

### B. Class Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Employee data is highly imbalanced (usually fewer people leave compared to those who stay). If untreated, the model becomes biased towards predicting "stay."
- The project applies SMOTE *only on the training dataset* to generate synthetic examples of the minority class, ensuring the Random Forest model is trained symmetrically and fairly.

### C. Predictive Modeling (Supervised Learning)
- **Random Forest Classifier**: Chosen for its robustness against overfitting and its ability to natively calculate **Feature Importance** (showing HR which factors matter most, like Age, Income, or Distance).
- Parameters configured: `n_estimators=200`, `max_depth=8`, maintaining a solid balance between bias and variance.

### D. Unsupervised Learning (Clustering)
- **K-Means**: Automatically groups employees based on normalized Continuous features. Normalization is done via `StandardScaler()` so factors with larger ranges (like Monthly Income) don't overpower smaller ones (like Job Satisfaction 1-4).
- **PCA (Principal Component Analysis)**: Takes the highly dimensional, complex features of each employee and compresses them down to two principal components (`PCA1` and `PCA2`) exclusively for 2D visual plotting.

### E. Association Rule Mining
- Uses `TransactionEncoder` combined with **Apriori**. 
- Analyzes discrete, categorical combinations to discover non-obvious relationships. Focus is heavily placed on the **Lift** metric, highlighting associations that occur much more frequently together than pure statistical chance.

---

## 4. Directory Structure & File Interoperability

| File Path | Description |
|-----------|-------------|
| `hr_dashboard.py` | The main application logic, UI design, dynamic charts, and ML invocations. Run this file using `streamlit run hr_dashboard.py`. |
| `analyze_model.py` | A script designed purely to evaluate the Random Forest model's logic. Runs validation checks, classification reports (Precision/Recall), analyzes probability percentiles, and verifies how SMOTE behaves on the dataset. |
| `find_profiles.py` | A programmatic tool that hunts through the dataset and model to extract distinct employee profiles. These are then saved to `test.txt` providing realistic "Low", "Medium", and "High" risk profiles you can manually enter into the UI to test it out. |
| `test.txt` | Output of `find_profiles.py`. A generated testing sheet containing specific employee stats to test out the Risk Predictor. |
| `attrition_available_10.pkl` | The pre-packaged dataset serialized as a pandas Pickle file. |
| `requirements.txt` | Defines exactly what libraries (and versions) are needed to successfully execute the python scripts. |

---

## 5. How to Run Locally

If presenting or running this locally, execute these steps:

1. **Clone the Repository / Access the Files**
2. **Setup Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the Dashboard**:
   ```bash
   streamlit run hr_dashboard.py
   ```
   The application will boot up at `http://localhost:8501`.

---

## 6. Personal Learning Takeaways & Interview Talking Points

If you are using this document to study, here are the key technical points you can discuss during a presentation or technical interview:
- **"I handled class imbalance using SMOTE:"** Explain *why* (attrition datasets naturally have a minority of quitters, causing false-high accuracy if ignored) and *how* (applied only to training data to prevent data leakage into the test set).
- **"I used the Apriori Algorithm to extract HR rules, focusing on Lift:"** Understand that support and confidence can sometimes be misleading in heavily skewed populations, so "Lift" represents the true strength of an association over random chance.
- **"I utilized Unsupervised Learning for Segmentation:"** Discuss the application of K-Means paired with Standard Scaling (so large income numbers don't override 1-4 satisfaction numbers), and the strategic use of PCA to allow complex multi-dimensional clusters to be plotted gracefully on a 2D graph.
- **"Feature Importance extraction:"** Explain how the Random Forest classifier allowed you to rank the key drivers (like Overtime, Distance from home, Monthly Income) creating actionable insights organically from the model's design.
- **"Modular Code Design:"** Point out how you isolated model validation (`analyze_model.py`) and test data extraction (`find_profiles.py`) from the frontend UI (`hr_dashboard.py`). This keeps the application file dedicated to UI and caching, while analytical tasks live securely in their own environments.

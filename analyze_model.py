import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_pickle('attrition_available_10.pkl')
df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True, errors='ignore')
df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)

clf_cols = ['Age', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
            'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
            'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome',
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
            'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion',
            'YearsWithCurrManager', 'BusinessTravel', 'Attrition_Binary']
df_clf = df[[c for c in clf_cols if c in df.columns]].copy()

# --- Impute missing values ---
cat_cols = df_clf.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df_clf.columns if c not in cat_cols and c != 'Attrition_Binary']
for col in num_cols:
    df_clf[col] = df_clf[col].fillna(df_clf[col].median())
for col in cat_cols:
    df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])
df_clf = df_clf.dropna(subset=['Attrition_Binary'])

print('=== AFTER IMPUTATION ===')
print('Rows:', len(df_clf))
print('Attrition distribution:')
print(df_clf['Attrition_Binary'].value_counts())
print()

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clf[col] = le.fit_transform(df_clf[col])
    encoders[col] = le

X = df_clf.drop(columns=['Attrition_Binary'])
y = df_clf['Attrition_Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f'Train set BEFORE SMOTE: {len(X_train)} rows, Attrition: {y_train.sum()} ({y_train.mean()*100:.1f}%)')

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f'Train set AFTER SMOTE: {len(X_train)} rows, Attrition: {y_train.sum()} ({y_train.mean()*100:.1f}%)')
print(f'Test set: {len(X_test)} rows, Attrition: {y_test.sum()} ({y_test.mean()*100:.1f}%)')

rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print()
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

print('Probability stats (class 1 = Left):')
print(f'  Mean: {y_proba.mean():.4f}')
print(f'  Median: {np.median(y_proba):.4f}')
print(f'  Min: {y_proba.min():.4f}')
print(f'  Max: {y_proba.max():.4f}')
print(f'  > 0.35 (Medium): {(y_proba > 0.35).sum()} ({(y_proba > 0.35).mean()*100:.1f}%)')
print(f'  > 0.60 (High): {(y_proba > 0.60).sum()} ({(y_proba > 0.60).mean()*100:.1f}%)')

for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f'  P{p}: {np.percentile(y_proba, p):.4f}')

# Test extreme profiles
print()
print('=== Testing Risk Profiles ===')
input1 = {
    'Age': 20, 'DistanceFromHome': 25, 'Education': 1,
    'EnvironmentSatisfaction': 1, 'JobInvolvement': 1,
    'JobLevel': 1, 'JobSatisfaction': 1,
    'MonthlyIncome': 10000, 'NumCompaniesWorked': 9,
    'PercentSalaryHike': 11, 'PerformanceRating': 3,
    'StockOptionLevel': 0, 'TotalWorkingYears': 1,
    'TrainingTimesLastYear': 0, 'WorkLifeBalance': 1,
    'YearsAtCompany': 0, 'YearsSinceLastPromotion': 0,
    'YearsWithCurrManager': 0,
    'Department': 'Sales', 'EducationField': 'Marketing',
    'Gender': 'Male', 'JobRole': 'Sales Representative', 'MaritalStatus': 'Single',
    'BusinessTravel': 'Travel_Frequently',
}
row = pd.DataFrame([input1])
for col_name in row.select_dtypes(include='object').columns:
    if col_name in encoders:
        try:
            row[col_name] = encoders[col_name].transform(row[col_name])
        except ValueError:
            row[col_name] = 0
row = row[[c for c in X.columns if c in row.columns]]
for c in X.columns:
    if c not in row.columns:
        row[c] = 0
row = row[X.columns]
prob1 = rf.predict_proba(row)[0][1]
risk1 = 'HIGH' if prob1 > 0.6 else 'MEDIUM' if prob1 > 0.35 else 'LOW'
print(f'Profile 1 (extreme high-risk): prob={prob1:.4f}, risk={risk1}')

input2 = {
    'Age': 45, 'DistanceFromHome': 1, 'Education': 4,
    'EnvironmentSatisfaction': 4, 'JobInvolvement': 4,
    'JobLevel': 4, 'JobSatisfaction': 4,
    'MonthlyIncome': 150000, 'NumCompaniesWorked': 1,
    'PercentSalaryHike': 25, 'PerformanceRating': 4,
    'StockOptionLevel': 3, 'TotalWorkingYears': 20,
    'TrainingTimesLastYear': 6, 'WorkLifeBalance': 4,
    'YearsAtCompany': 15, 'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 10,
    'Department': 'Research & Development', 'EducationField': 'Life Sciences',
    'Gender': 'Female', 'JobRole': 'Research Director', 'MaritalStatus': 'Married',
    'BusinessTravel': 'Non-Travel',
}
row2 = pd.DataFrame([input2])
for col_name in row2.select_dtypes(include='object').columns:
    if col_name in encoders:
        try:
            row2[col_name] = encoders[col_name].transform(row2[col_name])
        except ValueError:
            row2[col_name] = 0
row2 = row2[[c for c in X.columns if c in row2.columns]]
for c in X.columns:
    if c not in row2.columns:
        row2[c] = 0
row2 = row2[X.columns]
prob2 = rf.predict_proba(row2)[0][1]
risk2 = 'HIGH' if prob2 > 0.6 else 'MEDIUM' if prob2 > 0.35 else 'LOW'
print(f'Profile 2 (very low-risk):     prob={prob2:.4f}, risk={risk2}')

# Medium-risk profile
input3 = {
    'Age': 25, 'DistanceFromHome': 20, 'Education': 2,
    'EnvironmentSatisfaction': 2, 'JobInvolvement': 2,
    'JobLevel': 1, 'JobSatisfaction': 2,
    'MonthlyIncome': 20000, 'NumCompaniesWorked': 5,
    'PercentSalaryHike': 12, 'PerformanceRating': 3,
    'StockOptionLevel': 0, 'TotalWorkingYears': 3,
    'TrainingTimesLastYear': 1, 'WorkLifeBalance': 2,
    'YearsAtCompany': 1, 'YearsSinceLastPromotion': 3,
    'YearsWithCurrManager': 1,
    'Department': 'Sales', 'EducationField': 'Marketing',
    'Gender': 'Male', 'JobRole': 'Sales Representative', 'MaritalStatus': 'Single',
    'BusinessTravel': 'Travel_Frequently',
}
row3 = pd.DataFrame([input3])
for col_name in row3.select_dtypes(include='object').columns:
    if col_name in encoders:
        try:
            row3[col_name] = encoders[col_name].transform(row3[col_name])
        except ValueError:
            row3[col_name] = 0
row3 = row3[[c for c in X.columns if c in row3.columns]]
for c in X.columns:
    if c not in row3.columns:
        row3[c] = 0
row3 = row3[X.columns]
prob3 = rf.predict_proba(row3)[0][1]
risk3 = 'HIGH' if prob3 > 0.6 else 'MEDIUM' if prob3 > 0.35 else 'LOW'
print(f'Profile 3 (medium-risk):       prob={prob3:.4f}, risk={risk3}')

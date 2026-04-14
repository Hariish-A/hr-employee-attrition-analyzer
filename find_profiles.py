import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
cat_cols = df_clf.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df_clf.columns if c not in cat_cols and c != 'Attrition_Binary']
for col in num_cols:
    df_clf[col] = df_clf[col].fillna(df_clf[col].median())
for col in cat_cols:
    df_clf[col] = df_clf[col].fillna(df_clf[col].mode()[0])
df_clf = df_clf.dropna(subset=['Attrition_Binary'])

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clf[col] = le.fit_transform(df_clf[col])
    encoders[col] = le

X = df_clf.drop(columns=['Attrition_Binary'])
y = df_clf['Attrition_Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

def predict(profile):
    row = pd.DataFrame([profile])
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
    prob = rf.predict_proba(row)[0][1]
    risk = 'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.35 else 'LOW'
    return prob, risk

profiles = [
    # === LOW RISK PROFILES (target: < 35%) ===
    ("LOW #1 - Stable Senior R&D Director", {
        'Age': 45, 'DistanceFromHome': 2, 'Education': 4,
        'EnvironmentSatisfaction': 4, 'JobInvolvement': 4,
        'JobLevel': 4, 'JobSatisfaction': 4,
        'MonthlyIncome': 150000, 'NumCompaniesWorked': 1,
        'PercentSalaryHike': 22, 'PerformanceRating': 4,
        'StockOptionLevel': 3, 'TotalWorkingYears': 20,
        'TrainingTimesLastYear': 3, 'WorkLifeBalance': 4,
        'YearsAtCompany': 15, 'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 10,
        'Department': 'Research & Development', 'EducationField': 'Life Sciences',
        'Gender': 'Female', 'JobRole': 'Research Director', 'MaritalStatus': 'Married',
        'BusinessTravel': 'Non-Travel',
    }),
    ("LOW #2 - Happy Research Scientist", {
        'Age': 38, 'DistanceFromHome': 5, 'Education': 3,
        'EnvironmentSatisfaction': 4, 'JobInvolvement': 3,
        'JobLevel': 3, 'JobSatisfaction': 4,
        'MonthlyIncome': 80000, 'NumCompaniesWorked': 2,
        'PercentSalaryHike': 18, 'PerformanceRating': 4,
        'StockOptionLevel': 2, 'TotalWorkingYears': 15,
        'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
        'YearsAtCompany': 10, 'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 7,
        'Department': 'Research & Development', 'EducationField': 'Medical',
        'Gender': 'Male', 'JobRole': 'Research Scientist', 'MaritalStatus': 'Married',
        'BusinessTravel': 'Travel_Rarely',
    }),
    ("LOW #3 - Settled Manufacturing Director", {
        'Age': 50, 'DistanceFromHome': 3, 'Education': 4,
        'EnvironmentSatisfaction': 3, 'JobInvolvement': 3,
        'JobLevel': 4, 'JobSatisfaction': 3,
        'MonthlyIncome': 120000, 'NumCompaniesWorked': 2,
        'PercentSalaryHike': 15, 'PerformanceRating': 3,
        'StockOptionLevel': 2, 'TotalWorkingYears': 25,
        'TrainingTimesLastYear': 2, 'WorkLifeBalance': 3,
        'YearsAtCompany': 20, 'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 10,
        'Department': 'Research & Development', 'EducationField': 'Life Sciences',
        'Gender': 'Male', 'JobRole': 'Manufacturing Director', 'MaritalStatus': 'Married',
        'BusinessTravel': 'Non-Travel',
    }),

    # === MEDIUM RISK PROFILES (target: 35% - 60%) ===
    ("MEDIUM #1 - Restless Sales Executive", {
        'Age': 28, 'DistanceFromHome': 15, 'Education': 2,
        'EnvironmentSatisfaction': 2, 'JobInvolvement': 2,
        'JobLevel': 1, 'JobSatisfaction': 2,
        'MonthlyIncome': 25000, 'NumCompaniesWorked': 4,
        'PercentSalaryHike': 12, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 5,
        'TrainingTimesLastYear': 1, 'WorkLifeBalance': 2,
        'YearsAtCompany': 1, 'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 1,
        'Department': 'Sales', 'EducationField': 'Marketing',
        'Gender': 'Male', 'JobRole': 'Sales Executive', 'MaritalStatus': 'Single',
        'BusinessTravel': 'Travel_Frequently',
    }),
    ("MEDIUM #2 - Disengaged Sales Rep", {
        'Age': 30, 'DistanceFromHome': 10, 'Education': 2,
        'EnvironmentSatisfaction': 1, 'JobInvolvement': 2,
        'JobLevel': 1, 'JobSatisfaction': 2,
        'MonthlyIncome': 30000, 'NumCompaniesWorked': 5,
        'PercentSalaryHike': 13, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 6,
        'TrainingTimesLastYear': 2, 'WorkLifeBalance': 2,
        'YearsAtCompany': 2, 'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 1,
        'Department': 'Sales', 'EducationField': 'Marketing',
        'Gender': 'Female', 'JobRole': 'Sales Representative', 'MaritalStatus': 'Single',
        'BusinessTravel': 'Travel_Frequently',
    }),
    ("MEDIUM #3 - Overlooked Lab Technician", {
        'Age': 28, 'DistanceFromHome': 15, 'Education': 2,
        'EnvironmentSatisfaction': 2, 'JobInvolvement': 2,
        'JobLevel': 1, 'JobSatisfaction': 2,
        'MonthlyIncome': 25000, 'NumCompaniesWorked': 3,
        'PercentSalaryHike': 11, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 4,
        'TrainingTimesLastYear': 1, 'WorkLifeBalance': 2,
        'YearsAtCompany': 2, 'YearsSinceLastPromotion': 2,
        'YearsWithCurrManager': 1,
        'Department': 'Research & Development', 'EducationField': 'Life Sciences',
        'Gender': 'Male', 'JobRole': 'Laboratory Technician', 'MaritalStatus': 'Married',
        'BusinessTravel': 'Travel_Rarely',
    }),

    # === HIGH RISK PROFILES (target: > 60%) ===
    ("HIGH #1 - Burnt-Out Lab Technician", {
        'Age': 20, 'DistanceFromHome': 25, 'Education': 1,
        'EnvironmentSatisfaction': 1, 'JobInvolvement': 1,
        'JobLevel': 1, 'JobSatisfaction': 1,
        'MonthlyIncome': 12000, 'NumCompaniesWorked': 7,
        'PercentSalaryHike': 11, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 1,
        'TrainingTimesLastYear': 0, 'WorkLifeBalance': 1,
        'YearsAtCompany': 0, 'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0,
        'Department': 'Research & Development', 'EducationField': 'Technical Degree',
        'Gender': 'Male', 'JobRole': 'Laboratory Technician', 'MaritalStatus': 'Single',
        'BusinessTravel': 'Travel_Frequently',
    }),
    ("HIGH #2 - Neglected HR Newcomer", {
        'Age': 22, 'DistanceFromHome': 28, 'Education': 1,
        'EnvironmentSatisfaction': 1, 'JobInvolvement': 1,
        'JobLevel': 1, 'JobSatisfaction': 1,
        'MonthlyIncome': 10000, 'NumCompaniesWorked': 8,
        'PercentSalaryHike': 11, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 1,
        'TrainingTimesLastYear': 0, 'WorkLifeBalance': 1,
        'YearsAtCompany': 0, 'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0,
        'Department': 'Human Resources', 'EducationField': 'Human Resources',
        'Gender': 'Female', 'JobRole': 'Human Resources', 'MaritalStatus': 'Single',
        'BusinessTravel': 'Travel_Frequently',
    }),
    ("HIGH #3 - Underpaid Young Lab Tech (Long Commute)", {
        'Age': 18, 'DistanceFromHome': 29, 'Education': 1,
        'EnvironmentSatisfaction': 1, 'JobInvolvement': 1,
        'JobLevel': 1, 'JobSatisfaction': 1,
        'MonthlyIncome': 10000, 'NumCompaniesWorked': 5,
        'PercentSalaryHike': 11, 'PerformanceRating': 3,
        'StockOptionLevel': 0, 'TotalWorkingYears': 0,
        'TrainingTimesLastYear': 0, 'WorkLifeBalance': 1,
        'YearsAtCompany': 0, 'YearsSinceLastPromotion': 0,
        'YearsWithCurrManager': 0,
        'Department': 'Research & Development', 'EducationField': 'Technical Degree',
        'Gender': 'Female', 'JobRole': 'Laboratory Technician', 'MaritalStatus': 'Single',
        'BusinessTravel': 'Travel_Frequently',
    }),
]

results = []
for name, profile in profiles:
    prob, risk = predict(profile)
    results.append((name, prob, risk, profile))
    print(f'{risk:6s} ({prob:.2%}) | {name}')

# Write file
with open('test.txt', 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('  HR ATTRITION RISK PREDICTOR - TEST VALUES FOR DASHBOARD\n')
    f.write('=' * 80 + '\n')
    f.write('\n  Use these values on the "Risk Predictor" page.\n')
    f.write('  Fields are listed in the same order as the dashboard form.\n')
    f.write('\n  Risk Thresholds:\n')
    f.write('    LOW    = score <= 35%\n')
    f.write('    MEDIUM = score 35% - 60%\n')
    f.write('    HIGH   = score > 60%\n')
    f.write('\n  Note: Some fields (Education, Job Level, Performance Rating,\n')
    f.write('  Percent Salary Hike) are hardcoded in the dashboard form.\n')
    f.write('  The values listed under "Hardcoded defaults" below show what\n')
    f.write('  the dashboard uses - you cannot change them via the UI.\n\n')

    for name, prob, risk, profile in results:
        f.write('-' * 80 + '\n')
        f.write(f'  {name}\n')
        f.write(f'  >>> Expected Risk: {risk} ({prob:.1%})\n')
        f.write('-' * 80 + '\n\n')
        f.write('  Row 1:\n')
        f.write(f'    Age                        = {profile["Age"]}\n')
        f.write(f'    Monthly Income             = {profile["MonthlyIncome"]}\n')
        f.write(f'    Distance From Home (km)    = {profile["DistanceFromHome"]}\n')
        f.write(f'    Years at Company           = {profile["YearsAtCompany"]}\n')
        f.write('\n  Row 2:\n')
        f.write(f'    Job Satisfaction (1-4)     = {profile["JobSatisfaction"]}\n')
        f.write(f'    Environment Satisfaction   = {profile["EnvironmentSatisfaction"]}\n')
        f.write(f'    Work Life Balance          = {profile["WorkLifeBalance"]}\n')
        f.write(f'    Job Involvement            = {profile["JobInvolvement"]}\n')
        f.write('\n  Row 3:\n')
        f.write(f'    Department                 = {profile["Department"]}\n')
        f.write(f'    Job Role                   = {profile["JobRole"]}\n')
        f.write(f'    Gender                     = {profile["Gender"]}\n')
        f.write(f'    Marital Status             = {profile["MaritalStatus"]}\n')
        f.write('\n  Row 4:\n')
        f.write(f'    Business Travel            = {profile["BusinessTravel"]}\n')
        f.write(f'    Education Field            = {profile["EducationField"]}\n')
        f.write(f'    Stock Option Level         = {profile["StockOptionLevel"]}\n')
        f.write(f'    Years Since Last Promotion = {profile["YearsSinceLastPromotion"]}\n')
        f.write('\n  Row 5:\n')
        f.write(f'    Total Working Years        = {profile["TotalWorkingYears"]}\n')
        f.write(f'    Training Times Last Year   = {profile["TrainingTimesLastYear"]}\n')
        f.write(f'    Num Companies Worked       = {profile["NumCompaniesWorked"]}\n')
        f.write(f'    Years With Curr Manager    = {profile["YearsWithCurrManager"]}\n')
        f.write(f'\n  Hardcoded defaults: Education={profile["Education"]}, ')
        f.write(f'Job Level={profile["JobLevel"]}, ')
        f.write(f'Perf Rating={profile["PerformanceRating"]}, ')
        f.write(f'Salary Hike={profile["PercentSalaryHike"]}%\n\n')

print('\n=> test.txt written successfully!')

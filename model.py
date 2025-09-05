# data analysis and wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\Anand\Desktop\archive\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# drop the unnecessary columns
df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)

# Change Factors to Numerics
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == "Yes" else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == "Yes" else 0)

# 'Environment Satisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorklifeBalance' can
# be clubbed into a single feature 'TotalSatisfaction'
df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                            df['JobInvolvement'] +
                            df['JobSatisfaction'] +
                            df['RelationshipSatisfaction'] +
                            df['WorkLifeBalance']) / 5

# Drop Columns
df.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction',
         'RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)

# Convert Total satisfaction into boolean
df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
df.drop('Total_Satisfaction', axis=1, inplace=True)

# Age -> boolean
df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
df.drop('Age', axis=1, inplace=True)

# DailyRate -> boolean
df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
df.drop('DailyRate', axis=1, inplace=True)

# Department -> boolean
df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
df.drop('Department', axis=1, inplace=True)

# DistanceFromHome -> boolean
df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
df.drop('DistanceFromHome', axis=1, inplace=True)

# JobRole -> boolean
df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
df.drop('JobRole', axis=1, inplace=True)

# HourlyRate -> boolean
df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
df.drop('HourlyRate', axis=1, inplace=True)

# MonthlyIncome -> boolean
df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
df.drop('MonthlyIncome', axis=1, inplace=True)

# NumCompaniesWorked -> boolean
df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
df.drop('NumCompaniesWorked', axis=1, inplace=True)

# TotalWorkingYears -> boolean
df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
df.drop('TotalWorkingYears', axis=1, inplace=True)

# YearsAtCompany -> boolean
df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
df.drop('YearsAtCompany', axis=1, inplace=True)

# YearsInCurrentRole -> boolean
df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
df.drop('YearsInCurrentRole', axis=1, inplace=True)

# YearsSinceLastPromotion -> boolean
df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

# YearsWithCurrManager -> boolean
df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
df.drop('YearsWithCurrManager', axis=1, inplace=True)

# Gender -> numeric
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Drop more unused columns
df.drop('MonthlyRate', axis=1, inplace=True)
df.drop('PercentSalaryHike', axis=1, inplace=True)

# Convert to category
convert_category = ['BusinessTravel','Education','EducationField','MaritalStatus',
                    'StockOptionLevel','OverTime','Gender','TrainingTimesLastYear']
for col in convert_category:
    df[col] = df[col].astype('category')

# Separate categorical and numerical
X_categorical = df.select_dtypes(include=['category'])
X_numerical = df.select_dtypes(include=['int64'])
X_numerical.drop('Attrition', axis=1, inplace=True)

y = df['Attrition']

# One Hot Encoding
onehotencoder = OneHotEncoder()
X_categorical = onehotencoder.fit_transform(X_categorical).toarray()
X_categorical = pd.DataFrame(X_categorical)

# Combine categorical + numerical
X_all = pd.concat([X_categorical, X_numerical], axis=1)

# ✅ Fix mixed column names
X_all.columns = X_all.columns.astype(str)

# Split Test and Train Data
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.20)

# Train model
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

# Save model
pickle.dump(regressor, open('model.pkl','wb'))

# Load model to verify
model = pickle.load(open('model.pkl','rb'))

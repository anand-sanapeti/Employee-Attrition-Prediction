import numpy as np
import scipy as sp
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    try:
        Age = request.form.get("Age")
        BusinessTravel = request.form['BusinessTravel']
        DailyRate = request.form.get('Daily Rate')
        Department = request.form['Department']
        DistanceFromHome = request.form.get("Distance From Home")
        Education = request.form.get("Education")
        EducationField = request.form['Education Field']
        EnvironmentSatisfaction = request.form.get("Environment Satisfaction")
        Gender = request.form['Gender']
        HourlyRate = request.form.get("Hourly Rate")
        JobInvolvement = request.form.get("Job Involvement")  # FIXED: was getting Environment Satisfaction
        JobLevel = request.form.get("Job Level")
        JobRole = request.form['Job Role']
        JobSatisfaction = request.form.get("Job Satisfaction")
        MaritalStatus = request.form['Marital Status']
        MonthlyIncome = request.form.get("Monthly Income")
        NumCompaniesWorked = request.form.get("Number of Companies Worked in")
        OverTime = request.form['Over Time']
        PerformanceRating = request.form.get("Performance Rating")
        RelationshipSatisfaction = request.form.get("Relationship Satisfaction")
        StockOptionLevel = request.form.get("Stock Option Level")
        TotalWorkingYears = request.form.get("Total Working Years")
        TrainingTimesLastYear = request.form.get("Training Times Last Year")
        WorkLifeBalance = request.form.get("Work Life Balance")
        YearsAtCompany = request.form.get("Years At Company")
        YearsInCurrentRole = request.form.get("Years In Current Role")
        YearsSinceLastPromotion = request.form.get("Years Since Last Promotion")
        YearsWithCurrManager = request.form.get("Years With Curr Manager")

        dict_data = {
            'Age': int(Age),
            'BusinessTravel': str(BusinessTravel),
            'DailyRate': int(DailyRate),
            'Department': Department,
            'DistanceFromHome': int(DistanceFromHome),
            'Education': int(Education),
            'EducationField': str(EducationField),
            'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
            'Gender': str(Gender),
            'HourlyRate': int(HourlyRate),
            'JobInvolvement': int(JobInvolvement),
            'JobLevel': int(JobLevel),
            'JobRole': JobRole,
            'JobSatisfaction': int(JobSatisfaction),
            'MaritalStatus': str(MaritalStatus),
            'MonthlyIncome': int(MonthlyIncome),
            'NumCompaniesWorked': int(NumCompaniesWorked),
            'OverTime': str(OverTime),
            'PerformanceRating': int(PerformanceRating),
            'RelationshipSatisfaction': int(RelationshipSatisfaction),
            'StockOptionLevel': int(StockOptionLevel),
            'TotalWorkingYears': int(TotalWorkingYears),
            'TrainingTimesLastYear': int(TrainingTimesLastYear),
            'WorkLifeBalance': int(WorkLifeBalance),
            'YearsAtCompany': int(YearsAtCompany),
            'YearsInCurrentRole': int(YearsInCurrentRole),
            'YearsSinceLastPromotion': int(YearsSinceLastPromotion),
            'YearsWithCurrManager': int(YearsWithCurrManager)
        }

        df = process_features(dict_data)
        prediction = model.predict(df)

        if prediction == 0:
            return render_template('index.html', prediction_text='✅ Employee Might Not Leave The Job', prediction_class='success')
        else:
            return render_template('index.html', prediction_text='⚠️ Employee Might Leave The Job', prediction_class='warning')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error: {str(e)}', prediction_class='error')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    For API requests (JSON)
    """
    try:
        data = request.get_json()
        df = process_features(data)
        prediction = model.predict(df)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'message': 'Employee Might Not Leave' if prediction[0] == 0 else 'Employee Might Leave'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# def process_features(data):
#     """
#     Process and transform features for prediction
#     """
#     df = pd.DataFrame([data])

#     # Calculate Total Satisfaction
#     df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
#                                 df['JobInvolvement'] +
#                                 df['JobSatisfaction'] +
#                                 df['RelationshipSatisfaction'] +
#                                 df['WorkLifeBalance']) / 5

#     # Drop original satisfaction columns
#     df.drop(['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 
#              'RelationshipSatisfaction', 'WorkLifeBalance'], axis=1, inplace=True)

#     # Convert Total satisfaction into boolean
#     df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
#     df.drop('Total_Satisfaction', axis=1, inplace=True)

#     # Feature engineering - create boolean features
#     df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
#     df.drop('Age', axis=1, inplace=True)

#     df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
#     df.drop('DailyRate', axis=1, inplace=True)

#     df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
#     df.drop('Department', axis=1, inplace=True)

#     df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
#     df.drop('DistanceFromHome', axis=1, inplace=True)

#     df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
#     df.drop('JobRole', axis=1, inplace=True)

#     df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
#     df.drop('HourlyRate', axis=1, inplace=True)

#     df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
#     df.drop('MonthlyIncome', axis=1, inplace=True)

#     df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
#     df.drop('NumCompaniesWorked', axis=1, inplace=True)

#     df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
#     df.drop('TotalWorkingYears', axis=1, inplace=True)

#     df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
#     df.drop('YearsAtCompany', axis=1, inplace=True)

#     df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
#     df.drop('YearsInCurrentRole', axis=1, inplace=True)

#     df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
#     df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

#     df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
#     df.drop('YearsWithCurrManager', axis=1, inplace=True)

#     # Encode categorical variables
#     encode_business_travel(df, data['BusinessTravel'])
#     encode_education(df, data['Education'])
#     encode_education_field(df, data['EducationField'])
#     encode_gender(df, data['Gender'])
#     encode_marital_status(df, data['MaritalStatus'])
#     encode_overtime(df, data['OverTime'])
#     encode_stock_option(df, data['StockOptionLevel'])
#     encode_training_times(df, data['TrainingTimesLastYear'])

#     return df
def process_features(data):
    """
    Process and transform features for prediction
    """

    # Create DF first
    df = pd.DataFrame([data])

    # ----------------------------- #
    #  Feature Engineering
    # ----------------------------- #

    # Total Satisfaction
    df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                df['JobInvolvement'] +
                                df['JobSatisfaction'] +
                                df['RelationshipSatisfaction'] +
                                df['WorkLifeBalance']) / 5

    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)

    df.drop(['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
             'RelationshipSatisfaction', 'WorkLifeBalance',
             'Total_Satisfaction'], axis=1, inplace=True)

    # Boolean features
    df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
    df.drop('Age', axis=1, inplace=True)

    df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
    df.drop('DailyRate', axis=1, inplace=True)

    df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
    df.drop('Department', axis=1, inplace=True)

    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
    df.drop('DistanceFromHome', axis=1, inplace=True)

    df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
    df.drop('JobRole', axis=1, inplace=True)

    df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
    df.drop('HourlyRate', axis=1, inplace=True)

    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
    df.drop('MonthlyIncome', axis=1, inplace=True)

    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
    df.drop('NumCompaniesWorked', axis=1, inplace=True)

    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
    df.drop('TotalWorkingYears', axis=1, inplace=True)

    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsAtCompany', axis=1, inplace=True)

    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsInCurrentRole', axis=1, inplace=True)

    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsWithCurrManager', axis=1, inplace=True)

    # ----------------------------- #
    #  FIXED ONE-HOT ENCODERS
    # ----------------------------- #
    
    # Business Travel (match training data names)
    df['BusinessTravel_Travel_Rarely'] = 1 if data['BusinessTravel'] == 'Rarely' else 0
    df['BusinessTravel_Travel_Frequently'] = 1 if data['BusinessTravel'] == 'Frequently' else 0
    df['BusinessTravel_Non-Travel'] = 1 if data['BusinessTravel'] == 'No Travel' else 0
    df.drop('BusinessTravel', axis=1, inplace=True)

    # Education Field (match training names)
    fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
    for f in fields:
        col = f"EducationField_{f}"
        df[col] = 1 if data['EducationField'] == f else 0
    df.drop("EducationField", axis=1, inplace=True)

    # Gender
    df['Gender_Male'] = 1 if data['Gender'] == 'Male' else 0
    df['Gender_Female'] = 1 if data['Gender'] == 'Female' else 0
    df.drop("Gender", axis=1, inplace=True)

    # Marital Status
    df['MaritalStatus_Married'] = 1 if data['MaritalStatus'] == 'Married' else 0
    df['MaritalStatus_Single'] = 1 if data['MaritalStatus'] == 'Single' else 0
    df['MaritalStatus_Divorced'] = 1 if data['MaritalStatus'] == 'Divorced' else 0
    df.drop("MaritalStatus", axis=1, inplace=True)

    # OverTime
    df['OverTime_Yes'] = 1 if data['OverTime'] == 'Yes' else 0
    df['OverTime_No'] = 1 if data['OverTime'] == 'No' else 0
    df.drop("OverTime", axis=1, inplace=True)

    # Education Level
    for i in range(1,6):
        df[f"Education_{i}"] = 1 if int(data['Education']) == i else 0
    df.drop("Education", axis=1, inplace=True)

    # Training Times
    for i in range(7):
        df[f"TrainingTimesLastYear_{i}"] = 1 if int(data['TrainingTimesLastYear']) == i else 0
    df.drop("TrainingTimesLastYear", axis=1, inplace=True)

    # Stock Option Level
    for i in range(4):
        df[f"StockOptionLevel_{i}"] = 1 if int(data['StockOptionLevel']) == i else 0
    df.drop("StockOptionLevel", axis=1, inplace=True)

    # ----------------------------- #
    #  FINAL COLUMN ALIGNMENT
    # ----------------------------- #

    training_columns = pickle.load(open("feature_names.pkl", "rb"))

    # Add missing cols
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop extra cols
    df = df[training_columns]

    return df


# Load training columns
# training_columns = pickle.load(open("feature_names.pkl", "rb"))

# # Add missing columns
# for col in training_columns:
#     if col not in df.columns:
#         df[col] = 0

# Remove any extra columns not seen during training
# df = df[training_columns]

# def encode_business_travel(df, value):
#     df['BusinessTravel_Rarely'] = 1 if value == 'Rarely' else 0
#     df['BusinessTravel_Frequently'] = 1 if value == 'Frequently' else 0
#     df['BusinessTravel_No_Travel'] = 1 if value == 'No Travel' else 0
#     df.drop('BusinessTravel', axis=1, inplace=True)
def encode_business_travel(df, value):
    df['BusinessTravel_Travel_Rarely'] = 1 if value == 'Rarely' else 0
    df['BusinessTravel_Travel_Frequently'] = 1 if value == 'Frequently' else 0
    df['BusinessTravel_Non-Travel'] = 1 if value == 'No Travel' else 0
    df.drop('BusinessTravel', axis=1, inplace=True)



def encode_education(df, value):
    for i in range(1, 6):
        df[f'Education_{i}'] = 1 if int(value) == i else 0
    df.drop('Education', axis=1, inplace=True)


# def encode_education_field(df, value):
#     fields = ['Life_Sciences', 'Medical', 'Marketing', 'Technical_Degree', 'Human_Resources', 'Other']
#     field_map = {
#         'Life Sciences': 'Life_Sciences',
#         'Medical': 'Medical',
#         'Marketing': 'Marketing',
#         'Technical Degree': 'Technical_Degree',
#         'Human Resources': 'Human_Resources',
#         'Other': 'Other'
#     }
    
#     selected = field_map.get(value, 'Other')
#     for field in fields:
#         if field == 'Human_Resources':
#             df['Education_Human_Resources'] = 1 if selected == field else 0
#         elif field == 'Other':
#             df['Education_Other'] = 1 if selected == field else 0
#         else:
#             df[f'EducationField_{field}'] = 1 if selected == field else 0
#     df.drop('EducationField', axis=1, inplace=True)
def encode_education_field(df, value):
    fields = [
        "Life Sciences", 
        "Medical", 
        "Marketing", 
        "Technical Degree", 
        "Human Resources", 
        "Other"
    ]

    for f in fields:
        col = f"EducationField_{f}"
        df[col] = 1 if value == f else 0

    df.drop("EducationField", axis=1, inplace=True)



def encode_gender(df, value):
    df['Gender_Male'] = 1 if value == 'Male' else 0
    df['Gender_Female'] = 1 if value == 'Female' else 0
    df.drop('Gender', axis=1, inplace=True)


def encode_marital_status(df, value):
    df['MaritalStatus_Married'] = 1 if value == 'Married' else 0
    df['MaritalStatus_Single'] = 1 if value == 'Single' else 0
    df['MaritalStatus_Divorced'] = 1 if value == 'Divorced' else 0
    df.drop('MaritalStatus', axis=1, inplace=True)


def encode_overtime(df, value):
    df['OverTime_Yes'] = 1 if value == 'Yes' else 0
    df['OverTime_No'] = 1 if value == 'No' else 0
    df.drop('OverTime', axis=1, inplace=True)


def encode_stock_option(df, value):
    for i in range(4):
        df[f'StockOptionLevel_{i}'] = 1 if int(value) == i else 0
    df.drop('StockOptionLevel', axis=1, inplace=True)


def encode_training_times(df, value):
    for i in range(7):
        df[f'TrainingTimesLastYear_{i}'] = 1 if int(value) == i else 0
    df.drop('TrainingTimesLastYear', axis=1, inplace=True)


if __name__ == "__main__":
    app.run(debug=True)
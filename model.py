import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/employee data.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n" + "="*50)
print("PREPROCESSING DATA")
print("="*50)

# Create a copy for processing
df_processed = df.copy()

# Calculate Total Satisfaction
df_processed['Total_Satisfaction'] = (
    df_processed['EnvironmentSatisfaction'] +
    df_processed['JobInvolvement'] +
    df_processed['JobSatisfaction'] +
    df_processed['RelationshipSatisfaction'] +
    df_processed['WorkLifeBalance']
) / 5

# Drop original satisfaction columns
df_processed.drop([
    'EnvironmentSatisfaction', 
    'JobInvolvement', 
    'JobSatisfaction',
    'RelationshipSatisfaction', 
    'WorkLifeBalance'
], axis=1, inplace=True)

# Convert Total satisfaction into boolean
df_processed['Total_Satisfaction_bool'] = df_processed['Total_Satisfaction'].apply(
    lambda x: 1 if x >= 2.8 else 0
)
# df_processed.drop('(Total_Satisfaction)', axis=1, inplace=True)
df_processed.drop('Total_Satisfaction', axis=1, inplace=True)


# Feature engineering - create boolean features based on attrition patterns
print("\nCreating engineered features...")

# Age: attrition is high for employees below 35
df_processed['Age_bool'] = df_processed['Age'].apply(lambda x: 1 if x < 35 else 0)
df_processed.drop('Age', axis=1, inplace=True)

# Daily Rate: attrition is high if daily rate < 800
df_processed['DailyRate_bool'] = df_processed['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
df_processed.drop('DailyRate', axis=1, inplace=True)

# Department: R&D has higher attrition
df_processed['Department_bool'] = df_processed['Department'].apply(
    lambda x: 1 if x == 'Research & Development' else 0
)
df_processed.drop('Department', axis=1, inplace=True)

# Distance From Home: attrition is high if distance > 10
df_processed['DistanceFromHome_bool'] = df_processed['DistanceFromHome'].apply(
    lambda x: 1 if x > 10 else 0
)
df_processed.drop('DistanceFromHome', axis=1, inplace=True)

# Job Role: Laboratory Technicians have higher attrition
df_processed['JobRole_bool'] = df_processed['JobRole'].apply(
    lambda x: 1 if x == 'Laboratory Technician' else 0
)
df_processed.drop('JobRole', axis=1, inplace=True)

# Hourly Rate: attrition is high if hourly rate < 65
df_processed['HourlyRate_bool'] = df_processed['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
df_processed.drop('HourlyRate', axis=1, inplace=True)

# Monthly Income: attrition is high if income < 4000
df_processed['MonthlyIncome_bool'] = df_processed['MonthlyIncome'].apply(
    lambda x: 1 if x < 4000 else 0
)
df_processed.drop('MonthlyIncome', axis=1, inplace=True)

# Num Companies Worked: attrition is high if > 3
df_processed['NumCompaniesWorked_bool'] = df_processed['NumCompaniesWorked'].apply(
    lambda x: 1 if x > 3 else 0
)
df_processed.drop('NumCompaniesWorked', axis=1, inplace=True)

# Total Working Years: attrition is high if < 8
df_processed['TotalWorkingYears_bool'] = df_processed['TotalWorkingYears'].apply(
    lambda x: 1 if x < 8 else 0
)
df_processed.drop('TotalWorkingYears', axis=1, inplace=True)

# Years At Company: attrition is high if < 3
df_processed['YearsAtCompany_bool'] = df_processed['YearsAtCompany'].apply(
    lambda x: 1 if x < 3 else 0
)
df_processed.drop('YearsAtCompany', axis=1, inplace=True)

# Years In Current Role: attrition is high if < 3
df_processed['YearsInCurrentRole_bool'] = df_processed['YearsInCurrentRole'].apply(
    lambda x: 1 if x < 3 else 0
)
df_processed.drop('YearsInCurrentRole', axis=1, inplace=True)

# Years Since Last Promotion: attrition is high if < 1
df_processed['YearsSinceLastPromotion_bool'] = df_processed['YearsSinceLastPromotion'].apply(
    lambda x: 1 if x < 1 else 0
)
df_processed.drop('YearsSinceLastPromotion', axis=1, inplace=True)

# Years With Current Manager: attrition is high if < 1
df_processed['YearsWithCurrManager_bool'] = df_processed['YearsWithCurrManager'].apply(
    lambda x: 1 if x < 1 else 0
)
df_processed.drop('YearsWithCurrManager', axis=1, inplace=True)

# Drop columns that won't be useful for prediction
columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18']
for col in columns_to_drop:
    if col in df_processed.columns:
        df_processed.drop(col, axis=1, inplace=True)

# Encode categorical variables
print("\nEncoding categorical variables...")

# One-hot encoding for categorical features
df_processed = pd.get_dummies(df_processed, columns=[
    'BusinessTravel', 
    'EducationField', 
    'Gender', 
    'MaritalStatus', 
    'OverTime'
])

# One-hot encoding for ordinal features
df_processed = pd.get_dummies(df_processed, columns=[
    'Education',
    'JobLevel',
    'StockOptionLevel',
    'TrainingTimesLastYear',
    'PerformanceRating'
])

# Encode target variable
print("\nEncoding target variable...")
le = LabelEncoder()
df_processed['Attrition'] = le.fit_transform(df_processed['Attrition'])

print(f"\nProcessed dataset shape: {df_processed.shape}")
print(f"Features: {df_processed.columns.tolist()}")

# Separate features and target
X = df_processed.drop('Attrition', axis=1)
y = df_processed['Attrition']

print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Attrition rate: {(y.sum() / len(y)) * 100:.2f}%")

# Split the data
print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train the model
print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
print("Training Logistic Regression model...")
model.fit(X_train, y_train)
print("Training completed!")

# Make predictions
print("\n" + "="*50)
print("EVALUATING MODEL")
print("="*50)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Stay', 'Leave']))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Stay', 'Leave'], 
            yticklabels=['Stay', 'Leave'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15)[['feature', 'coefficient']])

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['coefficient'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")

# Save the model
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as 'model.pkl'")

# Save feature names for reference
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)

print("Feature names saved as 'feature_names.pkl'")

# Test the saved model
print("\n" + "="*50)
print("TESTING SAVED MODEL")
print("="*50)

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

test_prediction = loaded_model.predict(X_test[:5])
print(f"\nTest predictions on first 5 samples: {test_prediction}")
print(f"Actual values: {y_test[:5].values}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print("\nFiles generated:")
print("1. model.pkl - Trained model")
print("2. feature_names.pkl - Feature names for reference")
print("3. confusion_matrix.png - Confusion matrix visualization")
print("4. feature_importance.png - Feature importance plot")
print("\nYou can now run the Flask application with: python app.py")
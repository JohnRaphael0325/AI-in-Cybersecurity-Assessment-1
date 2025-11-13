import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Simple rule-based prediction function
def expert_system(row):
    if row['Glucose'] > 140 and row['BMI'] > 30:
        return 1  # Diabetic
    elif row['Glucose'] > 130 and row['Age'] > 40:
        return 1
    else:
        return 0  # Non-diabetic

# Apply the rules to each row
data['Prediction_Expert'] = data.apply(expert_system, axis=1)

# Calculate metrics
accuracy = accuracy_score(data['Outcome'], data['Prediction_Expert']) * 100
precision = precision_score(data['Outcome'], data['Prediction_Expert']) * 100
recall = recall_score(data['Outcome'], data['Prediction_Expert']) * 100
f1 = f1_score(data['Outcome'], data['Prediction_Expert']) * 100

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-score: {f1:.2f}%")

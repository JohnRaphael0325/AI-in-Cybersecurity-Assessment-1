import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Weighted rule-based prediction function
def weighted_rule_system(row):
    score = 0
    score += 0.6 * (1 if row['Glucose'] > 140 else 0)
    score += 0.3 * (1 if row['BMI'] > 30 else 0)
    score += 0.1 * (1 if row['Age'] > 40 else 0)

    # If total score passes 0.5, classify as diabetic
    return 1 if score >= 0.5 else 0

# Apply rules to each row
data['Prediction_Weighted'] = data.apply(weighted_rule_system, axis=1)

# Calculate metrics
accuracy = accuracy_score(data['Outcome'], data['Prediction_Weighted']) * 100
precision = precision_score(data['Outcome'], data['Prediction_Weighted']) * 100
recall = recall_score(data['Outcome'], data['Prediction_Weighted']) * 100
f1 = f1_score(data['Outcome'], data['Prediction_Weighted']) * 100

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-score: {f1:.2f}%")
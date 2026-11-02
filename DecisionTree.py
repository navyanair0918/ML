import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Edit these 10 sample rows for your problem.
data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 65, 60, 70, 75, 80, 85, 90, 95],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)
print("Dataset (Pass: 0 = No, 1 = Yes):")
print(df)

X = df[["StudyHours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClassification metrics (test data):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (Pass=Yes):", precision_score(y_test, y_pred, zero_division=0))
print("Recall (Pass=Yes):", recall_score(y_test, y_pred, zero_division=0))
print("F1 score (Pass=Yes):", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))
new_student = pd.DataFrame(
    {"StudyHours": [hours], "Attendance": [attendance]}
)
prediction = model.predict(new_student)[0]
print("Prediction:", "Pass" if prediction == 1 else "Fail")

plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
)
plt.title("Decision Tree: Student Pass Prediction")
plt.tight_layout()
plt.show()

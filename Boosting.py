import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

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

model = GradientBoostingClassifier(
    n_estimators=50, learning_rate=0.1, random_state=42
)
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

plt.scatter(
    df.loc[y == 0, "StudyHours"],
    df.loc[y == 0, "Attendance"],
    color="red",
    label="Fail (0)",
)
plt.scatter(
    df.loc[y == 1, "StudyHours"],
    df.loc[y == 1, "Attendance"],
    color="blue",
    label="Pass (1)",
)
plt.scatter(
    hours,
    attendance,
    color="green",
    marker="*",
    s=180,
    label=f"New student: {'Pass' if prediction == 1 else 'Fail'}",
)
plt.xlabel("Study hours")
plt.ylabel("Attendance (%)")
plt.title("Gradient Boosting: Student Pass Prediction")
plt.legend()
plt.grid(True)
plt.show()

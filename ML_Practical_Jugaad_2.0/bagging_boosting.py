# BAGGING AND BOOSTING
# Case study: Predict whether a student will pass based on attendance and study hours.
# Bagging: Random Forest. Boosting: AdaBoost.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 10 rows of assumed data
# Columns: [attendance percentage, study hours per day]
X = np.array([
    [45, 1], [50, 1.5], [55, 2], [60, 2.5], [62, 2],
    [70, 3], [75, 4], [80, 5], [90, 6], [95, 7]
], dtype=float)

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Decision Tree baseline
dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:", round(dt_accuracy, 4))

# Bagging using Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest (Bagging) Accuracy:", round(rf_accuracy, 4))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n",
      classification_report(y_test, y_pred_rf, zero_division=0))

# Boosting using AdaBoost
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print("\nAdaBoost (Boosting) Accuracy:", round(ada_accuracy, 4))
print("AdaBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))
print("AdaBoost Classification Report:\n",
      classification_report(y_test, y_pred_ada, zero_division=0))

# Compare models
print("\nMODEL COMPARISON")
print("Decision Tree:", round(dt_accuracy * 100, 2), "%")
print("Random Forest (Bagging):", round(rf_accuracy * 100, 2), "%")
print("AdaBoost (Boosting):", round(ada_accuracy * 100, 2), "%")

models = ["Decision Tree", "Random Forest", "AdaBoost"]
accuracies = [dt_accuracy, rf_accuracy, ada_accuracy]
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.05)
plt.show()

# Predict for a new student: attendance 78%, study 4 hours/day
new_student = np.array([[78, 4]])
print("New student prediction (Random Forest):",
      "Pass" if rf_model.predict(new_student)[0] == 1 else "Fail")

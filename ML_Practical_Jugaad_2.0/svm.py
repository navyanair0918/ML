# SUPPORT VECTOR MACHINE (SVM)
# Case study: Classify whether a student passes based on attendance and study hours.
# Target: 0 = Fail, 1 = Pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 10 rows of assumed data
# Features: attendance percentage, study hours per day
X = np.array([
    [40, 1], [45, 1.5], [50, 2], [55, 2],
    [60, 2.5], [65, 3], [70, 3.5], [80, 5],
    [85, 6], [95, 7]
], dtype=float)

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM model
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train_scaled, y_train)
print("\nSVM Model Trained Successfully.")

# Predict and evaluate
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM RESULTS")
print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()

# Predict for a new student
new_student = np.array([[75, 4]], dtype=float)
new_student_scaled = scaler.transform(new_student)
prediction = svm_model.predict(new_student_scaled)[0]
print("New student prediction:", "Pass" if prediction == 1 else "Fail")

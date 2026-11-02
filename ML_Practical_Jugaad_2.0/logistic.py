# LOGISTIC REGRESSION
# Case study: Predict whether a student passes based on study hours.
# Target: 0 = Fail, 1 = Pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 10 rows of assumed data (one feature: study hours)
X = np.array([[1], [1.5], [2], [2.5], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Display the sigmoid function, as in the original practical code
x_curve = np.linspace(-10, 10, 100)
y_curve = 1 / (1 + np.exp(-x_curve))
plt.plot(x_curve, y_curve)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print("\nLogistic Regression Model Trained Successfully!")

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Actual values:", y_test)
print("Predicted values:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Predict for a new student
new_hours = np.array([[3.5]])
prediction = model.predict(new_hours)[0]
print("Prediction for 3.5 study hours:", "Pass" if prediction == 1 else "Fail")

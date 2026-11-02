# LINEAR REGRESSION
# Case study: Predict a student's exam score from study hours.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 10 rows of assumed data
study_hours = np.array([1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8], dtype=float)
exam_score = np.array([35, 40, 42, 50, 55, 60, 65, 70, 78, 85], dtype=float)

# Explore the data
print("Study Hours:", study_hours)
print("Exam Scores:", exam_score)

# Train the simple linear regression model
result = linregress(study_hours, exam_score)

print("\nLINEAR REGRESSION RESULTS")
print("Slope:", result.slope)
print("Intercept:", result.intercept)
print("R-value:", result.rvalue)
print("P-value:", result.pvalue)
print("Standard Error:", result.stderr)

# Predict values
y_pred = result.slope * study_hours + result.intercept
print("\nPredicted scores:", np.round(y_pred, 2))

# Evaluation
mae = mean_absolute_error(exam_score, y_pred)
mse = mean_squared_error(exam_score, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(exam_score, y_pred)

print("\nMODEL EVALUATION")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Plot data and regression line
plt.scatter(study_hours, exam_score, label="Actual data")
plt.plot(study_hours, y_pred, color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.legend()
plt.grid(True)
plt.show()

# Predict for a new input
new_hours = 6.5
new_prediction = result.slope * new_hours + result.intercept
print("\nPredicted exam score for", new_hours, "study hours:", round(new_prediction, 2))

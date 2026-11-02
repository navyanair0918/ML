# MULTIVARIATE LINEAR REGRESSION
# Case study: Predict house price using area, bedrooms and house age.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 10 rows of assumed data
# Features: area in sq.ft, number of bedrooms, age in years
X = np.array([
    [800, 2, 10],
    [900, 2, 8],
    [1000, 2, 6],
    [1100, 3, 10],
    [1200, 3, 7],
    [1300, 3, 5],
    [1400, 3, 4],
    [1500, 4, 6],
    [1600, 4, 3],
    [1800, 4, 2]
], dtype=float)

# Target: house price in lakh rupees
y = np.array([35, 40, 45, 48, 55, 62, 68, 72, 80, 95], dtype=float)

print("Features (Area, Bedrooms, Age):\n", X)
print("Target (Price in lakhs):\n", y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed successfully.")

# Predict
y_pred = model.predict(X_test)
print("Actual prices:", y_test)
print("Predicted prices:", np.round(y_pred, 2))

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMODEL EVALUATION")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))

# Coefficients and intercept
feature_names = ["Area", "Bedrooms", "Age"]
print("\nRegression Coefficients:")
for name, coefficient in zip(feature_names, model.coef_):
    print(name, ":", round(coefficient, 4))
print("Intercept:", round(model.intercept_, 4))

# Actual vs predicted graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price (lakhs)")
plt.ylabel("Predicted House Price (lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Predict price for a new house: 1250 sq.ft, 3 bedrooms, 5 years old
new_house = np.array([[1250, 3, 5]], dtype=float)
print("\nPredicted price for new house (lakhs):",
      round(model.predict(new_house)[0], 2))

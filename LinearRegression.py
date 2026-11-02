
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

# Create dataset
data = {
    'Area': [500, 600, 700, 800, 900,
             1000, 1100, 1200, 1300, 1400],
    'Price': [25, 30, 35, 40, 45,
              50, 55, 60, 65, 70]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Separate input and output
X = df[['Area']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate regression model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
sample_count = len(y_test)
feature_count = X_test.shape[1]

print("\nRegression evaluation metrics (test data):")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):",
      f"{mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
print("Median Absolute Error:", median_absolute_error(y_test, y_pred))
print("Maximum Error:", max_error(y_test, y_pred))
print("R2 Score:", r2)
print("Explained Variance Score:",
      explained_variance_score(y_test, y_pred))

if sample_count > feature_count + 1:
    adjusted_r2 = 1 - (1 - r2) * (sample_count - 1) / (
        sample_count - feature_count - 1
    )
    print("Adjusted R2 Score:", adjusted_r2)
else:
    print("Adjusted R2 Score: undefined (not enough test samples)")

# Take input from user
area = float(input("\nEnter house area in sq.ft: "))

# Predict house price
new_data = pd.DataFrame({'Area': [area]})
prediction = model.predict(new_data)

print("Predicted House Price:", round(prediction[0], 2), "Lakhs")

# Plot graph
plt.scatter(X["Area"], y, color="blue", label="Actual Data")
plt.plot(X["Area"], model.predict(X), color="red", label="Regression Line")
plt.scatter(
    area,
    prediction[0],
    color="green",
    marker="*",
    s=180,
    label="Your prediction",
)

plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (Lakhs)")
plt.title("Linear Regression: House Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

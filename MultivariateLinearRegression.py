import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Edit these 10 sample rows for your problem.
data = {
    "Area": [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
    "Bedrooms": [1, 2, 2, 3, 3, 4, 3, 4, 4, 5],
    "Price": [25, 34, 38, 47, 51, 60, 59, 68, 72, 81],
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

X = df[["Area", "Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nRegression metrics (test data):")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

area = float(input("\nEnter house area in sq.ft: "))
bedrooms = float(input("Enter number of bedrooms: "))
new_house = pd.DataFrame({"Area": [area], "Bedrooms": [bedrooms]})
prediction = model.predict(new_house)[0]
print("Predicted House Price:", round(prediction, 2), "Lakhs")

plt.scatter(y_test, y_pred, color="blue", label="Test predictions")
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())
plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="red",
    linestyle="--",
    label="Perfect prediction",
)
plt.xlabel("Actual price (Lakhs)")
plt.ylabel("Predicted price (Lakhs)")
plt.title("Multivariate Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Edit these 10 sample rows for your problem.
data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 65, 60, 70, 75, 80, 85, 90, 95],
    "PreviousScore": [40, 45, 50, 55, 60, 65, 70, 75, 85, 90],
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

model = PCA(n_components=2)
X_pca = model.fit_transform(X_scaled)
print("\nExplained variance ratio:", model.explained_variance_ratio_)
print("Total variance retained:",
      f"{model.explained_variance_ratio_.sum() * 100:.2f}%")
print("\nRows transformed to two principal components:")
print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).round(3))

study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
new_student = pd.DataFrame(
    {
        "StudyHours": [study_hours],
        "Attendance": [attendance],
        "PreviousScore": [previous_score],
    }
)
new_scaled = scaler.transform(new_student)
new_pca = model.transform(new_scaled)
print("Transformed input (PC1, PC2):", new_pca[0].round(3))

plt.scatter(X_pca[:, 0], X_pca[:, 1], color="blue", label="Dataset rows")
for row_number, (pc1, pc2) in enumerate(X_pca, start=1):
    plt.annotate(str(row_number), (pc1, pc2), xytext=(4, 4),
                 textcoords="offset points")
plt.scatter(
    new_pca[0, 0],
    new_pca[0, 1],
    color="red",
    marker="*",
    s=180,
    label="Your input",
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Data Reduced to Two Dimensions")
plt.legend()
plt.grid(True)
plt.show()

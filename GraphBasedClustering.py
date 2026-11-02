import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Spectral clustering is a graph-based clustering algorithm.
# Edit these 10 sample rows for your problem.
data = {
    "AnnualIncome": [20, 22, 25, 28, 30, 65, 70, 75, 80, 85],
    "SpendingScore": [25, 30, 20, 35, 28, 70, 75, 65, 80, 72],
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

model = SpectralClustering(
    n_clusters=2,
    affinity="rbf",
    gamma=1.0,
    assign_labels="kmeans",
    random_state=42,
)
cluster_labels = model.fit_predict(X_scaled)
df["Cluster"] = cluster_labels

print("\nCluster assignments (cluster numbers are arbitrary):")
print(df)
print("Silhouette score:", silhouette_score(X_scaled, cluster_labels))

income = float(input("\nEnter annual income (in thousands): "))
spending = float(input("Enter spending score (1-100): "))
new_customer = pd.DataFrame(
    {"AnnualIncome": [income], "SpendingScore": [spending]}
)

# SpectralClustering has no predict method; assign new points to the
# nearest cluster center in scaled feature space as a simple approximation.
scaled_centers = pd.DataFrame(X_scaled).groupby(cluster_labels).mean().to_numpy()
new_scaled = scaler.transform(new_customer)
distances = ((scaled_centers - new_scaled[0]) ** 2).sum(axis=1)
nearest_cluster = distances.argmin()
print(
    "Approximate nearest cluster:",
    nearest_cluster,
    "(SpectralClustering does not directly predict new rows)",
)

for cluster in sorted(df["Cluster"].unique()):
    members = df[df["Cluster"] == cluster]
    plt.scatter(
        members["AnnualIncome"],
        members["SpendingScore"],
        label=f"Cluster {cluster}",
    )
plt.scatter(
    income,
    spending,
    color="black",
    marker="*",
    s=180,
    label="Your input",
)
plt.xlabel("Annual income (thousands)")
plt.ylabel("Spending score")
plt.title("Graph-Based Clustering: Spectral Clustering")
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from itertools import combinations

# Step 1: Load the dataset
try:
    data = pd.read_csv("WINE.txt", delim_whitespace=True, header=None)  # Use whitespace delimiter
except FileNotFoundError:
    print("Error: WINE.txt file not found. Ensure the file is in the correct directory.")
    exit()

# Step 2: Prepare the features (columns 2-14)
features = data.iloc[:, 1:]  # Select columns 2-14
features = features.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, replace errors with NaN
if features.isnull().values.any():
    print("Warning: Data contains missing values. Dropping rows with NaN values.")
    features = features.dropna()

# Step 3: Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 4: Manual PCA Calculation
# Compute the covariance matrix
cov_matrix = np.cov(features_scaled.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors by descending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project data onto principal components
principal_components = features_scaled @ eigenvectors

# Save the PCA coefficient matrix
np.savetxt("manual_pca_coefficients.txt", eigenvectors)

# Step 5: Implement K-means algorithm
def k_means_clustering(X, n_clusters, max_iter=300):
    rng = np.random.default_rng()
    centroids = X[rng.choice(X.shape[0], n_clusters, replace=False)]  # Random initialization
    for i in range(max_iter):
        clusters = pairwise_distances_argmin(X, centroids)  # Assign points to the nearest centroid
        new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Run K-means clustering on the top 2 principal components
clusters_2, centroids_2 = k_means_clustering(principal_components[:, :2], 3)

# Visualize the objects using the top 2 principal components with clusters
plt.figure(figsize=(8, 6))
for cluster in range(3):  # 3 clusters
    cluster_data = principal_components[clusters_2 == cluster, :2]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster + 1}", alpha=0.7)
plt.scatter(centroids_2[:, 0], centroids_2[:, 1], color="black", marker="x", s=200, label="Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA (Manual): Clusters with First Two Principal Components")
plt.legend()
plt.savefig("PCA_clusters_2_components.png")
plt.show()

# Function to visualize higher principal components with clusters
def visualize_higher_pcs_with_clusters(principal_components, clusters, num_pcs, filename):
    comb = list(combinations(range(num_pcs), 2))  # Generate all pairwise combinations of PCs
    fig, axes = plt.subplots(len(comb), 1, figsize=(10, 6 * len(comb)))
    if len(comb) == 1:
        axes = [axes]  # Wrap into a list for consistency
    for i, (x_idx, y_idx) in enumerate(comb):
        for cluster in range(3):  # 3 clusters
            cluster_data = principal_components[clusters == cluster][:, [x_idx, y_idx]]
            axes[i].scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster + 1}", alpha=0.7)
        axes[i].set_xlabel(f"Principal Component {x_idx + 1}")
        axes[i].set_ylabel(f"Principal Component {y_idx + 1}")
        axes[i].set_title(f"Scatter Plot: PC{x_idx + 1} vs PC{y_idx + 1}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Run K-means clustering on the top 5 and 8 principal components
clusters_5, _ = k_means_clustering(principal_components[:, :5], 3)
clusters_8, _ = k_means_clustering(principal_components[:, :8], 3)

# Visualize the first 5 principal components
visualize_higher_pcs_with_clusters(principal_components, clusters_5, 5, "PCA_clusters_5_components.png")

# Visualize the first 8 principal components
visualize_higher_pcs_with_clusters(principal_components, clusters_8, 8, "PCA_clusters_8_components.png")

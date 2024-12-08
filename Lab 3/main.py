import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

# Load the dataset
def import_data():
    data = pd.read_csv("WINE.txt", sep='\s+', header=None) # Use regex for whitespace separator
    features = data.iloc[:, 1:]  # Select columns 2-14
    return features

def calculate_PCA(features):
    cov_matrix = np.cov(features.T)  # Compute the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Compute eigenvalues and eigenvectors

    # Sort eigenvalues and eigenvectors by descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    np.savetxt("manual_pca_coefficients.txt", eigenvectors)  # Save the PCA coefficient matrix

    # Project data onto principal components
    principal_components = features_scaled @ eigenvectors

    # Calculate variance ratio
    total_variance = np.sum(eigenvalues)
    variance_ratio = eigenvalues / total_variance
    plot_variance(variance_ratio) # Plot the variance

    return principal_components

# K-means algorithm
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

# Visualize variance ratio of each PC
def plot_variance(variance_ratio):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio, color='orange', alpha=0.7)
    plt.title("Variance Ratio of Principal Components", fontsize=14)
    plt.xlabel("Principal Components", fontsize=12)
    plt.ylabel("Variance Ratio", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, len(variance_ratio) + 1))
    plt.tight_layout()
    plt.savefig("variance_ratio.png")
    plt.show()

# Visualize the objects using the top 2 principal components with clusters
def plot_PC_2(principal_components, clusters, centroids):
    plt.figure(figsize=(8, 6))
    for cluster in range(3):  # 3 clusters
        cluster_data = principal_components[clusters == cluster, :2]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster + 1}", alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=200, label="Centroids")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: Clusters with First Two Principal Components")
    plt.legend()
    plt.savefig("PCA_clusters_2_components.png")
    plt.show()

def store_data(clusters_2, clusters_5, clusters_8, centroids_2, centroids_5, centroids_8):
    count_2 = np.bincount(clusters_2)
    count_5 = np.bincount(clusters_5)
    count_8 = np.bincount(clusters_8)

    # Optionally save the counts to a text file
    with open('Cluster_count', "w") as f:
        f.write("Cluster Counts for Each Principal Component\n\n")
        f.write("PC 2 Components:\n")
        f.write(f"Cluster 1: {count_2[0]}\n")
        f.write(f"Cluster 2: {count_2[1]}\n")
        f.write(f"Cluster 3: {count_2[2]}\n")
        f.write("Centroids:\n")
        for i, centroid in enumerate(centroids_2):
            f.write(f"Centroid {i + 1}: {centroid}\n")

        f.write("\nPC 5 Components:\n")
        f.write(f"Cluster 1: {count_5[0]}\n")
        f.write(f"Cluster 2: {count_5[1]}\n")
        f.write(f"Cluster 3: {count_5[2]}\n")
        f.write("Centroids:\n")
        for i, centroid in enumerate(centroids_5):
            f.write(f"Centroid {i + 1}: {centroid}\n")

        f.write("\nPC 8 Components:\n")
        f.write(f"Cluster 1: {count_8[0]}\n")
        f.write(f"Cluster 2: {count_8[1]}\n")
        f.write(f"Cluster 3: {count_8[2]}\n")
        f.write("Centroids:\n")
        for i, centroid in enumerate(centroids_8):
            f.write(f"Centroid {i + 1}: {centroid}\n")

#------------------------------------------------------------------------------#

# Import the data
features = import_data()

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA Calculation
principal_components = calculate_PCA(features_scaled)

# Run K-means clustering on the top 2 principal components
clusters_2, centroids_2 = k_means_clustering(principal_components[:, :2], 3)
clusters_5, centroids_5 = k_means_clustering(principal_components[:, :5], 3)
clusters_8, centroids_8 = k_means_clustering(principal_components[:, :8], 3)

# Visualize the objects using the top 2 PC
plot_PC_2(principal_components, clusters_2, centroids_2)

# Save the cluster counts and centroids
store_data(clusters_2, clusters_5, clusters_8, centroids_2, centroids_5, centroids_8)

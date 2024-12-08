def k_means_clustering(X, n_clusters, max_iter=300):
    # Random initialization of centroids
    rng = np.random.default_rng()
    centroids = X[
        rng.choice(X.shape[0], n_clusters, replace=False)]  # Randomly select n_clusters points as initial centroids

    for iteration in range(max_iter):
        # Step 1: Assign points to the nearest centroid
        clusters = np.zeros(X.shape[0], dtype=int)  # Array to store cluster assignments
        for i, point in enumerate(X):
            distances = np.linalg.norm(point - centroids, axis=1)  # Compute distance to each centroid
            clusters[i] = np.argmin(distances)  # Assign to the closest centroid

        # Step 2: Recalculate centroids
        new_centroids = np.zeros_like(centroids)  # Array to store new centroids
        for k in range(n_clusters):
            points_in_cluster = X[clusters == k]  # Points assigned to cluster k
            if len(points_in_cluster) > 0:  # Avoid empty cluster
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                # If a cluster is empty, reinitialize its centroid randomly
                new_centroids[k] = X[rng.choice(X.shape[0])]

        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids, atol=1e-6):  # Allow for slight numerical differences
            break

        centroids = new_centroids  # Update centroids for the next iteration

    return clusters, centroids

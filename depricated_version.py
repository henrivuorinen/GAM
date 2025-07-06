import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# --- 1. Data Generation (Modified for Regression) ---
def generate_regression_data(n_samples=400, type='moons_blobs'):
    """Generates synthetic 2D regression data with a target variable y."""
    if type == 'moons_blobs':
        X_moons, _ = make_moons(n_samples=n_samples // 2, noise=0.05, random_state=42)
        X_blobs, _ = make_blobs(n_samples=n_samples // 2, centers=2, cluster_std=0.5, random_state=42)
        X_blobs[:, 0] += 2
        X_blobs[:, 1] -= 0.5
        X = np.vstack([X_moons, X_blobs])

        # Define a non-linear y for this complex X
        y = np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 3) + (X[:, 0] * X[:, 1]) * 0.5
        y += np.random.normal(0, 0.1, y.shape) # Add some noise

    elif type == 'linear_like':
        X = np.random.rand(n_samples, 2) * 5 # Random points in a square
        y = 3 * X[:, 0] - 2 * X[:, 1] + 10 # Linear relationship
        y += np.random.normal(0, 0.5, y.shape) # Add some noise

    elif type == 'parabola':
        X = np.random.rand(n_samples, 2) * 10 - 5 # Points from -5 to 5
        y = 0.5 * X[:, 0]**2 + 2 * X[:, 1] # Parabolic in X0, linear in X1
        y += np.random.normal(0, 1, y.shape) # Add noise

    return X, y

# --- Gravity-Based Clustering Function (from Phase 1) ---
def gravity_clustering(X, gamma, n_iterations):
    """
    Performs gravity-based clustering by iteratively moving points.
    Returns final positions and original indices for each converged point.
    """
    current_positions = np.copy(X)
    N = X.shape[0]

    for k_iter in range(n_iterations):
        distances_sq = euclidean_distances(current_positions, current_positions, squared=True)
        influence_matrix = np.exp(-gamma * distances_sq)
        np.fill_diagonal(influence_matrix, 0)

        sum_influences = np.sum(influence_matrix, axis=1, keepdims=True)
        sum_influences[sum_influences == 0] = 1e-9 # Avoid division by zero

        new_positions = np.dot(influence_matrix, current_positions) / sum_influences
        current_positions = new_positions

    return current_positions # Return only final positions

# --- Phase 2: Adaptive Local Modeling and Blending ---
class GravityAdaptiveModel:
    def __init__(self, gamma_clustering, n_iterations_clustering,
                 blending_alpha, local_model_degree=1,
                 clustering_distance_threshold=0.2):

        self.gamma_clustering = gamma_clustering
        self.n_iterations_clustering = n_iterations_clustering
        self.blending_alpha = blending_alpha # Blending bandwidth (similar to gamma, but for blending)
        self.local_model_degree = local_model_degree # Max degree for polynomial features
        self.clustering_distance_threshold = clustering_distance_threshold

        self.cluster_centers = None
        self.local_models = []
        self.cluster_data_indices = [] # Stores original indices for each cluster

    def fit(self, X, y):
        # Phase 1: Gravity-based clustering
        print(f"Phase 1: Running gravity clustering for {X.shape[0]} points...")
        final_grav_positions = gravity_clustering(X, self.gamma_clustering, self.n_iterations_clustering)

        # Identify clusters from converged positions
        agg_clustering = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=self.clustering_distance_threshold,
                                                 linkage='single')
        cluster_labels = agg_clustering.fit_predict(final_grav_positions)

        self.cluster_centers = []
        self.local_models = []
        self.cluster_data_indices = []

        unique_labels = np.unique(cluster_labels)
        print(f"Phase 1 complete. Found {len(unique_labels)} clusters.")

        # Phase 2: Train local models for each cluster
        print("Phase 2: Training local models...")
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) == 0: # Should not happen if unique_labels are from np.where
                continue

            self.cluster_data_indices.append(cluster_indices)

            # Cluster center: Mean of original data points in this cluster
            # Or mean of converged gravity positions - using original is often better for model fitting.
            cluster_center = np.mean(X[cluster_indices], axis=0)
            self.cluster_centers.append(cluster_center)

            # Train local model
            X_cluster = X[cluster_indices]
            y_cluster = y[cluster_indices]

            # Adaptive complexity (simple version): Fixed polynomial degree for now.
            # Could be adapted based on cluster size/variance/etc.
            model = make_pipeline(PolynomialFeatures(degree=self.local_model_degree),
                                  LinearRegression())

            try:
                model.fit(X_cluster, y_cluster)
                self.local_models.append(model)
            except ValueError as e:
                print(f"Warning: Could not fit model for cluster {label} with {len(X_cluster)} points. {e}")
                # Append a dummy model that predicts the mean, or handle gracefully
                self.local_models.append(DummyMeanPredictor(y_cluster)) # Custom simple predictor

        print("Phase 2 complete. All local models trained.")

    def predict(self, X_new):
        if self.cluster_centers is None:
            raise RuntimeError("Model must be fitted before prediction.")

        N_new = X_new.shape[0]
        predictions = np.zeros(N_new)

        for i in range(N_new):
            point = X_new[i]

            # Calculate blending weights (proximity to each cluster center)
            # Use self.blending_alpha as the gamma for blending
            distances_to_centers_sq = np.sum((self.cluster_centers - point)**2, axis=1)
            blending_weights = np.exp(-self.blending_alpha * distances_to_centers_sq)

            # Normalize blending weights
            sum_blending_weights = np.sum(blending_weights)
            if sum_blending_weights == 0: # All centers too far, or alpha too high
                predictions[i] = np.mean([model.predict(point.reshape(1, -1))[0] for model in self.local_models])
                continue # Fallback to average of all models

            normalized_blending_weights = blending_weights / sum_blending_weights

            # Blend predictions from local models
            blended_prediction = 0
            for j, model in enumerate(self.local_models):
                # Reshape point for model.predict (expects 2D array)
                model_prediction = model.predict(point.reshape(1, -1))[0]
                blended_prediction += normalized_blending_weights[j] * model_prediction

            predictions[i] = blended_prediction

        return predictions

# Custom Dummy Predictor for very small clusters
class DummyMeanPredictor:
    def __init__(self, y_data):
        self.mean_val = np.mean(y_data)

    def predict(self, X_new):
        return np.full(X_new.shape[0], self.mean_val)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Experiment with different data types ---
    DATA_TYPE = 'moons_blobs' # Options: 'moons_blobs', 'linear_like', 'parabola'
    X_original, y_original = generate_regression_data(n_samples=400, type=DATA_TYPE)

    # --- Hyperparameters for the GravityAdaptiveModel ---
    GRAV_GAMMA = 10.0            # Gamma for gravity clustering (Phase 1)
    GRAV_ITERATIONS = 5          # Number of gravity iterations (Phase 1)
    CLUSTERING_DIST_THRESHOLD = 0.2 # Distance to group converged points into clusters (Phase 1)

    BLENDING_ALPHA = 5.0         # Alpha for blending (how local is the blending influence)
    LOCAL_MODEL_DEGREE = 2       # Polynomial degree for local models (Phase 2)

    print(f"Running Gravity Adaptive Model on '{DATA_TYPE}' data...")
    model = GravityAdaptiveModel(
        gamma_clustering=GRAV_GAMMA,
        n_iterations_clustering=GRAV_ITERATIONS,
        blending_alpha=BLENDING_ALPHA,
        local_model_degree=LOCAL_MODEL_DEGREE,
        clustering_distance_threshold=CLUSTERING_DIST_THRESHOLD
    )

    model.fit(X_original, y_original)

    # --- Generate a grid for plotting the prediction surface ---
    x_min, x_max = X_original[:, 0].min() - 0.5, X_original[:, 0].max() + 0.5
    y_min, y_max = X_original[:, 1].min() - 0.5, X_original[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict over the grid
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Evaluate the model on the training data
    y_pred_train = model.predict(X_original)
    rmse = np.sqrt(mean_squared_error(y_original, y_pred_train))
    print(f"\nRMSE on training data: {rmse:.4f}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 8))

    # Plot the prediction surface
    plt.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Predicted y')

    # Plot original data points, colored by their actual y-value
    scatter = plt.scatter(X_original[:, 0], X_original[:, 1], c=y_original,
                          cmap='cividis', edgecolors='k', s=50, zorder=2)
    plt.colorbar(scatter, label='Actual y')

    # Plot cluster centers (optional)
    if model.cluster_centers is not None:
        plt.scatter(np.array(model.cluster_centers)[:, 0],
                    np.array(model.cluster_centers)[:, 1],
                    marker='X', s=200, color='red', label='Cluster Centers', zorder=3)

    plt.title(f'Gravity Adaptive Model Prediction Surface ({DATA_TYPE} data)\nRMSE: {rmse:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Optional: Visualize cluster assignments on original data ---
    # To do this, we need to get the cluster labels from within the model
    # (they are currently not stored explicitly as an attribute).
    # We can run the clustering again quickly for visualization.

    # This block is for visualization of clusters on original points only
    final_grav_pos_for_viz = gravity_clustering(X_original, GRAV_GAMMA, GRAV_ITERATIONS)
    agg_clustering_viz = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=CLUSTERING_DIST_THRESHOLD,
                                                 linkage='single')
    cluster_labels_viz = agg_clustering_viz.fit_predict(final_grav_pos_for_viz)

    plt.figure(figsize=(8, 6))
    scatter_clusters = plt.scatter(X_original[:, 0], X_original[:, 1],
                                   c=cluster_labels_viz, cmap='tab10', s=50, edgecolors='k')
    plt.title(f'Original Data with Detected Clusters ({DATA_TYPE})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter_clusters, label='Cluster Label')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

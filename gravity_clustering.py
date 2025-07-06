import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def gravity_clustering(X, gamma, n_iterations):
    """
    Performs gravity-based clustering by iteratively moving points.
    Returns final positions of the points after iterations.
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

    return current_positions

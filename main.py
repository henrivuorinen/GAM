import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from data_utils import (
    generate_regression_data,
    generate_classification_data,
    load_and_preprocess_titanic_data,
)
from gravity_adaptive_model import GravityAdaptiveModel
from gravity_clustering import gravity_clustering
import argparse

# Global seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Configuration ---
# --- Command-line Argument Parsing ---
parser = argparse.ArgumentParser(description='Run Gravity Adaptive Model experiments.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['regression', 'classification'],
                    help='Type of machine learning task (regression or classification).')
parser.add_argument('--data_source', type=str, default='circles', # Default to a visual one
                    choices=['moons_blobs', 'circles', 'blobs_non_linear', 'titanic', 'parabola', 'linear_like'],
                    help='Source of the data for the experiment.')
parser.add_argument('--n_samples', type=int, default=400,
                    help='Number of samples for synthetic datasets.')
parser.add_argument('--gamma_clustering', type=float, default=0.8,
                    help='Gamma parameter for gravity clustering (Phase 1).')
parser.add_argument('--n_iterations_clustering', type=int, default=5,
                    help='Number of iterations for gravity convergence.')
parser.add_argument('--blending_alpha', type=float, default=1.0,
                    help='Alpha parameter for blending (how local blending influence is).')
parser.add_argument('--local_model_degree', type=int, default=2,
                    help='Polynomial degree for local models (Phase 2).')
parser.add_argument('--clustering_distance_threshold', type=float, default=0.5,
                    help='Distance threshold for Agglomerative Clustering.')
parser.add_argument('--run_grid_search', action='store_true',
                    help='If set, performs GridSearchCV for hyperparameter optimization.')

args = parser.parse_args()

# Map parsed arguments to EXPERIMENT_CONFIG format
EXPERIMENT_CONFIG = {
    'task_type': args.task_type,
    'data_source': args.data_source,
    'n_samples': args.n_samples,
    'hyperparams': {
        'gamma_clustering': args.gamma_clustering,
        'n_iterations_clustering': args.n_iterations_clustering,
        'blending_alpha': args.blending_alpha,
        'local_model_degree': args.local_model_degree,
        'clustering_distance_threshold': args.clustering_distance_threshold,
    }
}

# --- Data Loading / Generation ---
def load_data(config):
    data_source = config['data_source']
    task_type = config['task_type']
    n_samples = config['n_samples']

    if data_source == 'titanic':
        X_train, X_test, y_train_pd, y_test_pd = load_and_preprocess_titanic_data(filepath='titanic.csv', random_state=RANDOM_SEED)

        y_train = y_train_pd.values
        y_test = y_test_pd.values

        return X_train, X_test, y_train, y_test
    elif task_type == 'regression' and data_source in ['moons_blobs', 'linear_like', 'parabola']:
        X, y = generate_regression_data(n_samples=n_samples, type=data_source)
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    elif task_type == 'classification' and data_source in ['circles', 'blobs_non_linear']:
        X, y = generate_classification_data(n_samples=n_samples, type=data_source)
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    # Add more data loading logic here as needed
    else:
        raise ValueError(f"Unsupported data_source '{data_source}' for task_type '{task_type}'.")

print(f"Loading data from '{EXPERIMENT_CONFIG['data_source']}' for '{EXPERIMENT_CONFIG['task_type']}' task...")
X_train_full, X_test, y_train_full, y_test = load_data(EXPERIMENT_CONFIG)

# For GridSearchCV, train on X_train_full and y_train_full
# For regular runs, these are the X_train/y_train
X_train, y_train = X_train_full, y_train_full

# --- Model Initialization & Training ---
if args.run_grid_search:
    print("\n--- Running GridSearchCV for Hyperparameter Optimization ---")
    # Define the parameter grid to search
    param_grid = {
        'gamma_clustering': [0.5, 0.8, 1.0, 2.0],
        'n_iterations_clustering': [5, 10],
        'blending_alpha': [0.5, 1.0, 2.0],
        'local_model_degree': [1, 2, 3], # Max polynomial degree
        'clustering_distance_threshold': [0.1, 0.3, 0.5, 0.8]
    }

    # Initialize the base model with task_type
    base_model = GravityAdaptiveModel(task_type=EXPERIMENT_CONFIG['task_type'])

    # Define scoring based on task type
    if EXPERIMENT_CONFIG['task_type'] == 'regression':
        scorer = 'neg_mean_squared_error'
    elif EXPERIMENT_CONFIG['task_type'] == 'classification':
        scorer = make_scorer(f1_score, average='weighted')
    # KFold for cross-validation
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scorer,
        verbose=2, # Higher verbose for more output during search
        n_jobs=-1 # Use all available CPU cores
    )

    grid_search.fit(X_train_full, y_train_full) # Fit GridSearchCV on the full training data

    print("\nBest parameters found:", grid_search.best_params_)
    if EXPERIMENT_CONFIG['task_type'] == 'regression':
        print("Best RMSE:", np.sqrt(-grid_search.best_score_))
    else:
        print("Best F1-score (weighted):", grid_search.best_score_)

    model = grid_search.best_estimator_ # Use the best model found by GridSearchCV

else: # run without GridSearchCV
    print(f"Running Gravity Adaptive Model...")
    model = GravityAdaptiveModel(
        task_type=EXPERIMENT_CONFIG['task_type'],
        **EXPERIMENT_CONFIG['hyperparams']
    )
    model.fit(X_train, y_train)

# Fit the model ONLY on the training data
model.fit(X_train, y_train)

# --- Evaluate the model ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n--- Evaluation for {EXPERIMENT_CONFIG['data_source']} ({EXPERIMENT_CONFIG['task_type']}) ---")
if EXPERIMENT_CONFIG['task_type'] == 'regression':
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"RMSE on training data: {rmse_train:.4f}")
    print(f"RMSE on test data: {rmse_test:.4f}")
elif EXPERIMENT_CONFIG['task_type'] == 'classification':
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy on training data: {acc_train:.4f}")
    print(f"Accuracy on test data: {acc_test:.4f}")
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_pred_test))

# --- Plotting Results (only for 2D synthetic data at the moment) ---
# Check if plotting is applicable (only for 2D synthetic data)
is_plot_applicable = (
        (EXPERIMENT_CONFIG['task_type'] == 'regression' and EXPERIMENT_CONFIG['data_source'] in ['moons_blobs', 'linear_like', 'parabola']) or
        (EXPERIMENT_CONFIG['task_type'] == 'classification' and EXPERIMENT_CONFIG['data_source'] in ['circles', 'blobs_non_linear'])
)

if is_plot_applicable and X_train.shape[1] == 2:
    # Re-generate the full synthetic dataset for plotting grid if needed
    if EXPERIMENT_CONFIG['task_type'] == 'regression':
        X_plot, y_plot = generate_regression_data(n_samples=EXPERIMENT_CONFIG['n_samples'], type=EXPERIMENT_CONFIG['data_source'])
    else: # classification
        X_plot, y_plot = generate_classification_data(n_samples=EXPERIMENT_CONFIG['n_samples'], type=EXPERIMENT_CONFIG['data_source'])

    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    plt.figure(figsize=(12, 8))

    if EXPERIMENT_CONFIG['task_type'] == 'regression':
        plt.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Predicted y')
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot,
                              cmap='cividis', edgecolors='k', s=50, zorder=2)
        plt.colorbar(scatter, label='Actual y')
    elif EXPERIMENT_CONFIG['task_type'] == 'classification':
        plt.contourf(xx, yy, Z, levels=np.arange(Z.max() + 2) - 0.5, cmap='viridis', alpha=0.8)
        plt.colorbar(ticks=np.unique(Z), label='Predicted Class')
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot,
                              cmap='cividis', edgecolors='k', s=50, zorder=2)
        plt.colorbar(scatter, ticks=np.unique(y_plot), label='Actual Class')

    if model.cluster_centers is not None: #
        cluster_centers_np = np.array(model.cluster_centers)

        # Use cluster_centers_np for checks and plotting
        if cluster_centers_np.shape[0] > 0 and cluster_centers_np.shape[1] == 2:
            plt.scatter(cluster_centers_np[:, 0],
                        cluster_centers_np[:, 1],
                        marker='X', s=200, color='red', label='Cluster Centers', zorder=3)

    plt.title(f'Gravity Adaptive Model Prediction Surface ({EXPERIMENT_CONFIG["data_source"]} data)\n'
              f'{EXPERIMENT_CONFIG["task_type"].capitalize()} Test Accuracy: {acc_test:.4f}' if EXPERIMENT_CONFIG['task_type'] == 'classification' else \
                  f'RMSE Test: {rmse_test:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Visualize cluster assignments on original data ---
    plt.figure(figsize=(8, 6))
    final_grav_pos_for_viz = gravity_clustering(X_plot, EXPERIMENT_CONFIG['hyperparams']['gamma_clustering'], EXPERIMENT_CONFIG['hyperparams']['n_iterations_clustering'])
    agg_clustering_viz = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=EXPERIMENT_CONFIG['hyperparams']['clustering_distance_threshold'],
                                                 linkage='single')
    cluster_labels_viz = agg_clustering_viz.fit_predict(final_grav_pos_for_viz)

    scatter_clusters = plt.scatter(X_plot[:, 0], X_plot[:, 1],
                                   c=cluster_labels_viz, cmap='tab10', s=50, edgecolors='k')
    plt.title(f'Original Data with Detected Clusters ({EXPERIMENT_CONFIG["data_source"]})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter_clusters, label='Cluster Label')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from gravity_clustering import gravity_clustering

# Global seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Dummy Predictors for small clusters ---
class DummyMeanPredictor: # For regression
    def __init__(self, y_data):
        self.mean_val = np.mean(y_data)
    def predict(self, X_new):
        return np.full(X_new.shape[0], self.mean_val)

class DummyMajorityClassPredictor: # For classification
    def __init__(self, y_data):
        if len(y_data) > 0:
            unique_classes, counts = np.unique(y_data, return_counts=True)
            self.majority_class = unique_classes[np.argmax(counts)]
            self.num_classes = len(unique_classes) # For predict_proba
            self.class_probabilities = np.zeros(self.num_classes)
            self.class_probabilities[np.argmax(counts)] = 1.0 # 100% for majority class
        else:
            self.majority_class = 0 # Default if no data
            self.num_classes = 2 # Assume binary for default
            self.class_probabilities = np.array([0.5, 0.5]) # Assume 50/50 if no data

    def predict(self, X_new):
        return np.full(X_new.shape[0], self.majority_class)

    def predict_proba(self, X_new):
        # Returns probabilities for each class
        return np.tile(self.class_probabilities, (X_new.shape[0], 1))


# --- GravityAdaptiveModel Class ---
class GravityAdaptiveModel:
    def __init__(self, gamma_clustering, n_iterations_clustering,
                 blending_alpha, local_model_degree=1,
                 clustering_distance_threshold=0.2,
                 task_type='regression'):

        self.gamma_clustering = gamma_clustering
        self.n_iterations_clustering = n_iterations_clustering
        self.blending_alpha = blending_alpha
        self.local_model_degree = local_model_degree
        self.clustering_distance_threshold = clustering_distance_threshold
        self.task_type = task_type # Store task type

        self.cluster_centers = None
        self.local_models = []
        self.cluster_data_indices = []
        self.unique_classes = None # To store unique classes for classification

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

        if self.task_type == 'classification':
            self.unique_classes = np.unique(y) # Store unique classes

        # Phase 2: Train local models for each cluster
        print("Phase 2: Training local models...")
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            if len(cluster_indices) == 0:
                continue

            self.cluster_data_indices.append(cluster_indices)
            cluster_center = np.mean(X[cluster_indices], axis=0)
            self.cluster_centers.append(cluster_center)

            X_cluster = X[cluster_indices]
            y_cluster = y[cluster_indices]

            # Determine if a dummy model is needed
            use_dummy = False
            if self.task_type == 'regression':
                if len(y_cluster) == 0: # No data in cluster
                    use_dummy = True
            elif self.task_type == 'classification':
                # Too few unique classes or too few samples to train Logistic Regression
                if len(np.unique(y_cluster)) < 2 or len(y_cluster) < 2:
                    use_dummy = True

            if use_dummy:
                if self.task_type == 'regression':
                    local_model = DummyMeanPredictor(y_cluster)
                else:
                    local_model = DummyMajorityClassPredictor(y_cluster)
                self.local_models.append(local_model)
            else:
                # Actual sklearn model pipeline
                if self.task_type == 'regression':
                    local_model = make_pipeline(PolynomialFeatures(degree=self.local_model_degree), Ridge(alpha=1.0))
                elif self.task_type == 'classification':
                    local_model = make_pipeline(PolynomialFeatures(degree=self.local_model_degree),
                                                LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000, random_state=RANDOM_SEED,  class_weight='balanced'))

                # Try fitting the actual model
                try:
                    local_model.fit(X_cluster, y_cluster)
                    self.local_models.append(local_model)
                except Exception as e:
                    print(f"Warning: Model fit failed for cluster {label} with {len(X_cluster)} points. Error: {e}. Using dummy predictor.")
                    # Fallback to dummy if fit fails
                    if self.task_type == 'regression':
                        self.local_models.append(DummyMeanPredictor(y_cluster))
                    else:
                        self.local_models.append(DummyMajorityClassPredictor(y_cluster))

        print("Phase 2 complete. All local models trained.")

    def predict(self, X_new):
        if self.cluster_centers is None:
            raise RuntimeError("Model must be fitted before prediction.")

        N_new = X_new.shape[0]

        # Determine prediction method based on task_type
        if self.task_type == 'regression':
            predictions = np.zeros(N_new)
        elif self.task_type == 'classification':
            # Initialize array for blended probabilities for each class
            num_classes = len(self.unique_classes) if self.unique_classes is not None else 2 # Default to 2 if not set
            predictions_proba = np.zeros((N_new, num_classes))
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        for i in range(N_new):
            point = X_new[i]
            distances_to_centers_sq = np.sum((self.cluster_centers - point)**2, axis=1)
            blending_weights = np.exp(-self.blending_alpha * distances_to_centers_sq)

            sum_blending_weights = np.sum(blending_weights)
            if sum_blending_weights == 0:
                # Fallback for completely isolated points
                if self.task_type == 'regression':
                    predictions[i] = np.mean([model.predict(point.reshape(1, -1))[0] for model in self.local_models])
                elif self.task_type == 'classification':
                    # Fallback to average probability across all local models
                    avg_proba = np.mean([model.predict_proba(point.reshape(1, -1))[0] for model in self.local_models], axis=0)
                    predictions_proba[i] = avg_proba
                continue

            normalized_blending_weights = blending_weights / sum_blending_weights

            if self.task_type == 'regression':
                blended_prediction = 0
                for j, model in enumerate(self.local_models):
                    model_prediction = model.predict(point.reshape(1, -1))[0]
                    blended_prediction += normalized_blending_weights[j] * model_prediction
                predictions[i] = blended_prediction
            elif self.task_type == 'classification':
                blended_proba = np.zeros(num_classes)
                for j, model in enumerate(self.local_models):
                    # Ensure model.predict_proba returns probabilities for all expected classes
                    model_proba = model.predict_proba(point.reshape(1, -1))[0]

                    # Handle cases where model_proba might have fewer columns than expected
                    # This can happen if a local LR only saw 1 class during training
                    if len(model_proba) != num_classes:
                        # Create a full probability vector, fill with 0s and then assign
                        # This requires careful mapping of classes if not 0,1,2...
                        full_model_proba = np.zeros(num_classes)
                        if hasattr(model, 'classes_'):
                            for k, class_val in enumerate(model.classes_):
                                if class_val in self.unique_classes:
                                    full_model_proba[np.where(self.unique_classes == class_val)[0][0]] = model_proba[k]
                        else: # For dummy predictors or other cases
                            # Default to uniform or based on unique_classes if possible
                            # (Assuming DummyMajorityClassPredictor correctly sets num_classes/probabilities)
                            full_model_proba = model_proba
                        model_proba = full_model_proba


                    blended_proba += normalized_blending_weights[j] * model_proba
                predictions_proba[i] = blended_proba

        if self.task_type == 'regression':
            return predictions
        elif self.task_type == 'classification':
            # Return predicted class labels
            return np.argmax(predictions_proba, axis=1)

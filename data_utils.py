import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def generate_regression_data(n_samples=400, type='moons_blobs'):
    """Generates synthetic 2D regression data with a target variable y."""
    if type == 'moons_blobs':
        X_moons, _ = make_moons(n_samples=n_samples // 2, noise=0.05, random_state=42)
        X_blobs, _ = make_blobs(n_samples=n_samples // 2, centers=2, cluster_std=0.5, random_state=42)
        X_blobs[:, 0] += 2
        X_blobs[:, 1] -= 0.5
        X = np.vstack([X_moons, X_blobs])
        y = np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * 3) + (X[:, 0] * X[:, 1]) * 0.5
        y += np.random.normal(0, 0.1, y.shape) # Add some noise
    elif type == 'linear_like':
        X = np.random.rand(n_samples, 2) * 5
        y = 3 * X[:, 0] - 2 * X[:, 1] + 10
        y += np.random.normal(0, 0.5, y.shape)
    elif type == 'parabola':
        X = np.random.rand(n_samples, 2) * 10 - 5
        y = 0.5 * X[:, 0]**2 + 2 * X[:, 1]
        y += np.random.normal(0, 1, y.shape)
    return X, y

def generate_classification_data(n_samples=400, type='circles'):
    """Generates synthetic 2D classification data with a target variable y."""
    if type == 'circles':
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=42)
    elif type == 'blobs_non_linear':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, random_state=42, class_sep=0.8)
        # Introduce non-linearity to make it harder for a single linear model
        X = X + np.random.normal(0, 0.5, X.shape) # Add some noise/spread
        X[y==0, 0] = X[y==0, 0] * 2 + 1
        X[y==0, 1] = X[y==0, 1] * 2 - 1
        X[y==1, 0] = X[y==1, 0] * 0.5 - 1
        X[y==1, 1] = X[y==1, 1] * 0.5 + 1
    # Add more types if needed
    return X, y


def load_and_preprocess_titanic_data(filepath='titanic.csv', test_size=0.2, random_state=42):
    """
    Loads and preprocesses the Titanic dataset.

    Args:
        filepath (str): Path to the titanic.csv file.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please download it from Kaggle (Titanic competition) "
              "and place it in the same directory as main.py.")
        exit()

    # Define target and features
    y = df['Survived']
    # Drop irrelevant or complex features for now
    X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Preprocessing steps
    # Numerical features to impute and scale
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
    # Categorical features to impute and one-hot encode
    categorical_features = ['Sex', 'Embarked', 'Pclass'] # Pclass as categorical

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Fill missing Age/Fare with mean
        ('scaler', StandardScaler()) # Scale to mean 0, variance 1
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing Embarked with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode Sex, Embarked, Pclass
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full preprocessing pipeline
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply preprocessing and get the transformed X
    X_processed = full_pipeline.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

"""
Model Training Module for Fake News Detection
Contains functions for training and evaluating ML models.
"""

import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float, default=0.2
        Proportion of test set
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
        
    Returns:
    --------
    tuple
        scaler, X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully")
    
    return scaler, X_train_scaled, X_test_scaled


def train_logistic_regression(X_train, y_train, params=None):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    model : LogisticRegression
        Trained model
    """
    if params is None:
        params = {'max_iter': 1000, 'random_state': 42}
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    print("Logistic Regression trained")
    
    return model


def train_random_forest(X_train, y_train, params=None):
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    if params is None:
        params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    print("Random Forest trained")
    
    return model


def train_svm(X_train, y_train, params=None):
    """
    Train SVM model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    model : SVC
        Trained model
    """
    if params is None:
        params = {'kernel': 'linear', 'random_state': 42}
    
    model = SVC(**params)
    model.fit(X_train, y_train)
    
    print("SVM trained")
    
    return model


def train_mlp(X_train, y_train, params=None):
    """
    Train Multi-Layer Perceptron model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    model : MLPClassifier
        Trained model
    """
    if params is None:
        params = {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': 42,
            'early_stopping': True
        }
    
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)
    
    print("MLP Neural Network trained")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str, default="Model"
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    return metrics


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters:
    -----------
    model : estimator
        Base model
    param_grid : dict
        Parameter grid
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    best_model : estimator
        Best model from grid search
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Parameters:
    -----------
    model : estimator
        Trained model
    filepath : str
        Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    model : estimator
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {filepath}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Model training module loaded successfully!")

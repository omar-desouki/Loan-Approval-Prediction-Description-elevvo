import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import shutil
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(
    X, y, test_size=0.2, random_state=42, grid_search=False, balanced=None
):
    """
    Train a logistic regression model with optional grid search and evaluate performance.

    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series
        Target variable
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    grid_search : bool, default=False
        Whether to perform grid search for hyperparameter tuning

    Returns:
    --------
    dict : Dictionary containing model, predictions, and evaluation metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if grid_search:
        # Define parameter grid for grid search
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000, 2000, 3000],
        }

        # Create logistic regression model
        lr = LogisticRegression(random_state=random_state, class_weight=balanced)

        # Perform grid search
        print(f"Performing grid search using {os.cpu_count()} CPU cores...")
        grid_search_cv = GridSearchCV(
            lr, param_grid, cv=5, scoring="f1", n_jobs=os.cpu_count(), verbose=1
        )
        grid_search_cv.fit(X_train, y_train)

        # Use best model
        model = grid_search_cv.best_estimator_
        print(f"Best parameters: {grid_search_cv.best_params_}")
        print(f"Best cross-validation score: {grid_search_cv.best_score_:.4f}")

    else:
        # Train simple logistic regression
        model = LogisticRegression(
            random_state=random_state, max_iter=1000, class_weight=balanced
        )
        model.fit(X_train, y_train)

    # print evaluation results for training -> # i am printing the evaluation of train to compare with the test and see if there is overfitting or not
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION EVALUATION RESULTS (train set)")
    print("=" * 50)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    print(f"Training Accuracy:  {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Training Recall:    {train_recall:.4f}")
    print(f"Training F1-Score:  {train_f1:.4f}")

    # Print evaluation results for t
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation results
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return


def train_random_forest(
    X, y, test_size=0.2, random_state=42, grid_search=False, balanced=None
):
    """
    Train a Random Forest model with optional grid search and evaluate performance.

    Parameters:
    -----------
    X : array-like or DataFrame
        Features
    y : array-like or Series
        Target variable
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    grid_search : bool, default=False
        Whether to perform grid search for hyperparameter tuning
    balanced : str or None, default=None
        Class weight balancing ('balanced' or None)

    Returns:
    --------
    dict : Dictionary containing model, predictions, and evaluation metrics
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if grid_search:
        # Define parameter grid for grid search
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }

        # Create Random Forest model
        rf = RandomForestClassifier(random_state=random_state, class_weight=balanced)

        # Perform grid search
        print(f"Performing grid search using {os.cpu_count()} CPU cores...")
        grid_search_cv = GridSearchCV(
            rf, param_grid, cv=5, scoring="f1", n_jobs=os.cpu_count(), verbose=1
        )
        grid_search_cv.fit(X_train, y_train)

        # Use best model
        model = grid_search_cv.best_estimator_
        print(f"Best parameters: {grid_search_cv.best_params_}")
        print(f"Best cross-validation score: {grid_search_cv.best_score_:.4f}")

    else:
        # Train simple Random Forest
        model = RandomForestClassifier(
            random_state=random_state, n_estimators=100, class_weight=balanced
        )
        model.fit(X_train, y_train)

    # Print evaluation results for training
    print("\n" + "=" * 50)
    print("RANDOM FOREST EVALUATION RESULTS (train set)")
    print("=" * 50)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    print(f"Training Accuracy:  {train_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Training Recall:    {train_recall:.4f}")
    print(f"Training F1-Score:  {train_f1:.4f}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Print evaluation results
    print("\n" + "=" * 50)
    print("RANDOM FOREST EVALUATION RESULTS (test set)")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (Random Forest specific)
    if hasattr(model, "feature_importances_"):
        print("\nTop 10 Feature Importances:")
        feature_names = (
            X.columns
            if hasattr(X, "columns")
            else [f"Feature_{i}" for i in range(X.shape[1])]
        )
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
        plt.title("Top 10 Feature Importances - Random Forest")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.show()

    return

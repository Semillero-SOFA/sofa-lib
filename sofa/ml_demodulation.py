"""
Machine learning-based demodulation methods.

This module contains functions for demodulating signals using various
machine learning algorithms including KNN, SVM, and K-Means clustering.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .modulation import realify


def kfold_cross_validation(
    X: np.ndarray, y: np.ndarray, n_splits: int, algorithm_func, *args, **kwargs
) -> tuple:
    """
    Performs k-fold cross-validation using the specified algorithm function.

    Parameters:
        X : np.ndarray
            Input data.
        y : np.ndarray
            Target labels.
        n_splits : int
            Number of folds.
        algorithm_func : callable
            Algorithm function to be used for each fold.
        *args : Any
            Variable length arguments to be passed to the algorithm function.
        **kwargs : Any
            Keyword arguments to be passed to the algorithm function.

    Returns:
        tuple
            Results and test data for each fold.
    """
    results = []
    tests = []
    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        result = algorithm_func(X_train, y_train, X_test, *args, **kwargs)
        results.append(result)
        tests.append(y_test)

    return np.array(results), np.array(tests)


def demodulate_knn(
    X_rx: np.ndarray, sym_tx: np.ndarray, k: int, n_splits: int = 5
) -> tuple:
    """
    Demodulates using KNN with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        sym_tx : np.ndarray
            Transmitted symbols.
        k : int
            Parameter k for the KNN algorithm.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, y_train, X_test, k):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func, k=k)


def demodulate_svm(
    X_rx: np.ndarray, sym_tx: np.ndarray, C: float, gamma: float, n_splits: int = 5
) -> tuple:
    """
    Demodulates using SVM with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        sym_tx : np.ndarray
            Transmitted symbols.
        C : float
            Parameter C for the SVM algorithm.
        gamma : float
            Parameter gamma for the SVM algorithm.
        n_splits : int, optional
            Number of folds for k-fold cross-validation. Default is 5.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, y_train, X_test, C, gamma):
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func, C=C, gamma=gamma)


def demodulate_kmeans(X_rx: np.ndarray, mod_dict: dict, n_splits: int = 5) -> tuple:
    """
    Demodulates using K-Means with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        mod_dict : dict
            Modulation dictionary.
        n_splits : int, optional
            Number of folds for cross-validation, by default 5.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, _, X_test):
        A_mc = np.array([(x.real, x.imag) for x in list(mod_dict.values())])
        model = KMeans(n_clusters=16, n_init=1, init=A_mc)
        model.fit(X_train)
        return model.predict(X_test)

    X = realify(X_rx)
    # Create an empty array as a placeholder for y
    y = np.empty_like(X)

    return kfold_cross_validation(X, y, n_splits, algorithm_func)


def find_best_params(
    model, param_grid: dict, X_rx: np.ndarray, sym_tx: np.ndarray
) -> dict:
    """
    Finds the best parameters for a given model using specific data.

    Parameters:
        model: ML model to optimize.
        param_grid: Dictionary of model parameters.
        X_rx: Input data.
        sym_tx: Validated output data.

    Returns:
        dict: Optimized parameter dictionary.
    """
    X = realify(X_rx)
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)

    grid = GridSearchCV(model(), param_grid, verbose=0)

    grid.fit(X_train, y_train)

    return grid.best_params_

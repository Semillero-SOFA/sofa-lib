"""
Neural network-based demodulation methods.

This module contains functions for creating and using neural networks
for signal demodulation tasks.
"""

import numpy as np
import tensorflow as tf

from .modulation import realify
from .ml_demodulation import kfold_cross_validation


def classifier_model(
    layers_props_lst: list,
    loss_fn: tf.keras.losses.Loss,
    input_dim: int,
    n_classes: int,
) -> tf.keras.models.Sequential:
    """
    Creates a neural network classifier model with specified layers and loss function.

    Parameters:
        layers_props_lst (list): List of layer properties dictionaries.
            Each dictionary should contain the desired properties for each layer, such as the number of units and activation function.
        loss_fn (tf.keras.losses.Loss): Loss function to optimize in the neural network.

    Returns:
        tf.keras.models.Sequential: Compiled model.

    """
    model = tf.keras.Sequential()

    for i, layer_props in enumerate(layers_props_lst):
        if i == 0:
            model.add(tf.keras.layers.Dense(input_dim=input_dim, **layer_props))
        else:
            model.add(tf.keras.layers.Dense(**layer_props))

    model.add(tf.keras.layers.Dense(units=n_classes, activation="softmax"))

    model.compile(loss=loss_fn, optimizer="adam")

    return model


def demodulate_neural(
    X_rx: np.ndarray,
    sym_tx: np.ndarray,
    layer_props_lst: list,
    loss_fn: tf.keras.losses.Loss,
    n_splits: int = 5,
) -> tuple:
    """
    Demodulates using a neural network with k-fold cross-validation.

    Parameters:
        X_rx (np.ndarray): Received constellation.
        sym_tx (np.ndarray): Transmitted symbols.
        layer_props_lst (list): List of layer properties dictionaries for the neural network.
        loss_fn (tf.keras.losses.Loss): Loss function to optimize in the neural network.
        n_splits (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
        np.ndarray: Demodulated constellation.
    """

    def algorithm_func(X_train, y_train, X_test):
        model = classifier_model(layer_props_lst, loss_fn, X_train.shape[0])
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=300, mode="min", restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=5000,
            batch_size=64,
            callbacks=[callback],
            verbose=0,
        )
        return model.predict(X_test, verbose=0)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func)

"""
DemandCast - Deep Learning Model Architectures (TensorFlow / Keras)
Implements:
1. Traditional SimpleRNN Forecasting Model
2. Single-layer LSTM Forecasting Model
3. Two-layer Stacked LSTM Forecasting Model
Includes parameterization for arbitrary sequence lengths and feature dimensions.
"""

import os
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Models")


def get_keras():
    """Dynamically imports Keras / TensorFlow."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        return keras, layers, regularizers, tf
    except ImportError:
        import keras
        from keras import layers, regularizers
        return keras, layers, regularizers, None


def build_simplernn_model(input_shape: Tuple[int, int],
                          units: int = 64,
                          dropout: float = 0.2,
                          dense_units: int = 32,
                          learning_rate: float = 0.001):
    """
    Constructs a traditional SimpleRNN forecasting model.
    """
    keras, layers, _, _ = get_keras()

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(units, return_sequences=False, activation="tanh"),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout / 2.0),
        layers.Dense(1, activation="linear")
    ], name=f"SimpleRNN_{input_shape[0]}h")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_single_lstm_model(input_shape: Tuple[int, int],
                            units: int = 64,
                            dropout: float = 0.2,
                            dense_units: int = 32,
                            learning_rate: float = 0.001):
    """
    Constructs a single-layer Long Short-Term Memory (LSTM) forecasting model.
    """
    keras, layers, _, _ = get_keras()

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout / 2.0),
        layers.Dense(1, activation="linear")
    ], name=f"SingleLSTM_{input_shape[0]}h")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_stacked_lstm_model(input_shape: Tuple[int, int],
                             units_l1: int = 64,
                             units_l2: int = 32,
                             dropout: float = 0.2,
                             dense_units: int = 32,
                             learning_rate: float = 0.001):
    """
    Constructs a two-layer Stacked LSTM forecasting model.
    Layer 1 propagates sequential hidden representations (return_sequences=True).
    Layer 2 encodes temporal dependencies into a final state representation.
    """
    keras, layers, _, _ = get_keras()

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units_l1, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        layers.Dropout(dropout),
        layers.LSTM(units_l2, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1, activation="linear")
    ], name=f"StackedLSTM_{input_shape[0]}h")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def get_model(arch: str, input_shape: Tuple[int, int], **kwargs):
    """Factory method to instantiate models by architecture key."""
    arch_lower = arch.lower().replace("-", "_").replace(" ", "_")
    if "stacked" in arch_lower:
        return build_stacked_lstm_model(input_shape, **kwargs)
    elif "single" in arch_lower or "lstm" in arch_lower:
        return build_single_lstm_model(input_shape, **kwargs)
    elif "simplernn" in arch_lower or "rnn" in arch_lower:
        return build_simplernn_model(input_shape, **kwargs)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from 'simplernn', 'single_lstm', 'stacked_lstm'.")

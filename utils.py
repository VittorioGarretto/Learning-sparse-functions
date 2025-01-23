import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn
from keras import regularizers
from sklearn import metrics
from keras import layers, Sequential, Input

def nn_model(depth: int, width: int, input_shape: int) -> Sequential:
    """
    Function to create a neural network model with a given depth, width and input shape. This model
    is used in a regression task.

    Parameters:
    depth (int): Number of hidden layers in the model.
    width (int): Number of neurons in each hidden layer.
    input_shape (int): Number of features in the input data.

    Returns:
    model (keras.Sequential): Neural network model.
    """

    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for _ in range(depth):
        model.add(layers.Dense(width, activation="relu", kernel_regularizer=regularizers.L1(l1=0.02)))
    model.add(layers.Dense(1))

    return model
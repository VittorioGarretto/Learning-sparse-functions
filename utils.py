import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn
from keras import regularizers
from sklearn import metrics
from keras import layers, Sequential, Input


def load_data(dim: int, size: int, target: list, 
              msp: bool, batch_size: int) -> tuple[np.ndarray, np.ndarray, tf.data.Dataset]:
    """
    Function to generate the data for the regression task.

    Parameters:
    dim (int): Number of features in the input data.
    size (int): Number of samples
    target (list): List of features to be used to generate the target data. 
    msp (bool): Flag to indicate if the data follows the merged staircase property.
    batch_size (int): Batch size for the dataset.

    Returns:
    x (np.ndarray): Input data.
    y (np.ndarray): Target data.
    dataset (tf.data.Dataset): Dataset object for the input and target data.
    """

    x = np.random.normal((size, dim))
    if msp:
        y = np.zeros(size)
        for i in range(1, len(target)+1):
            y += np.prod(x[:, target[:i]], axis=1) #Target data is the product of the selected features with increasing order
    else:
        y = np.prod(x[:, target], axis=1)  #Target data is the product of the selected features

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return x, y, dataset


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


def nn_train(model: Sequential, dataset: tf.data.Dataset, lr: int, epochs: int) -> tuple[dict, Sequential]:
    """
    Function to train a neural network model.

    Parameters:
    model (keras.Sequential): Neural network model.
    lr (int): Learning rate for the optimizer.
    epochs (int): Number of epochs for training.

    Returns:
    metrics (dict): History of the model training.
    model (keras.Sequential): Trained neural network model.
    """

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"])
    
    history = model.fit(dataset, epochs=epochs, verbose=0)
    metrics = {"loss": history.history["loss"], "train_err": history.history["mean_squared_error"]}

    return metrics, model
 
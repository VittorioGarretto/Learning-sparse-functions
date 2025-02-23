import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn
from keras import regularizers
from sklearn import metrics


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

    x = np.random.normal(size=(size, dim))
    if msp:
        y = np.zeros(size)
        for i in range(1, len(target)+1):
            y += np.prod(x[:, target[:i]], axis=1) #Target data is the product of the selected features with increasing order
    else:
        y = np.prod(x[:, target], axis=1)  #Target data is the product of the selected features

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return x, y, dataset

 
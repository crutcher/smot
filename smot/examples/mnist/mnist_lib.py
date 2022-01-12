from typing import Tuple, Type, Union

import tensorflow as tf
from tensorflow.python.keras.layers.pooling import Pooling2D

from smot.common import expect

INPUT_SHAPE = (28, 28, 1)
N_CLASSES = 10


def load_mnist_data_28x28x1() -> Tuple[Tuple, Tuple]:
    """
    load the MNIST data as (..., 28, 28, 1) [0, 1.0] data sets.

    Restructures image data to be (None, 28, 28, 1) and (None, 10) shaped.

    Rescales into [0, 1.0] range.

    :return: ((x_train, y_train), (x_test, y_test))
    """
    # Load (and cache) standard MNIST dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # The dataset has no channel information:
    expect.Expect.is_eq(x_train.shape[1:], (INPUT_SHAPE[:-1]))

    # Rescale the train and test data into [0, 1.0]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # And add a trivial channel dim:
    x_train = tf.expand_dims(x_train, 3)
    x_test = tf.expand_dims(x_test, 3)

    assert len(set(y_train)) == N_CLASSES
    assert len(set(y_test)) == N_CLASSES
    assert x_train.shape[1:] == INPUT_SHAPE
    assert x_test.shape[1:] == INPUT_SHAPE

    # Convert the class numbers to 1-hot categorical values.
    y_train = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, N_CLASSES)

    return ((x_train, y_train), (x_test, y_test))


def construct_LeNet5_mnist_model(
    pooling=tf.keras.layers.AveragePooling2D,
) -> tf.keras.Model:
    """
    Construct a LeNet-5 model.

    :param pooling: pooling layer class.

    Model is not compiled.

    References
    ==========
    * GradientBased Learning Applied to Document Recognition
      http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

    * Understanding and Implementing LeNet-5 CNN Architecture (Deep Learning)
      https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
    """
    model = tf.keras.models.Sequential(
        name="LeNet5",
        layers=[
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=5,
                input_shape=INPUT_SHAPE,
                strides=1,
                activation="tanh",
                padding="same",
            ),  # C1
            pooling(),
            tf.keras.layers.Conv2D(
                16,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="valid",
            ),  # C3
            pooling(),
            tf.keras.layers.Flatten(),  # Flatten
            tf.keras.layers.Dense(120, activation="tanh"),  # C5
            tf.keras.layers.Dense(84, activation="tanh"),  # F6
            tf.keras.layers.Dense(10, activation="softmax"),  # Output layer
        ],
    )

    return model


def build_LeNet5_mnist_model(
    *,
    pooling: Type[Pooling2D] = tf.keras.layers.AveragePooling2D,
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
) -> tf.keras.Model:
    """
    Return a compiled LeNet-5 model.

    :return: the model.
    """
    model = construct_LeNet5_mnist_model(
        pooling=pooling,
    )

    # Compile using 'Adam'
    model.compile(
        optimizer=optimizer,
        # Needed for the categorical softmax layer.
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model


def construct_ijcai2011_mnist_model(
    pooling=tf.keras.layers.MaxPooling2D,
) -> tf.keras.Model:
    """
    Construct a ijcai2011 model.

    :param pooling: pooling layer class.

    Model is not compiled.

    References
    ==========
    * Flexible, High Performance Convolutional Neural Networks for Image Classification
      https://dl.acm.org/doi/10.5555/2283516.2283603
    """
    model = tf.keras.models.Sequential(
        name="LeNet5",
        layers=[
            tf.keras.layers.Conv2D(
                filters=20,
                kernel_size=5,
                input_shape=INPUT_SHAPE,
                strides=1,
                padding="same",
                activation="tanh",
            ),  # C1
            pooling(),
            tf.keras.layers.Conv2D(
                filters=40,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
            ),  # C3
            pooling(),
            tf.keras.layers.Conv2D(
                filters=60,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
            ),  # C3
            pooling(),
            tf.keras.layers.Conv2D(
                filters=80,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="same",
            ),  # C3
            pooling(),
            tf.keras.layers.Flatten(),  # Flatten
            # tf.keras.layers.Dense(100, activation="tanh"),  # C5
            # tf.keras.layers.Dense(120, activation="tanh"),  # C5
            tf.keras.layers.Dense(150, activation="tanh"),  # C5
            tf.keras.layers.Dense(N_CLASSES, activation="softmax"),  # Output layer
        ],
    )

    return model


def build_ijcal2011_mnist_model(
    *,
    pooling=tf.keras.layers.MaxPooling2D,
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
    loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
) -> tf.keras.Model:
    """
    Return a compiled LeNet-5 model.

    :param pooling: the pooling layer type.
    :param optimizer: the optimizer.
    :param loss: the loss function.

    :return: the model.
    """
    model = construct_ijcai2011_mnist_model(
        pooling=pooling,
    )

    # Compile using 'Adam'
    model.compile(
        optimizer=optimizer,
        # Needed for the categorical softmax layer.
        loss=loss,
        metrics=["accuracy"],
    )

    return model

from typing import Any, Tuple, Union

import nptyping
import tensorflow as tf

INPUT_SHAPE = (28, 28, 1)
CLASSES = 10


def load_LeNet5_datasets() -> Tuple[
    Tuple[nptyping.NDArray[(Any, 28, 28, 1)], nptyping.NDArray[(Any, 10)]],
    Tuple[nptyping.NDArray[(Any, 28, 28, 1)], nptyping.NDArray[(Any, 10)]],
]:
    """
    Download and load the LeNet-5 datasets.

    Restructures image data to be (None, 28, 28, 1) and (None, 10) shaped.

    :return: ((x_train, y_train), (x_test, y_test))
    """
    # Load (and cache) standard MNIST dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the train and test data.
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = tf.expand_dims(x_train, 3)
    x_test = tf.expand_dims(x_test, 3)

    assert len(set(y_train)) == CLASSES
    assert len(set(y_test)) == CLASSES
    assert x_train.shape[1:] == INPUT_SHAPE
    assert x_test.shape[1:] == INPUT_SHAPE

    # Convert the class numbers to 1-hot categorical values.
    y_train = tf.keras.utils.to_categorical(y_train, CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

    return ((x_train, y_train), (x_test, y_test))


def construct_LeNet5_model() -> tf.keras.Model:
    """
    Construct a LeNet-5 model.

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
            tf.keras.layers.AveragePooling2D(),  # S2
            tf.keras.layers.Conv2D(
                16,
                kernel_size=5,
                strides=1,
                activation="tanh",
                padding="valid",
            ),  # C3
            tf.keras.layers.AveragePooling2D(),  # S4
            tf.keras.layers.Flatten(),  # Flatten
            tf.keras.layers.Dense(120, activation="tanh"),  # C5
            tf.keras.layers.Dense(84, activation="tanh"),  # F6
            tf.keras.layers.Dense(10, activation="softmax"),  # Output layer
        ],
    )

    return model


def build_LeNet5_model(
    *,
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
) -> tf.keras.Model:
    """
    Return a compiled LeNet-5 model.

    :return: the model.
    """
    model = construct_LeNet5_model()

    # Compile using 'Adam'
    model.compile(
        optimizer=optimizer,
        # Needed for the categorical softmax layer.
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model

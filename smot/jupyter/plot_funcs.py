import matplotlib.pyplot as plt
import tensorflow as tf


def set_pyplot_defaults() -> None:
  """
  Set sensible defaults for ``matplotlib.pyplot``.

  :param plt: the ``matplotlib.pyplot`` module.
  """
  # Force a non-transparent plot gutter
  plt.style.use({'figure.facecolor': 'white'})


def plot_history(history: tf.keras.callbacks.History) -> None:
  """
  Plot training history.

  :param history: the history to plot.
  """

  def plot_graphs(h: tf.keras.callbacks.History, metric: str) -> None:
    plt.plot(h.history[metric])
    plt.plot(h.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")

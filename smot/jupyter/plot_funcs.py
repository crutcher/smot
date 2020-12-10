from typing import Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf


def set_pyplot_defaults() -> None:
  """
  Set sensible defaults for ``matplotlib.pyplot``.

  :param plt: the ``matplotlib.pyplot`` module.
  """
  # plt.xkcd()

  # Force a non-transparent plot gutter
  plt.style.use({'figure.facecolor': 'white'})


def plot_history(
  history: tf.keras.callbacks.History,
  test_eval: Optional[Tuple[float, float]] = None,
) -> None:
  """
  Plot training history.

  :param history: the history to plot.
  :param test_eval: optional, the (test_loss, test_accuracy).
  """
  plt.style.use({'figure.facecolor': 'white'})

  fig, ax1 = plt.subplots()

  plt.title('model fit report')

  has_val = "val_loss" in history.history

  ax1.set_ylabel("loss", color="red")
  ax1.plot(history.history["loss"], color='red')

  if has_val:
    ax1.plot(history.history["val_loss"], color='red', dashes=[5])

  ax2 = ax1.twinx()
  ax2.set_xlabel("Epochs")
  ax2.set_ylabel("accuracy", color="blue")

  legend_handles = []
  train_handle, = ax2.plot(
    history.history["accuracy"],
    color='blue',
    label='train',
  )
  legend_handles.append(train_handle)

  if has_val:
    val_handle, = ax2.plot(
      history.history["val_accuracy"],
      color='blue',
      dashes=[5],
      label='val',
    )
    legend_handles.append(val_handle)

  if test_eval:
    test_loss, test_accuracy = test_eval
    ax1.axhline(test_loss, color='lightcoral', linestyle='dotted')

    test_handle = ax2.axhline(
      test_accuracy,
      color='dodgerblue',
      linestyle='dotted',
      label='test',
    )
    legend_handles.append(test_handle)

  ax2.legend(handles=legend_handles)

  plt.show()

from typing import Optional, Tuple

from matplotlib import pyplot as plt
import nptyping
import tensorflow as tf


def _plot_model_fit(
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

  ax2.legend(handles=legend_handles, loc='lower left')

  plt.show()


def model_fit_report(
  *,
  model: tf.keras.models.Model,
  history: tf.keras.callbacks.History,
  test_data: Tuple[nptyping.NDArray, nptyping.NDArray],
) -> Tuple[float, float]:
  """
  Evaluate a model against its test data, and graph the results.

  Example::

  >>> # Evaluate the model with the test data.
  >>> test_loss, test_accuracy = model_reports.model_fit_report(
  >>>   model=model,
  >>>   history=history,
  >>>   test_data=(x_test, y_test),
  >>> )

  :param model: the trained model.
  :param history: the History result of ``model.fit``
  :param test_data: the (xs, ys) for the test data set.
  :return: the (loss, accuracy) results from ``model.evaluate`` on test_data.
  """
  x_test, y_test = test_data
  eval_result = model.evaluate(x_test, y_test)
  test_loss, test_accuracy = eval_result

  print(
    f"Test loss: {test_loss}, Test accuracy: {test_accuracy}, Error rate: {1.0 - test_accuracy}")

  _plot_model_fit(history, (test_loss, test_accuracy))

  return (test_loss, test_accuracy)

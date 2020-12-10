from typing import Tuple

import nptyping
import tensorflow as tf

from smot.jupyter import plot_funcs


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

  print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

  plot_funcs.plot_history(history, (test_loss, test_accuracy))

  return (test_loss, test_accuracy)

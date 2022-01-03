"""
import unittest

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
"""

from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import (
    backend,
    combinations,
    keras_parameterized,
    testing_utils,
)
from tensorflow.python.ops.ragged import (
    ragged_factory_ops,
    ragged_tensor,
    ragged_tensor_value,
)
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class MergeLayersTest(keras_parameterized.TestCase):
    def test_merge_add(self):
        i1 = keras.layers.Input(shape=(4, 5))
        i2 = keras.layers.Input(shape=(4, 5))
        i3 = keras.layers.Input(shape=(4, 5))

        add_layer = keras.layers.Add()
        o = add_layer([i1, i2, i3])
        self.assertListEqual(o.shape.as_list(), [None, 4, 5])
        model = keras.models.Model([i1, i2, i3], o)
        model.run_eagerly = testing_utils.should_run_eagerly()

        x1 = np.random.random((2, 4, 5))
        x2 = np.random.random((2, 4, 5))
        x3 = np.random.random((2, 4, 5))
        out = model.predict([x1, x2, x3])
        self.assertEqual(out.shape, (2, 4, 5))
        self.assertAllClose(out, x1 + x2 + x3, atol=1e-4)

        self.assertEqual(add_layer.compute_mask([i1, i2, i3], [None, None, None]), None)
        self.assertTrue(
            np.all(
                backend.eval(
                    add_layer.compute_mask(
                        [i1, i2], [backend.variable(x1), backend.variable(x2)]
                    )
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            add_layer.compute_mask([i1, i2, i3], x1)
        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            add_layer.compute_mask(i1, [None, None, None])
        with self.assertRaisesRegex(ValueError, " should have the same length."):
            add_layer.compute_mask([i1, i2, i3], [None, None])

import unittest

import numpy as np
import torch

from smot.testlib import eggs, torch_eggs


class FromNumpyTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.from_numpy.html

    def test_from_numpy(self):
        source = np.array([[1, 2], [3, 4]], dtype=float)

        # build a tensor that shares memory with the numpy array.
        view = torch.from_numpy(source)

        torch_eggs.assert_tensor(
            view,
            torch.tensor([[1, 2], [3, 4]], dtype=float),
        )

        # both objects share the same underlying data pointer.
        eggs.assert_match(
            view.data_ptr(),
            source.ctypes.data,
        )

        # mutations to one mutate the other.
        source[0, 0] = 8

        torch_eggs.assert_tensor(
            view,
            torch.tensor([[8, 2], [3, 4]], dtype=float),
        )

import unittest

import numpy as np
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.from_numpy",
    ref="https://pytorch.org/docs/stable/generated/torch.from_numpy.html",
)
class FromNumpyTest(unittest.TestCase):
    def test_from_numpy(self) -> None:
        source: np.typing.ArrayLike = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        # build a tensor that shares memory with the numpy array.
        view = torch.from_numpy(source)

        torch_eggs.assert_tensor_equals(
            view,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        )

        # both objects share the same underlying data pointer.
        eggs.assert_match(
            view.data_ptr(),
            source.ctypes.data,  # type: ignore
        )

        # mutations to one mutate the other.
        source[0, 0] = 8.0  # type: ignore

        torch_eggs.assert_tensor_equals(
            view,
            torch.tensor([[8, 2], [3, 4]], dtype=torch.float64),
        )

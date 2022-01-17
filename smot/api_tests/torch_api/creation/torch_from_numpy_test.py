import numpy as np
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class FromNumpyTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.from_numpy.html"
    TARGET = torch.from_numpy

    def test_from_numpy(self) -> None:
        source: np.typing.ArrayLike = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        # build a tensor that shares memory with the numpy array.
        view = torch.from_numpy(source)

        torch_eggs.assert_tensor(
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

        torch_eggs.assert_tensor(
            view,
            torch.tensor([[8, 2], [3, 4]], dtype=torch.float64),
        )

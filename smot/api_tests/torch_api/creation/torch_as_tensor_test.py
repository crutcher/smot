import unittest

import hamcrest
import numpy as np
import pytest
import torch

from smot.testlib import torch_eggs


class AsTensorTest(unittest.TestCase):
    def test_list(self):
        torch_eggs.assert_tensor(
            torch.as_tensor([1, 2]),
            torch.tensor([1, 2]),
        )

    def test_numpy(self):
        torch_eggs.assert_tensor(
            torch.as_tensor(np.array([1, 2])),
            torch.tensor([1, 2]),
        )

    def test_tensor(self):
        t = torch.tensor([1, 2])

        # when possible (same device, dtype), `.as_tensor` is identity.
        hamcrest.assert_that(
            torch.as_tensor(t),
            hamcrest.same_instance(t),
        )

    def test_tensor_dtype_conversion(self):
        t = torch.tensor([1, 2], dtype=torch.float32)

        # when possible (same device, dtype), `.as_tensor` is identity.
        hamcrest.assert_that(
            torch.as_tensor(t, dtype=torch.float32),
            hamcrest.same_instance(t),
        )

        # dtype conversion is a copy.
        x = torch.as_tensor(t, dtype=torch.float64)
        torch_eggs.assert_tensor(
            x,
            torch.tensor([1, 2], dtype=torch.float64),
        )
        hamcrest.assert_that(
            x,
            hamcrest.not_(hamcrest.same_instance(t)),
        )

    @pytest.mark.slow
    def test_tensor_cuda_conversion(self):
        if torch.cuda.is_available():
            t = torch.tensor([1.0, 2.0], device="cpu")

            hamcrest.assert_that(
                torch.as_tensor(t, device="cpu"),  # type: ignore
                hamcrest.same_instance(t),
            )

            x = torch.as_tensor(t, device="cuda")  # type: ignore
            torch_eggs.assert_tensor(
                x,
                torch.tensor([1.0, 2.0], device="cuda"),
            )
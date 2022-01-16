import unittest

import hamcrest
import numpy as np
import pytest
import torch

from smot.testlib import eggs, torch_eggs


class TensorTest(unittest.TestCase):
    def test_scalar_tensor(self):
        t = torch.tensor(1)

        eggs.assert_match(
            t.item(),
            1,
        )

        eggs.assert_match(
            t.shape,
            torch.Size([]),
        )

        eggs.assert_match(
            t.numel(),
            1,
        )

    def test_copy(self):
        source_list = [1, 2]
        t = torch.tensor(source_list)
        torch_eggs.assert_tensor(t, source_list)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source_list)),
        )

        source_t = torch.tensor([1, 2])
        t = source_t.clone().detach()
        torch_eggs.assert_tensor(t, source_t)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source_t)),
        )

        source_nd = np.array([1, 2])
        torch_eggs.assert_tensor(t, source_nd)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source_nd)),
        )

    def test_requires_grad(self):
        for dtype in [torch.float32, torch.complex32]:
            eggs.assert_false(
                torch.tensor([1], dtype=dtype).requires_grad,
            )
            eggs.assert_true(
                torch.tensor([1], dtype=dtype, requires_grad=True).requires_grad,
            )

        # Error: Only float and complex types can require a gradiant.
        eggs.assert_raises(
            lambda: torch.tensor([1], dtype=torch.int8, requires_grad=True),
            RuntimeError,
        )

    @pytest.mark.slow
    def test_tensor_device(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]

        for d in devices:
            t = torch.tensor([1], device=d)
            eggs.assert_match(t.device, torch.device(d))

    @pytest.mark.slow
    def test_create_pinned(self):
        # this is expensive.
        if torch.cuda.is_available():
            t = torch.tensor([1], pin_memory=True)  # type: ignore
            eggs.assert_true(t.is_pinned())

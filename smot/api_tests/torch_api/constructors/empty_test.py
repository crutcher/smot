import unittest

import pytest
import torch

from smot.testlib import eggs, torch_eggs


class EmptyTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.empty.html

    def test_empty_zero(self):
        torch_eggs.assert_tensor(
            torch.empty(0),
            torch.ones(0),
        )

    def test_empty(self):
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor_structure(
                torch.empty(3, dtype=dtype),
                torch.tensor(
                    # random data ...
                    [0, 0, 0],
                    dtype=dtype,
                ),
            )

    @pytest.mark.slow
    def test_empty_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                torch_eggs.assert_tensor_structure(
                    torch.empty(
                        3,
                        dtype=dtype,
                        device="cuda",
                    ),
                    torch.tensor(
                        # random data ...
                        [0, 0, 0],
                        dtype=dtype,
                        device="cuda",
                    ),
                )


class EmptyLikeTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.empty_like.html

    def test_empty_like_scalar(self):
        t = torch.tensor(0)

        torch_eggs.assert_tensor_structure(
            torch.empty_like(t),
            torch.tensor(0),
        )

    def test_empty_like(self):
        for dtype in [torch.int8, torch.float32]:
            for data in [0, [[0]], [[1], [2]]]:
                t = torch.tensor(data, dtype=dtype)

                torch_eggs.assert_tensor_structure(
                    torch.empty_like(t),
                    t,
                )

    @pytest.mark.slow
    def test_empty_like_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                for data in [0, [[0]], [[1], [2]]]:
                    t = torch.tensor(data, dtype=dtype, device="cuda")

                    torch_eggs.assert_tensor_structure(
                        torch.empty_like(t),
                        t,
                    )


class EmptyStridedTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.empty_strided.html

    def test_empty_strided_scalar(self):
        eggs.assert_raises(
            lambda: torch.empty_strided(1, 1),
            TypeError,
            "must be tuple of ints, not int",
        )

    def test_empty_strided(self):
        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (1, 2)),
            torch.ones(2, 3),
        )

        # out of order strides ...
        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (2, 1)),
            torch.ones(2, 3),
        )

        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (1, 4)),
            torch.ones(2, 3),
        )

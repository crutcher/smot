import unittest

import pytest
import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.empty",
    ref="https://pytorch.org/docs/stable/generated/torch.empty.html",
)
class EmptyTest(unittest.TestCase):
    def test_empty_zero(self) -> None:
        torch_eggs.assert_tensor(
            torch.empty(0),
            torch.ones(0),
        )

    def test_empty(self) -> None:
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
    def test_empty_cuda(self) -> None:
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

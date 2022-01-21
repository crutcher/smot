import unittest

import pytest
import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.empty_like",
    ref="https://pytorch.org/docs/stable/generated/torch.empty_like.html",
)
class EmptyLikeTest(unittest.TestCase):
    def test_empty_like_scalar(self) -> None:
        t = torch.tensor(0)

        torch_eggs.assert_tensor_structure(
            torch.empty_like(t),
            torch.tensor(0),
        )

    def test_empty_like(self) -> None:
        for dtype in [torch.int8, torch.float32]:
            for data in [0, [[0]], [[1], [2]]]:
                t = torch.tensor(data, dtype=dtype)

                torch_eggs.assert_tensor_structure(
                    torch.empty_like(t),
                    t,
                )

    @pytest.mark.slow
    def test_empty_like_cuda(self) -> None:
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                for data in [0, [[0]], [[1], [2]]]:
                    t = torch.tensor(data, dtype=dtype, device="cuda")

                    torch_eggs.assert_tensor_structure(
                        torch.empty_like(t),
                        t,
                    )

import unittest

import pytest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.ones_like",
    ref="https://pytorch.org/docs/stable/generated/torch.ones_like.html",
)
class OnesLikeTest(unittest.TestCase):
    def dense(self, device: str) -> None:
        # Dense Tensors
        for dtype in [torch.int8, torch.float32]:
            source = torch.tensor(
                [1, 2],
                dtype=dtype,
                device=device,
            )

            torch_eggs.assert_tensor(
                torch.ones_like(source),
                torch.ones(
                    *source.size(),
                    dtype=source.dtype,
                    device=source.device,
                    layout=source.layout,
                ),
            )

    def test_cpu_dense(self) -> None:
        self.dense("cpu")

    @pytest.mark.slow
    def test_cuda_dense(self) -> None:
        if torch.cuda.is_available():
            self.dense("cuda")

    def test_sparse(self) -> None:
        coo = torch.tensor([[0, 0], [2, 2]])
        vals = torch.tensor([3, 4])
        source = torch.sparse_coo_tensor(
            indices=coo,
            values=vals,
        )

        # torch.ones_like() refuses to play nice with sparse coo tensors.
        #
        # Arguably, this makes sense. There would only be two valid answers:
        #  1. a sparse-ones tensor, filled in at places the source had presence.
        #  2. an entirely filled in tensor (which ... why sparse at that point)?
        #
        # However, rather than just throwing an AssertionError telling you
        # that you can't do this, it throws a NotImplementedError complaining
        # about the tensor backend provider not supporting the operation.
        eggs.assert_raises(
            lambda: torch.ones_like(source),
            NotImplementedError,
            "with arguments from the 'SparseCPU' backend",
        )

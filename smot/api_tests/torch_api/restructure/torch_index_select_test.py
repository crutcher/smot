import unittest

import torch

from smot.testlib import eggs, torch_eggs


class IndexSelectTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.index_select.html

    def source_tensor(self) -> torch.Tensor:
        source = torch.arange(9).reshape(3, 3)
        torch_eggs.assert_tensor(
            source,
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        )
        return source

    def test_index_select(self):
        source = self.source_tensor()

        torch_eggs.assert_tensor(
            torch.index_select(source, 0, torch.tensor([1, 2, 0])),
            [
                [3, 4, 5],
                [6, 7, 8],
                [0, 1, 2],
            ],
        )

        torch_eggs.assert_tensor(
            torch.index_select(source, 1, torch.tensor([1, 2, 0])),
            [
                [1, 2, 0],
                [4, 5, 3],
                [7, 8, 6],
            ],
        )

        # No enforcement is done that each index is used, or used only once:
        torch_eggs.assert_tensor(
            torch.index_select(source, 1, torch.tensor([0, 0, 0])),
            [
                [0, 0, 0],
                [3, 3, 3],
                [6, 6, 6],
            ],
        )

        # any number of indexes can be used.
        torch_eggs.assert_tensor(
            torch.index_select(source, 1, torch.tensor([0, 1, 0, 2, 0])),
            [
                [0, 1, 0, 2, 0],
                [3, 4, 3, 5, 3],
                [6, 7, 6, 8, 6],
            ],
        )

    def test_index_degenerate(self):
        source = self.source_tensor()

        torch_eggs.assert_tensor(
            torch.index_select(source, 1, torch.tensor([], dtype=torch.int64)),
            torch.tensor(
                [
                    [],
                    [],
                    [],
                ],
                dtype=source.dtype,
            ),
        )

    def test_errors(self):
        source = self.source_tensor()

        eggs.assert_raises(
            lambda: torch.index_select(source, 4, torch.tensor([0])),
            IndexError,
            "Dimension out of range",
        )

        eggs.assert_raises(
            lambda: torch.index_select(source, 1, torch.tensor([4])),
            RuntimeError,
            "INDICES element is out of DATA bounds",
        )

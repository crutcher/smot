import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.dsplit",
    ref="https://pytorch.org/docs/stable/generated/torch.dsplit.html",
)
class DsplitTest(unittest.TestCase):
    def test_view(self) -> None:
        source = torch.arange(16.0).reshape(2, 2, 4)
        torch_eggs.assert_tensor_equals(
            source,
            torch.tensor(
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                    ],
                    [
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ],
                ]
            ),
        )

        # dsplit(input, indices) => tensor_split(input, indices, dim=2)
        splits = torch.dsplit(source, 2)
        torch_eggs.assert_tensor_sequence_equals(
            splits,
            *torch.tensor_split(source, 2, dim=2),
        )
        torch_eggs.assert_tensor_sequence_equals(
            splits,
            [
                [
                    [0.0, 1.0],
                    [4.0, 5.0],
                ],
                [
                    [8.0, 9.0],
                    [12.0, 13.0],
                ],
            ],
            [
                [
                    [2.0, 3.0],
                    [6.0, 7.0],
                ],
                [
                    [10.0, 11.0],
                    [14.0, 15.0],
                ],
            ],
        )

        # split is a view:
        source[0, 0, 0] = 77.0
        splits[0][0, 0, 1] = 88.0

        torch_eggs.assert_tensor_equals(
            splits[0],
            [
                [
                    [77.0, 88.0],
                    [4.0, 5.0],
                ],
                [
                    [8.0, 9.0],
                    [12.0, 13.0],
                ],
            ],
        )

        torch_eggs.assert_tensor_equals(
            source,
            torch.tensor(
                [
                    [
                        [77.0, 88.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                    ],
                    [
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ],
                ]
            ),
        )

    def test_indices(self) -> None:
        source = torch.arange(16.0).reshape(2, 2, 4)
        torch_eggs.assert_tensor_equals(
            source,
            torch.tensor(
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                    ],
                    [
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ],
                ]
            ),
        )
        torch_eggs.assert_tensor_sequence_equals(
            torch.dsplit(source, [1, 3]),
            *torch.tensor_split(source, [1, 3], dim=2),
        )
        torch_eggs.assert_tensor_sequence_equals(
            torch.dsplit(source, [1, 3]),
            [
                [
                    [0.0],
                    [4.0],
                ],
                [
                    [8.0],
                    [12.0],
                ],
            ],
            [
                [
                    [1.0, 2.0],
                    [5.0, 6.0],
                ],
                [
                    [9.0, 10.0],
                    [13.0, 14.0],
                ],
            ],
            [
                [
                    [3.0],
                    [7.0],
                ],
                [
                    [11.0],
                    [15.0],
                ],
            ],
        )

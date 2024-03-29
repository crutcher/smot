import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.gather",
    ref="https://pytorch.org/docs/stable/generated/torch.gather.html",
)
class GatherTest(unittest.TestCase):
    def test_gather(self) -> None:
        source = torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.gather(
                input=source,
                dim=1,
                index=torch.tensor(
                    [
                        [0, 1],
                        [1, 0],
                    ],
                ),
            ),
            [
                [1, 2],
                [4, 3],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.gather(
                input=source,
                dim=0,
                index=torch.tensor(
                    [
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [1, 1],
                    ],
                ),
            ),
            [
                [1, 4],
                [3, 2],
                [1, 2],
                [3, 4],
            ],
        )

    def test_out(self) -> None:
        source = torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
        )

        target = torch.arange(4)
        orig_data_ptr = target.data_ptr()
        with eggs.ignore_warnings():
            torch.gather(
                input=source,
                dim=1,
                index=torch.tensor(
                    [
                        [0, 1],
                        [1, 0],
                    ],
                ),
                out=target,
            )

        torch_eggs.assert_tensor_equals(
            target,
            [
                [1, 2],
                [4, 3],
            ],
        )
        eggs.assert_match(
            target.data_ptr(),
            orig_data_ptr,
        )

import unittest

import torch

from smot.testlib import eggs, torch_eggs


class GatherTest(unittest.TestCase):
    def test_gather(self):
        source = torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
        )

        target = torch.arange(4)

        torch_eggs.assert_tensor(
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

        torch_eggs.assert_tensor(
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

    def test_out(self):
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

        torch_eggs.assert_tensor(
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

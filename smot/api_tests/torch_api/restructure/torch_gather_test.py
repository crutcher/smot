import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class GatherTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.gather.html"
    TARGET = torch.gather

    def test_gather(self):
        source = torch.tensor(
            [
                [1, 2],
                [3, 4],
            ],
        )

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

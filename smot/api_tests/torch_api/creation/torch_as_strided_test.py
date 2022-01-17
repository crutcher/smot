import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class AsStridedTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.as_strided.html"
    TARGET = torch.as_strided

    def test_as_strided(self):
        t = torch.rand(9)
        v = t.data

        x = torch.as_strided(t, (2, 2), (1, 2))

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr(),
        )

        torch_eggs.assert_tensor(
            x,
            [
                [v[1 * 0 + 2 * 0], v[1 * 0 + 2 * 1]],
                [v[1 * 1 + 2 * 0], v[1 * 1 + 2 * 1]],
            ],
        )

    def test_storage_offset(self):
        t = torch.rand(9, dtype=torch.float32)
        v = t.data
        offset = 2

        x = torch.as_strided(t, (2, 2), (1, 2), 2)

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr() + 4 * offset,
        )

        torch_eggs.assert_tensor(
            x,
            [
                [v[2 + 1 * 0 + 2 * 0], v[2 + 1 * 0 + 2 * 1]],
                [v[2 + 1 * 1 + 2 * 0], v[2 + 1 * 1 + 2 * 1]],
            ],
        )

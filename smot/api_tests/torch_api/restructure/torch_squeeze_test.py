import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class SqueezeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.squeeze.html"
    TARGET = torch.squeeze

    def test_squeeze(self):
        source = torch.zeros(2, 1, 3, 1, 4)
        eggs.assert_match(source.size(), torch.Size([2, 1, 3, 1, 4]))

        view = torch.squeeze(source)

        torch_eggs.assert_views(source, view)
        eggs.assert_match(view.size(), torch.Size([2, 3, 4]))

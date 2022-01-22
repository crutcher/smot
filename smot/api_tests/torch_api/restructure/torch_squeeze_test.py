import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.squeeze",
    ref="https://pytorch.org/docs/stable/generated/torch.squeeze.html",
)
class SqueezeTest(unittest.TestCase):
    def test_squeeze(self) -> None:
        source = torch.zeros(2, 1, 3, 1, 4)
        eggs.assert_match(source.size(), torch.Size([2, 1, 3, 1, 4]))

        view = torch.squeeze(source)

        torch_eggs.assert_views(source, view)
        eggs.assert_match(view.size(), torch.Size([2, 3, 4]))

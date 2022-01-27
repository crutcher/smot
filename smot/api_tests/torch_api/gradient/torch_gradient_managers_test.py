import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs


class GradientManagersTest(unittest.TestCase):
    @api_link(
        target="torch.no_grad",
        ref="https://pytorch.org/docs/stable/generated/torch.no_grad.html",
    )
    @api_link(
        target="torch.enable_grad",
        ref="https://pytorch.org/docs/stable/generated/torch.enable_grad.html",
    )
    def test_no_grad_enable_grad(self) -> None:
        g_source = torch.zeros(1, requires_grad=True)
        ng_source = torch.zeros(1, requires_grad=False)

        x = g_source * 2
        eggs.assert_true(x.requires_grad)

        x = ng_source * 2
        eggs.assert_false(x.requires_grad)

        # `no_grad()` disables gradient computation while it is in effect;
        # thread-local.
        with torch.no_grad():
            x = g_source * 2
            eggs.assert_false(x.requires_grad)

            x = ng_source * 2
            eggs.assert_false(x.requires_grad)

        x = g_source * 2
        eggs.assert_true(x.requires_grad)

        x = ng_source * 2
        eggs.assert_false(x.requires_grad)

        # `enable_grad()` doesn't do anything if `no_grad()` hasn't been called;
        # it re-enables only those gradient computations which would have
        # happened if no_grad() hadn't been in effect.
        with torch.enable_grad():
            x = g_source * 2
            eggs.assert_true(x.requires_grad)

            x = ng_source * 2
            eggs.assert_false(x.requires_grad)

        x = g_source * 2
        eggs.assert_true(x.requires_grad)

        x = ng_source * 2
        eggs.assert_false(x.requires_grad)

        with torch.no_grad():
            with torch.enable_grad():
                x = g_source * 2
                eggs.assert_true(x.requires_grad)

                # grad-less computations are still grad-less
                x = ng_source * 2
                eggs.assert_false(x.requires_grad)

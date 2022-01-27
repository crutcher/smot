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
    @api_link(
        target="torch.is_grad_enabled",
        ref="https://pytorch.org/docs/stable/generated/torch.is_grad_enabled.html",
    )
    def test_no_grad_enable_grad(self) -> None:
        g_source = torch.zeros(1, requires_grad=True)
        ng_source = torch.zeros(1, requires_grad=False)

        eggs.assert_true(torch.is_grad_enabled())

        x = g_source * 2
        eggs.assert_true(x.requires_grad)

        x = ng_source * 2
        eggs.assert_false(x.requires_grad)

        # `no_grad()` disables gradient computation while it is in effect;
        # thread-local.
        with torch.no_grad():
            eggs.assert_false(torch.is_grad_enabled())

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
            eggs.assert_true(torch.is_grad_enabled())

            x = g_source * 2
            eggs.assert_true(x.requires_grad)

            x = ng_source * 2
            eggs.assert_false(x.requires_grad)

        x = g_source * 2
        eggs.assert_true(x.requires_grad)

        x = ng_source * 2
        eggs.assert_false(x.requires_grad)

        with torch.no_grad():
            eggs.assert_false(torch.is_grad_enabled())

            with torch.enable_grad():
                eggs.assert_true(torch.is_grad_enabled())

                x = g_source * 2
                eggs.assert_true(x.requires_grad)

                # grad-less computations are still grad-less
                x = ng_source * 2
                eggs.assert_false(x.requires_grad)

    @api_link(
        target="torch.set_grad_enabled",
        ref="https://pytorch.org/docs/stable/generated/torch.set_grad_enabled.html",
    )
    def test_set_grad_enabled(self) -> None:
        # set_grad_enabled(mode: bool)
        #   - records the previous grad enabled state on init.
        #   - changes the state to `mode`
        #   - on __exit__ (when used as a context manager) restores the state.

        previous_state = torch.is_grad_enabled()

        with torch.set_grad_enabled(True):
            eggs.assert_match(torch.is_grad_enabled(), True)

        with torch.set_grad_enabled(False):
            eggs.assert_match(torch.is_grad_enabled(), False)

        eggs.assert_match(torch.is_grad_enabled(), previous_state)

        with torch.no_grad():
            eggs.assert_match(torch.is_grad_enabled(), False)

            with torch.set_grad_enabled(True):
                eggs.assert_match(torch.is_grad_enabled(), True)

    @api_link(
        target="torch.inference_mode",
        ref="https://pytorch.org/docs/stable/generated/torch.inference_mode.html",
    )
    @api_link(
        target="torch.is_inference_mode_enabled",
        ref="https://pytorch.org/docs/stable/generated/torch.is_inference_mode_enabled.html",
    )
    def test_inference_mode(self) -> None:
        with torch.enable_grad():
            eggs.assert_match(torch.is_grad_enabled(), True)
            eggs.assert_match(torch.is_inference_mode_enabled(), False)

            with torch.inference_mode():
                # inference_mode implies no_grad.
                eggs.assert_match(torch.is_grad_enabled(), False)
                eggs.assert_match(torch.is_inference_mode_enabled(), True)

            eggs.assert_match(torch.is_grad_enabled(), True)
            eggs.assert_match(torch.is_inference_mode_enabled(), False)

import unittest

import torch

from smot.doc_link.link_annotations import WEIRD_BUG, api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.quasirandom.SobolEngine",
    ref="https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html",
)
class SobolEngineTest(unittest.TestCase):
    def test_draw(self) -> None:
        with torch_eggs.reset_generator_seed():
            engine = torch.quasirandom.SobolEngine(dimension=4)

        eggs.assert_raises(
            # You'd think that this would return an empty tensor,
            # but instead it throws.
            lambda: engine.draw(0),
            RuntimeError,
            "tensor with negative dimension",
        )

        torch_eggs.assert_tensor_equals(
            engine.draw(1),
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
            ],
        )
        torch_eggs.assert_tensor_equals(
            engine.draw(2),
            [
                [0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500],
            ],
        )

        engine.reset()
        engine.fast_forward(2)

        torch_eggs.assert_tensor_equals(
            engine.draw(1),
            [
                [0.7500, 0.2500, 0.2500, 0.2500],
            ],
        )

    def test_draw_base2(self) -> None:
        with torch_eggs.reset_generator_seed():
            engine = torch.quasirandom.SobolEngine(dimension=4)

        torch_eggs.assert_tensor_equals(
            engine.draw_base2(0),
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
            ],
        )

        WEIRD_BUG(
            "torch.quasirandom.SobolEngine",
            (
                "`SobolEngine.draw_base2()` has a weird ordering requirement not shared by `draw()`, "
                + "and not discussed in the documentation, only described in an error."
            ),
        )

        eggs.assert_raises(
            lambda: engine.draw_base2(1),
            ValueError,
            "require n to be a power of 2",
        )

        torch_eggs.assert_tensor_equals(
            engine.draw_base2(0),
            [
                [0.5000, 0.5000, 0.5000, 0.5000],
            ],
        )
        torch_eggs.assert_tensor_equals(
            engine.draw_base2(1),
            [
                [0.7500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.7500, 0.7500, 0.7500],
            ],
        )

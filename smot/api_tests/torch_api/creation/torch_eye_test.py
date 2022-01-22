import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.eye",
    ref="https://pytorch.org/docs/stable/generated/torch.eye.html",
)
class EyeTest(unittest.TestCase):
    def test_eye_zero(self) -> None:
        # eye(0) still returns a (0,0) tensor.
        torch_eggs.assert_tensor(
            torch.eye(0),
            torch.ones(0, 0),
        )

    def test_eye(self) -> None:
        torch_eggs.assert_tensor(
            torch.eye(3),
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        torch_eggs.assert_tensor(
            torch.eye(3, 2),
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
        )

        torch_eggs.assert_tensor(
            torch.eye(3, 4),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )

    def test_eye_out(self) -> None:
        t = torch.ones(9)
        original_data = t.data_ptr()

        # same size, same data ptr.
        torch_eggs.assert_tensor(
            torch.eye(3, out=t),
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # smaller size, same data ptr.
        torch_eggs.assert_tensor(
            torch.eye(3, 2, out=t),
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # larger size, NEW data ptr.
        torch_eggs.assert_tensor(
            torch.eye(3, 4, out=t),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )

        eggs.assert_match(
            t.data_ptr(),
            hamcrest.not_(original_data),
        )

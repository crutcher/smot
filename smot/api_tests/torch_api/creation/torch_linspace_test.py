import unittest

import hamcrest
import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.linspace",
    ref="https://pytorch.org/docs/stable/generated/torch.linspace.html",
)
class LinspaceTest(unittest.TestCase):
    def test_linspace(self) -> None:
        torch_eggs.assert_tensor(
            torch.linspace(3, 10, steps=5),
            [3.0, 4.75, 6.5, 8.25, 10.0],
        )

        # negative steps.
        torch_eggs.assert_tensor(
            torch.linspace(10, 3, steps=5),
            [10.0, 8.25, 6.5, 4.75, 3.0],
        )

    def test_linspace_out(self) -> None:
        t = torch.ones(5)
        original_data = t.data_ptr()

        # same size, same data ptr.
        torch_eggs.assert_tensor(
            torch.linspace(3, 10, steps=5, out=t),
            [3.0, 4.75, 6.5, 8.25, 10.0],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # smaller size, same data ptr.
        torch_eggs.assert_tensor(
            torch.linspace(3, 10, steps=3, out=t),
            [3.0, 6.5, 10.0],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # larger size, NEW data ptr.
        torch_eggs.assert_tensor(
            torch.linspace(1, 7, steps=7, out=t),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )

        eggs.assert_match(
            t.data_ptr(),
            hamcrest.not_(original_data),
        )

    def test_linspace_100(self) -> None:
        # steps=100 as a default is deprecated.
        with eggs.ignore_warnings():
            torch_eggs.assert_tensor(
                torch.linspace(1, 100),
                [float(i) for i in range(1, 101)],
            )

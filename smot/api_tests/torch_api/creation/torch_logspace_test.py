import unittest

import hamcrest
import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.logspace",
    ref="https://pytorch.org/docs/stable/generated/torch.logspace.html",
)
class LogspaceTest(unittest.TestCase):
    def test_logspace(self) -> None:
        torch_eggs.assert_tensor(
            torch.logspace(-10, 10, steps=5),
            [1.0e-10, 1.0e-05, 1.0e00, 1.0e05, 1.0e10],
        )

        # negative steps.
        torch_eggs.assert_tensor(
            torch.logspace(10, -10, steps=5),
            [1.0e10, 1.0e05, 1.0e00, 1.0e-05, 1.0e-10],
        )

    def test_logspace_out(self) -> None:
        t = torch.ones(5)
        original_data = t.data_ptr()

        # same size, same data ptr.
        torch_eggs.assert_tensor(
            torch.logspace(-10, 10, steps=5, out=t),
            [1.0e-10, 1.0e-05, 1.0e00, 1.0e05, 1.0e10],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # smaller size, same data ptr.
        torch_eggs.assert_tensor(
            torch.logspace(-10, 10, steps=3, out=t),
            [1.0e-10, 1.0e00, 1.0e10],
        )

        eggs.assert_match(
            t.data_ptr(),
            original_data,
        )

        # larger size, NEW data ptr.
        torch_eggs.assert_tensor(
            torch.logspace(-12, 12, steps=9, out=t),
            [1.0e-12, 1.0e-09, 1.0e-06, 1.0e-03, 1.0e0, 1.0e03, 1.0e06, 1.0e09, 1.0e12],
        )

        eggs.assert_match(
            t.data_ptr(),
            hamcrest.not_(original_data),
        )

    def test_logspace_100(self) -> None:
        # steps=100 as a default is deprecated.
        with eggs.ignore_warnings():
            torch_eggs.assert_tensor(
                torch.logspace(1, 100),
                [float(f"1.e{i}") for i in range(1, 101)],
            )

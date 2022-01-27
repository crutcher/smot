import unittest

import psutil
import pytest
import torch

from smot.doc_link.link_annotations import WEIRD_API, api_link
from smot.testlib import eggs


@api_link(
    target="torch.get_num_threads",
    ref="https://pytorch.org/docs/stable/generated/torch.get_num_threads.html",
)
@api_link(
    target="torch.set_num_threads",
    ref="https://pytorch.org/docs/stable/generated/torch.set_num_threads.html",
)
class ThreadsTest(unittest.TestCase):
    def test_num_threads(self) -> None:
        original_threads = torch.get_num_threads()

        eggs.assert_match(
            torch.get_num_threads(),
            psutil.cpu_count(logical=False),
        )

        try:
            torch.set_num_threads(2)
            eggs.assert_match(
                torch.get_num_threads(),
                2,
            )

        finally:
            torch.set_num_threads(original_threads)

    @pytest.mark.forked
    def test_num_interop_threads(self) -> None:
        eggs.assert_match(
            torch.get_num_interop_threads(),
            psutil.cpu_count(logical=False),
        )

        WEIRD_API(
            target="torch.set_num_interop_threads",
            note=(
                "Run this test in a fork! Can only be called once and before "
                + "any inter-op parallel work is started (e.g. JIT execution)."
            ),
        )

        torch.set_num_interop_threads(2)

        eggs.assert_match(
            torch.get_num_interop_threads(),
            2,
        )

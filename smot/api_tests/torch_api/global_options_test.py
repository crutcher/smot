import copy
import logging
import math
import unittest

import hamcrest
import torch

from smot.testlib import eggs


class GlobalOptionsTest(unittest.TestCase):
    def test_set_printoptions(self):
        """torch.set_printoptions(...)

        Set global print options for pytorch.

        NOTE: there is no `torch.get_printoptions()`
        But you can grab it from

        Signature:
            torch.set_printoptions(
              precision=None,
              threshold=None,
              edgeitems=None,
              linewidth=None,
              profile=None,
              sci_mode=None,
            )

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.set_printoptions.html
        """
        original = copy.copy(torch._tensor_str.PRINT_OPTS)
        try:
            torch.set_printoptions(profile="default")
            hamcrest.assert_that(
                torch._tensor_str.PRINT_OPTS,
                hamcrest.has_properties(
                    dict(
                        precision=4,
                        threshold=1000,
                        edgeitems=3,
                        linewidth=80,
                    )
                ),
            )

            torch.set_printoptions(profile="short")
            hamcrest.assert_that(
                torch._tensor_str.PRINT_OPTS,
                hamcrest.has_properties(
                    dict(
                        precision=2,
                        threshold=1000,
                        edgeitems=2,
                        linewidth=80,
                    )
                ),
            )

            torch.set_printoptions(profile="full")
            hamcrest.assert_that(
                torch._tensor_str.PRINT_OPTS,
                hamcrest.has_properties(
                    dict(
                        precision=4,
                        threshold=math.inf,
                        edgeitems=3,
                        linewidth=80,
                    )
                ),
            )

            t = torch.tensor(1 / 3.0, dtype=torch.float32)
            for p in [1, 2, 4]:
                torch.set_printoptions(precision=p)
                eggs.assert_match(
                    repr(t),
                    f"tensor(0.{'3' * p})",
                )

            torch.set_printoptions(threshold=4)
            eggs.assert_match(
                repr(torch.tensor([1, 2, 3])),
                f"tensor([1, 2, 3])",
            )
            eggs.assert_match(
                repr(torch.tensor([1, 2, 3, 4, 5])),
                f"tensor([1, 2, 3, 4, 5])",
            )
            eggs.assert_match(
                repr(torch.tensor([1, 2, 3, 4, 5, 6])),
                f"tensor([1, 2, 3, 4, 5, 6])",
            )
            # Summarization begins at: (2*threshold)-1
            eggs.assert_match(
                repr(torch.tensor([1, 2, 3, 4, 5, 6, 7])),
                f"tensor([1, 2, 3,  ..., 5, 6, 7])",
            )
            eggs.assert_match(
                repr(torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])),
                f"tensor([1, 2, 3,  ..., 7, 8, 9])",
            )

            torch.set_printoptions(precision=2)
            torch.set_printoptions(sci_mode=False)
            eggs.assert_match(
                repr(torch.tensor(10000.0)),
                "tensor(10000.)",
            )
            torch.set_printoptions(sci_mode=True)
            eggs.assert_match(
                repr(torch.tensor(100000.0)),
                "tensor(1.00e+05)",
            )

            torch.set_printoptions(linewidth=15)
            eggs.assert_match(
                repr(torch.ones([4])),
                "\n".join(
                    [
                        "tensor([1., 1.,",
                        "        1., 1.])",
                    ]
                ),
            )

        finally:
            torch.set_printoptions(
                precision=original.precision,
                threshold=original.threshold,
                edgeitems=original.edgeitems,
                linewidth=original.linewidth,
                sci_mode=original.sci_mode,
            )

    def test_set_flush_denormal(self):
        """torch.set_flush_denormal(mode)

        Disables denormal floating numbers on the CPU.

        Only supported if the system supports flushing denormal numbers,
        and it successfully configures the mode.

        .. _Online Docs:
            https://pytorch.org/docs/stable/generated/torch.set_flush_denormal.html
        """
        if not torch.set_flush_denormal(True):
            logging.error("set_flush_denormal() failed")
            return

        try:
            eggs.assert_match(
                torch.tensor(1e-323, dtype=torch.float64).item(),
                0.0,
            )

            torch.set_flush_denormal(False)
            hamcrest.assert_that(
                torch.tensor(1e-323, dtype=torch.float64).item(),
                hamcrest.close_to(
                    9.88e-324,
                    1.0e-324,
                ),
            )

        finally:
            torch.set_flush_denormal(False)

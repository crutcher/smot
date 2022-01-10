import copy
import logging
import math
import unittest

import hamcrest
import numpy as np
import torch

from smot.testing import eggs


class TensorOpsTest(unittest.TestCase):
    def test_is_tensor(self):
        """torch.is_tensor(obj)

        Returns True if obj is a PyTorch tensor.

        Defined in terms of `isinstance(obj, Tensor)`

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_tensor.html
        """
        # True:
        # =======
        eggs.assert_true(
            torch.is_tensor(torch.tensor([1, 2])),
        )

        # False:
        # =======
        eggs.assert_false(
            torch.is_tensor("abc"),
        )

        eggs.assert_false(
            torch.is_tensor([1, 2]),
        )

        eggs.assert_false(
            torch.is_tensor(np.array([1, 2])),
        )

    def test_is_storage(self):
        """torch.is_storage(obj)

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_storage.html
        """

        t = torch.tensor([1, 2])
        ts = t.data.storage()

        eggs.assert_true(
            torch.is_storage(ts),
        )

        # Errors:
        # =======
        eggs.assert_false(
            torch.is_storage(t),
        )

    def test_is_complex(self):
        """torch.is_complex(input: Tensor)
        Also: `<tensor>.is_complex()`

        Returns true if the dtype of the input is complex.
        Requires input to be Tensor.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_complex.html
        """
        # True:
        # =======
        complex_t = torch.tensor([1j])

        eggs.assert_true(
            torch.is_complex(complex_t),
        )

        eggs.assert_true(
            complex_t.is_complex(),
        )

        # False:
        # =======
        eggs.assert_false(
            torch.is_complex(torch.tensor([1], dtype=torch.float32)),
        )

        # Errors:
        # =======
        eggs.assert_raises(
            lambda: torch.is_complex([1j]),
            TypeError,
        )

    def test_is_conj(self):
        """torch.is_conj(input: Tensor)
        Also: `<tensor>.is_complex()`

        Returns true if the conjugate bit of the tensor is flipped.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_conj.html
        """
        # True:
        # =======
        t = torch.tensor([1], dtype=torch.complex32)
        conj_t = torch.conj(t)

        complex_t = torch.tensor([1], dtype=torch.complex32)
        complex_conj_t = torch.conj(complex_t)

        # both torch.is_conj(t) and <tensor>.is_conj()
        eggs.assert_true(
            torch.is_conj(complex_conj_t),
        )
        eggs.assert_true(
            complex_conj_t.is_conj(),
        )

        eggs.assert_true(
            torch.is_conj(conj_t),
        )
        eggs.assert_true(
            conj_t.is_conj(),
        )

        # False:
        # =======
        eggs.assert_false(
            torch.is_conj(t),
        )

        eggs.assert_false(
            t.is_conj(),
        )

        eggs.assert_false(
            torch.is_conj(complex_t),
        )

        eggs.assert_false(
            complex_t.is_conj(),
        )

        # Errors:
        # =======
        eggs.assert_raises(
            lambda: torch.is_conj([1j]),
            TypeError,
        )

    def test_is_floating_point(self):
        """torch.is_floating_point(input: Tensor)
        Also: `<tensor>.is_floating_point()`

        Returns true if the data type of the tensor is floating point.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_floating_point.html
        """

        # True
        # =====
        for dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            t = torch.ones([1], dtype=dtype)
            eggs.assert_true(torch.is_floating_point(t))
            eggs.assert_true(t.is_floating_point())

        # False
        # =====
        int_t = torch.tensor([1], dtype=torch.int8)
        eggs.assert_false(
            torch.is_floating_point(int_t),
        )
        eggs.assert_false(
            int_t.is_floating_point(),
        )

        # Errors:
        # =======
        eggs.assert_raises(
            lambda: torch.is_floating_point([1j]),
            TypeError,
        )

    def test_is_nonzero(self):
        """torch.is_nonzero(input: Tensor)
        Also: `<tensor>.is_nonzero()`

        Returns true if the input is a single element tensor
        which is not equal to zero after type conversions.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.is_nonzero.html
        """
        # True:
        # =====
        for s in [1, [1], [1.0], [[1]], [[1.0]]]:
            t = torch.tensor(s)
            hamcrest.assert_that(t.numel(), 1)
            eggs.assert_true(
                torch.is_nonzero(t),
            )
            eggs.assert_true(
                t.is_nonzero(),
            )

        # False:
        # =====
        for s in [0, [0], [0.0], [[0]], [[0.0]]]:
            t = torch.tensor(s)
            hamcrest.assert_that(t.numel(), 1)
            eggs.assert_false(
                torch.is_nonzero(t),
            )
            eggs.assert_false(
                t.is_nonzero(),
            )

        # Errors:
        # =======
        # Throws RuntimeError if t.numel() != 1
        for s in [[], [1, 1]]:
            t = torch.tensor(s)
            hamcrest.assert_that(t.numel(), hamcrest.is_not(1))
            eggs.assert_raises(
                lambda: torch.is_nonzero(t),
                RuntimeError,
            )
            eggs.assert_raises(
                lambda: t.is_nonzero(),
                RuntimeError,
            )

        # Throws TypeError if input isn't a Tensor
        eggs.assert_raises(
            lambda: torch.is_nonzero("abc"),
            TypeError,
        )

    def test_default_dtype(self):
        """torch.set_default_dtype(input: Tensor)

        Set the default dtype for new tensors.
        Must be a floating point type.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html
        """
        original = torch.get_default_dtype()
        try:
            for dtype in [torch.float32, torch.float64, torch.bfloat16]:
                torch.set_default_dtype(dtype)
                hamcrest.assert_that(
                    torch.get_default_dtype(),
                    dtype,
                )

                t = torch.tensor([1])
                hamcrest.assert_that(t.dtype, dtype)

            # Errors:
            # =======
            # Throws RuntimeError if type is not floating point.
            eggs.assert_raises(
                lambda: torch.set_default_dtype(torch.int8),
                TypeError,
            )

        finally:
            torch.set_default_dtype(original)

    def test_numel(self):
        """torch.numel(input: Tensor)
        Also: `<tensor>.numel()`

        Returns the number of elements in the tensor.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.numel.html
        """
        for s in [1, [1], [[1]]]:
            t = torch.tensor(s)
            hamcrest.assert_that(torch.numel(t), 1)
            hamcrest.assert_that(t.numel(), 1)

        for s in [[1, 2, 3, 4], [[1, 2], [3, 4]], [[1], [2], [3], [4]]]:
            t = torch.tensor(s)
            hamcrest.assert_that(torch.numel(t), 3)
            hamcrest.assert_that(t.numel(), 3)

        # Errors:
        # =======
        # Throws RuntimeError if type is not floating point.
        eggs.assert_raises(
            lambda: torch.numel([1, 2]),
            TypeError,
        )

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

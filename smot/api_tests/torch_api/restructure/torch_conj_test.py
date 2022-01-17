import unittest

import torch

from smot.testlib import eggs, torch_eggs


class ConjTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.conj.html

    def test_conj(self):
        source = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        conj_view = torch.conj(source)

        torch_eggs.assert_views(source, conj_view)

        torch_eggs.assert_tensor(
            conj_view,
            torch.tensor([-1 - 1j, -2 - 2j, 3 + 3j]),
        )

        # This is a VIEW.
        eggs.assert_match(
            conj_view.data_ptr(),
            source.data_ptr(),
        )

        # modification of the source is visible.
        source[0] = 4 + 4j  # type: ignore
        torch_eggs.assert_tensor(
            source,
            torch.tensor([4 + 4j, -2 + 2j, 3 - 3j]),
        )
        torch_eggs.assert_tensor(
            conj_view,
            torch.tensor([4 - 4j, -2 - 2j, 3 + 3j]),
        )

        # NOTE: Per the docs, modification of the conj view may become illegal in the future.
        conj_view[0] = 5 - 5j  # type: ignore
        torch_eggs.assert_tensor(
            source,
            torch.tensor([5 + 5j, -2 + 2j, 3 - 3j]),
        )
        torch_eggs.assert_tensor(
            conj_view,
            torch.tensor([5 - 5j, -2 - 2j, 3 + 3j]),
        )

import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.poisson",
    ref="https://pytorch.org/docs/stable/generated/torch.poisson.html",
)
class PoissonTest(unittest.TestCase):
    def test_degenerate(self) -> None:
        torch_eggs.assert_tensor(
            torch.poisson(torch.tensor([])),
            [],
        )

        torch_eggs.assert_tensor(
            torch.poisson(torch.tensor([[]])),
            [[]],
        )

    def test_scalar_error(self) -> None:
        eggs.assert_raises(
            lambda: torch.poisson(0.0),  # type: ignore
            TypeError,
            "must be Tensor",
        )

    def test_stats(self) -> None:
        k = 2000
        rates = [1.0, 2.0, 4.5]
        with torch_eggs.reset_generator_seed():
            samples = torch.poisson(
                torch.tile(
                    torch.tensor(rates).unsqueeze(1),
                    (1, k),
                )
            )
            torch_eggs.assert_tensor_structure(
                samples,
                torch.empty(
                    [3, k],
                    dtype=torch.float,
                ),
            )

        for i, r in enumerate(rates):
            row = samples[i, ...]
            mean = torch.mean(row).item()
            var = torch.var(row).item()

            eggs.assert_close_to(
                mean,
                r,
                rtol=0.05,
            )
            eggs.assert_close_to(
                var,
                r,
                rtol=0.05,
            )

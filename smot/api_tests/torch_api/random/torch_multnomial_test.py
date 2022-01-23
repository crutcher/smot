import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.multinomial",
    ref="https://pytorch.org/docs/stable/generated/torch.multinomial.html",
)
class MultinomialTest(unittest.TestCase):
    def test_degenerate(self) -> None:
        eggs.assert_raises(
            lambda: torch.multinomial(
                torch.tensor([5.0]),
                0,
            ),
            RuntimeError,
            "cannot sample n_sample <= 0 samples",
        )

        eggs.assert_raises(
            lambda: torch.multinomial(
                torch.tensor(
                    [
                        [5.0, 3.0],
                        [1.0, 1.0],
                    ],
                ),
                0,
            ),
            RuntimeError,
            "cannot sample n_sample <= 0 samples",
        )

    def test_too_many_dims(self) -> None:
        eggs.assert_raises(
            lambda: torch.multinomial(
                torch.tensor(
                    [
                        [
                            [5.0, 3.0],
                        ],
                        [
                            [1.0, 1.0],
                        ],
                    ]
                ),
                0,
            ),
            RuntimeError,
            "prob_dist must be 1 or 2 dim",
        )

    def test_one_choice(self) -> None:
        torch_eggs.assert_tensor(
            torch.multinomial(
                torch.tensor([5.0]),
                1,
            ),
            [0],
        )

        torch_eggs.assert_tensor(
            torch.multinomial(
                torch.tensor(
                    # m-row matrix input
                    [
                        [5.0, 0.0],
                        [0.0, 1.0],
                    ],
                ),
                1,
            ),
            # m x num_samples matrix output
            [
                [0],
                [1],
            ],
        )

        eggs.assert_raises(
            lambda: torch.multinomial(
                torch.tensor([5.0]),
                3,
            ),
            RuntimeError,
            r"cannot sample n_sample > prob_dist.size\(-1\) samples without replacement",
        )

        torch_eggs.assert_tensor(
            torch.multinomial(
                torch.tensor([5.0]),
                3,
                replacement=True,
            ),
            [0, 0, 0],
        )

    def test_permute(self) -> None:
        torch_eggs.assert_tensor(
            torch.sort(
                torch.multinomial(
                    torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                    5,
                ),
            )[0],
            [0, 1, 2, 3, 4],
        )

    def test_as_bernoulli(self) -> None:
        with torch_eggs.with_generator_seed(1234):
            k = 1000
            for p in [0.2, 0.5, 0.83]:
                eggs.assert_match(
                    torch.sum(
                        # Sampling from a multinomial distribution of size 2
                        # is equivalent to a bernoulli distribution.
                        torch.multinomial(
                            torch.tensor([1 - p, p]),
                            k,
                            replacement=True,
                        )
                    ),
                    hamcrest.close_to(k * p, k * 0.05),
                )

import hamcrest
import pytest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class GeneratorTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.Generator.html"
    TARGET = torch.Generator

    def test_cpu(self) -> None:
        g = torch.Generator()
        eggs.assert_match(
            g.device,
            torch.device("cpu"),
        )

        g = torch.Generator("cpu")
        eggs.assert_match(
            g.device,
            torch.device("cpu"),
        )

    @pytest.mark.slow
    def test_cuda(self) -> None:
        if torch.cuda.is_available():
            g = torch.Generator("cuda")
            eggs.assert_match(
                g.device,
                torch.device("cuda"),
            )

    def test_initial_seed(self) -> None:
        g = torch.Generator()
        seed = g.initial_seed()

        reference = torch.rand([2, 3], generator=g)

        # Replay with same seed.
        g.manual_seed(seed)

        torch_eggs.assert_tensor(
            torch.rand([2, 3], generator=g),
            reference,
        )

    def test_manual_seed(self) -> None:
        seed = 123324523

        g = torch.Generator()
        g.manual_seed(seed)

        reference = torch.rand([2, 3], generator=g)

        # Replay with same seed.
        g.manual_seed(seed)

        torch_eggs.assert_tensor(
            torch.rand([2, 3], generator=g),
            reference,
        )

    def test_set_state(self) -> None:
        g = torch.Generator()
        state = g.get_state()

        reference = torch.rand([2, 3], generator=g)

        # Replay with same state.
        g.set_state(state)

        torch_eggs.assert_tensor(
            torch.rand([2, 3], generator=g),
            reference,
        )


class GlobalGeneratorTest(TorchApiTestCase):
    TARGETS = {
        torch.seed: "https://pytorch.org/docs/stable/generated/torch.seed.html",
        torch.initial_seed: "https://pytorch.org/docs/stable/generated/torch.initial_seed.html",
        torch.get_rng_state: "https://pytorch.org/docs/stable/generated/torch.get_rng_state.html",
        torch.set_rng_state: "https://pytorch.org/docs/stable/generated/torch.set_rng_state.html",
        torch.default_generator: "https://pytorch.org/docs/stable/torch.html#torch.torch.default_generator",
    }

    API_DOC = "https://pytorch.org/docs/stable/generated/torch.seed.html"
    TARGET = torch.seed

    def test_initial_seed(self) -> None:
        orig_seed = torch.initial_seed()

        # NOTE: you probably shouldn't use this.
        # this is the initial seed for the default generator,
        eggs.assert_match(
            orig_seed,
            torch.default_generator.initial_seed(),
        )

        state = torch.get_rng_state()

        # but it is not equivalent to the current seed:
        torch.rand([2, 3])

        # sampling from the generator changes the state:
        eggs.assert_match(
            torch.get_rng_state(),
            hamcrest.not_(torch_eggs.expect_tensor(state)),
        )

        # but does not change the seed.
        eggs.assert_match(
            orig_seed,
            torch.default_generator.initial_seed(),
        )

    def test_manual_seed(self) -> None:
        seed = 12312323
        torch.manual_seed(seed)

        reference = torch.rand([2, 3])

        # Replay with same seed.
        torch.manual_seed(seed)

        torch_eggs.assert_tensor(
            torch.rand([2, 3]),
            reference,
        )

        # Pull the default generator
        g = torch.default_generator

        # Replay with same seed, using the generator ref.
        g.manual_seed(seed)

        torch_eggs.assert_tensor(
            torch.rand([2, 3], generator=g),
            reference,
        )

    def test_set_state(self) -> None:
        state = torch.get_rng_state()

        reference = torch.rand([2, 3])

        # Replay with same state.
        torch.set_rng_state(state)

        torch_eggs.assert_tensor(
            torch.rand([2, 3]),
            reference,
        )

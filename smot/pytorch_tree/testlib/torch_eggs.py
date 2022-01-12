import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
import torch

from smot.pytorch_tree.testlib import torch_eggs
from smot.testlib import eggs

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


class TensorMatcher(BaseMatcher[torch.Tensor]):
    expected: torch.Tensor

    def __init__(self, expected):
        self.expected = torch.as_tensor(expected)

        if self.expected.is_sparse and not self.expected.is_coalesced():
            self.expected = self.expected.coalesce()

    def _matches(self, item) -> bool:
        if not self.expected.is_sparse:
            try:
                return torch.equal(item, self.expected)
            except RuntimeError:
                # thrown on dtype miss-match.
                return False

        # torch.equal() doesn't handle sparse tensors correctly.

        if not item.is_coalesced():
            # non-coalesced tensors do not have observable indices,
            # so we assume the user wanted to coalesce the values.
            item = item.coalesce()

        eggs.assert_match(
            item.device,
            self.expected.device,
        )
        eggs.assert_match(
            item.size(),
            self.expected.size(),
        )
        eggs.assert_match(
            item.dtype,
            self.expected.dtype,
        )
        eggs.assert_match(
            item.layout,
            self.expected.layout,
        )
        torch_eggs.assert_tensor(
            item.indices(),
            self.expected.indices(),
        )

        torch_eggs.assert_tensor(
            item.values(),
            self.expected.values(),
        )
        return True

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_tensor(expected) -> TensorMatcher:
    return TensorMatcher(expected)


def assert_tensor(actual, expected):
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )

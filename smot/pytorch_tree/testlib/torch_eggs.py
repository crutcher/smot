import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
import torch

from smot.pytorch_tree.testlib import torch_eggs
from smot.testlib import eggs


class TensorMatcher(BaseMatcher[torch.Tensor]):
    expected: torch.Tensor

    def __init__(self, expected):
        self.expected = torch.as_tensor(expected)

    def _matches(self, item) -> bool:
        try:
            if not self.expected.is_sparse:
                return torch.equal(item, self.expected)

            else:
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

        except RuntimeError:
            # thrown on dtype miss-match.
            return False

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_tensor(expected) -> TensorMatcher:
    return TensorMatcher(expected)


def assert_tensor(actual, expected):
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )

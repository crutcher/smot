import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
import torch


class TensorMatcher(BaseMatcher[torch.Tensor]):
    expected: torch.Tensor

    def __init__(self, expected):
        if torch.is_tensor(expected):
            self.expected = expected.clone().detach()
        else:
            self.expected = torch.tensor(expected)

    def _matches(self, item) -> bool:
        return torch.equal(item, self.expected)

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_tensor(expected) -> TensorMatcher:
    return TensorMatcher(expected)


def assert_tensor(actual, expected):
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )

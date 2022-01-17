import hamcrest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class ComplexTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.complex.html"
    TARGET = torch.complex

    def test_complex(self) -> None:
        torch_eggs.assert_tensor(
            torch.complex(
                torch.tensor([1.0, 2.0], dtype=torch.float32),
                torch.tensor([3.0, 4.0], dtype=torch.float32),
            ),
            torch.tensor(
                [1.0 + 3.0j, 2.0 + 4.0j],
                dtype=torch.complex64,
            ),
        )

    def test_complex_out(self) -> None:
        target = torch.tensor([1j, 3j], dtype=torch.complex64)
        original_data = target.data_ptr()

        # same size, no realloc.
        torch_eggs.assert_tensor(
            torch.complex(
                torch.tensor([1, 2], dtype=torch.float32),
                torch.tensor([3, 4], dtype=torch.float32),
                out=target,
            ),
            torch.tensor([1.0 + 3.0j, 2.0 + 4.0j]),
        )
        eggs.assert_match(
            target.data_ptr(),
            original_data,
        )

        with eggs.ignore_warnings():
            # smaller size, no realloc.
            torch_eggs.assert_tensor(
                torch.complex(
                    torch.tensor([1], dtype=torch.float32),
                    torch.tensor([3], dtype=torch.float32),
                    out=target,
                ),
                torch.tensor([1.0 + 3.0j]),
            )
            eggs.assert_match(
                target.data_ptr(),
                original_data,
            )

            # larger size, NEW data.
            torch_eggs.assert_tensor(
                torch.complex(
                    torch.tensor([1, 2, 3], dtype=torch.float32),
                    torch.tensor([3, 4, 5], dtype=torch.float32),
                    out=target,
                ),
                torch.tensor([1.0 + 3.0j, 2.0 + 4.0j, 3.0 + 5.0j]),
            )
            eggs.assert_match(
                target.data_ptr(),
                hamcrest.not_(original_data),
            )

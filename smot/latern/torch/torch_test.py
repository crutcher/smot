import unittest
import torch
import hamcrest


class TorchTest(unittest.TestCase):
    def test_is_tensor(self):
        # https://pytorch.org/docs/stable/generated/torch.is_tensor.html#torch.is_tensor
        hamcrest.assert_that(
            torch.is_tensor([1, 2]),
            False,
        )

if __name__ == '__main__':
    unittest.main()

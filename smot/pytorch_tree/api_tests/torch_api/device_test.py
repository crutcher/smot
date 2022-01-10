import unittest

import hamcrest
import pytest
import torch

from smot.testlib import eggs


class DeviceTest(unittest.TestCase):
    @pytest.mark.slow
    def test_device(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda"] + [
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            ]

        for d in devices:
            device = torch.device(d)
            hamcrest.assert_that(device, hamcrest.instance_of(torch.device))
            eggs.assert_match(str(device), d)


if __name__ == "__main__":
    unittest.main()

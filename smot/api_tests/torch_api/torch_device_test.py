import unittest

import hamcrest
import pytest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs


@api_link(
    target="torch.device",
    ref="https://pytorch.org/docs/stable/generated/torch.device.html",
)
class DeviceTest(unittest.TestCase):
    @pytest.mark.slow
    def test_device(self) -> None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda"] + [
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            ]

        for d in devices:
            device = torch.device(d)
            hamcrest.assert_that(device, hamcrest.instance_of(torch.device))
            eggs.assert_match(str(device), d)

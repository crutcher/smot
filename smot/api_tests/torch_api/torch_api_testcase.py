import typing
import unittest

import torch


class TorchApiTestCase(unittest.TestCase):
    API_DOC: str = "https://pytorch.org/docs/stable/torch.html"
    TARGET: typing.Any = None
    ALIAS_FOR: typing.Any = None

    @classmethod
    def target_name(cls):
        target = cls.TARGET
        name = target.__name__

        if name in torch.__all__:
            return f"torch.{name}"

        return target.__qualname__

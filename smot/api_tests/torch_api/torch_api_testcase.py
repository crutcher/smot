import typing
import unittest


class TorchApiTestCase(unittest.TestCase):
    API_DOC: str = "https://pytorch.org/docs/stable/torch.html"
    TARGET: typing.Any = None
    ALIAS_FOR: typing.Any = None

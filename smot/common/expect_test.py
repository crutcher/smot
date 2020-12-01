import unittest
import hamcrest

from smot.common import expect


class ExpectTest(unittest.TestCase):

  def test_true(self) -> None:
    expect.Expect.truthy(True)

    hamcrest.assert_that(
      hamcrest.calling(lambda: expect.Expect.truthy(False)),
      hamcrest.raises(AssertionError, "Value is not truthy\\."),
    )

    hamcrest.assert_that(
      hamcrest.calling(
        lambda: expect.Expect.truthy(
          False,
          msg="%(abc)s 123",
          cls=KeyError,
          abc="frog",
        )),
      hamcrest.raises(
        KeyError,
        "frog 123",
      ),
    )

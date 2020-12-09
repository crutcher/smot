import unittest

import hamcrest

from smot.common import expect
from smot.testing import hamcrest_funcs


class ExpectTest(unittest.TestCase):

  def test_is_truthy(self) -> None:
    for val in [True, 1, [1], "abc"]:
      expect.Expect.is_truthy(val)

    for val in [False, 0, [], ""]:
      hamcrest_funcs.assert_raises(
        lambda: expect.Expect.is_truthy(val),
        AssertionError,
        matching=hamcrest.has_string(f"Value is not truthy: {val}"),
      )

    hamcrest_funcs.assert_raises(
      lambda: expect.Expect.is_truthy(
        False,
        msg="%(abc)s 123",
        cls=KeyError,
        abc="frog",
      ),
      KeyError,
      "frog 123",
    )

  def test_is_falsey(self) -> None:
    for val in [False, 0, [], ""]:
      expect.Expect.is_falsey(val)

    for val in [True, 1, [1], "abc"]:
      hamcrest_funcs.assert_raises(
        lambda: expect.Expect.is_falsey(val),
        AssertionError,
        matching=hamcrest.has_string(f"Value is not falsey: {val}"),
      )

    hamcrest_funcs.assert_raises(
      lambda: expect.Expect.is_falsey(
        True,
        msg="%(abc)s 123",
        cls=KeyError,
        abc="frog",
      ),
      KeyError,
      "frog 123",
    )

  def test_is_eq(self) -> None:
    expect.Expect.is_eq("abc", "abc")

    hamcrest_funcs.assert_raises(
      hamcrest.calling(lambda: expect.Expect.is_eq("abc", 1)),
      AssertionError,
      r"Value \(abc\) != \(1\)",
    )

    hamcrest_funcs.assert_raises(
      lambda: expect.Expect.is_eq(
        "abc",
        "xyz",
        msg="%(actual)s %(expected)s %(fff)s",
        cls=KeyError,
        fff="frog",
      ),
      KeyError,
      "abc xyz frog",
    )

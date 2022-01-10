import unittest

import hamcrest
import testfixtures

from smot.common import expect
from smot.testlib import eggs


class ExpectTest(unittest.TestCase):
    def test_is_truthy(self) -> None:
        for val in [True, 1, [1], "abc"]:
            expect.Expect.is_truthy(val)

        for val in [False, 0, [], ""]:
            eggs.assert_raises(
                lambda: expect.Expect.is_truthy(val),
                AssertionError,
                matching=hamcrest.has_string(f"Value is not truthy: {val}"),
            )

        eggs.assert_raises(
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
            eggs.assert_raises(
                lambda: expect.Expect.is_falsey(val),
                AssertionError,
                matching=hamcrest.has_string(f"Value is not falsey: {val}"),
            )

        eggs.assert_raises(
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

        eggs.assert_raises(
            hamcrest.calling(lambda: expect.Expect.is_eq("abc", 1)),
            AssertionError,
            r"Value \(abc\) != \(1\)",
        )

        eggs.assert_raises(
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


class ExpectPathTest(unittest.TestCase):
    def test_is_file(self) -> None:
        with testfixtures.TempDirectory() as tempdir:
            p = tempdir.getpath("foo.txt")

            eggs.assert_raises(
                lambda: expect.ExpectPath.is_file(p),
                AssertionError,
                matching=hamcrest.has_string(f"Path ({p}) is not a file."),
            )

            eggs.assert_raises(
                lambda: expect.ExpectPath.is_file(
                    p,
                    msg="%(path)s %(zzz)s",
                    zzz="abc",
                ),
                AssertionError,
                matching=hamcrest.has_string(f"{p} abc"),
            )

            with open(p, "w") as f:
                f.write("abc")

            expect.ExpectPath.is_file(p)

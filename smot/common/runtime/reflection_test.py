import os.path
import unittest

import hamcrest
import sys

import smot
from smot.common.runtime import reflection, reflection_testlib

this_module = sys.modules[__name__]


class RepositorySourceRootTest(unittest.TestCase):

  def test_root(self) -> None:
    hamcrest.assert_that(
      reflection.repository_source_root(), hamcrest.is_(os.path.dirname(smot.__file__)))


class CallingModuleTest(unittest.TestCase):

  def test_simple(self) -> None:
    hamcrest.assert_that(
      reflection.calling_module(stack_depth=0),
      hamcrest.same_instance(this_module),
    )

  def test_depth(self) -> None:
    # (lambda: ...) is defined in this_module
    # apply() is defined in reflection_testlib
    # test_simple() is defined in this_module

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(0)),
      hamcrest.same_instance(this_module),
    )

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(1)),
      hamcrest.same_instance(reflection_testlib),
    )

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(2)),
      hamcrest.same_instance(this_module),
    )

  def test_too_many(self) -> None:
    hamcrest.assert_that(
      hamcrest.calling(lambda: reflection.calling_module(1000)).with_args(),
      hamcrest.raises(
        ValueError,
        "not deep enough",
      ),
    )


class ModuleDirectoryTest(unittest.TestCase):

  def test_simple(self) -> None:
    hamcrest.assert_that(
      reflection.module_directory(stack_depth=0),
      hamcrest.is_(os.path.dirname(this_module.__file__)),
    )

    hamcrest.assert_that(
      reflection.module_directory(this_module),
      hamcrest.is_(os.path.dirname(this_module.__file__)),
    )

    hamcrest.assert_that(
      reflection.module_directory(smot),
      hamcrest.is_(os.path.dirname(smot.__file__)),
    )

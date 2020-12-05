import os.path
import sys
import unittest

import hamcrest

import smot
from smot.common.runtime import reflection, reflection_testlib

this_module = sys.modules[__name__]


class RepositorySourceRootTest(unittest.TestCase):

  def test(self) -> None:
    hamcrest.assert_that(
      reflection.repository_source_root(),
      hamcrest.is_(os.path.dirname(os.path.dirname(smot.__file__))),
    )


class ThisModuleTest(unittest.TestCase):

  def test(self) -> None:
    hamcrest.assert_that(
      reflection.this_module(),
      hamcrest.same_instance(this_module),
    )


class CallingModuleTest(unittest.TestCase):

  def test_simple(self) -> None:
    hamcrest.assert_that(
      reflection.calling_module(stack_depth=0),
      hamcrest.same_instance(this_module),
    )

    hamcrest.assert_that(
      reflection.calling_module(module=reflection),
      hamcrest.same_instance(reflection),
    )

  def test_depth(self) -> None:
    # (lambda: ...) is defined in this_module
    # apply() is defined in reflection_testlib
    # test_simple() is defined in this_module

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(stack_depth=0)),
      hamcrest.same_instance(this_module),
    )

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(stack_depth=1)),
      hamcrest.same_instance(reflection_testlib),
    )

    hamcrest.assert_that(
      reflection_testlib.apply(lambda: reflection.calling_module(stack_depth=2)),
      hamcrest.same_instance(this_module),
    )

  def test_too_many(self) -> None:
    hamcrest.assert_that(
      hamcrest.calling(lambda: reflection.calling_module(stack_depth=1000)).with_args(),
      hamcrest.raises(
        ValueError,
        "not deep enough",
      ),
    )


class ModuleDirectoryTest(unittest.TestCase):

  def test(self) -> None:
    hamcrest.assert_that(
      reflection.module_directory(this_module),
      hamcrest.is_(os.path.dirname(this_module.__file__)),
    )

    hamcrest.assert_that(
      reflection.module_directory(smot),
      hamcrest.is_(os.path.dirname(smot.__file__)),
    )


class ModuleNameAsRelativePathTest(unittest.TestCase):

  def test(self) -> None:
    hamcrest.assert_that(
      reflection.module_name_as_relative_path(this_module),
      hamcrest.is_('smot/common/runtime/reflection_test'),
    )

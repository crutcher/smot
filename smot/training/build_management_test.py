import os
import unittest.mock

from smot.common.runtime import reflection, reflection_testlib
from smot.testing import hamcrest_funcs
from smot.training import build_management


class CacheRootTest(unittest.TestCase):

  def test(self) -> None:
    hamcrest_funcs.assert_match(
      build_management.model_build_dir(),
      reflection.repository_source_root() + '/build/models',
    )


class ModelBuildCache(unittest.TestCase):

  def test_module_path(self) -> None:
    hamcrest_funcs.assert_match(
      build_management.build_cache().model_path(name="foo/bar"),
      os.path.join(
        build_management.build_cache().build_root(),
        "foo/bar",
      ),
    )

    # Default, by stack reflection lookup.
    hamcrest_funcs.assert_match(
      build_management.build_cache().model_path(
        name="foo/bar",
        relative=True,
      ),
      os.path.join(
        build_management.build_cache().build_root(),
        reflection.module_name_as_relative_path(reflection.this_module()),
        "foo/bar",
      ),
    )

    # By module lookup.
    hamcrest_funcs.assert_match(
      build_management.build_cache().model_path(
        name="foo/bar",
        module=build_management,
      ),
      os.path.join(
        build_management.build_cache().build_root(),
        reflection.module_name_as_relative_path(build_management),
        "foo/bar",
      ),
    )

    # By module lookup.
    hamcrest_funcs.assert_match(
      reflection_testlib.apply(
        build_management.build_cache().model_path,
        name="foo/bar",
        relative=True,
        stack_depth=1,
      ),
      os.path.join(
        build_management.build_cache().build_root(),
        reflection.module_name_as_relative_path(reflection.this_module()),
        "foo/bar",
      ),
    )

  def test_save_simple(self) -> None:
    mock_model = unittest.mock.Mock()
    expected_path = os.path.join(
      build_management.build_cache().build_root(),
      'foo/bar',
    )
    hamcrest_funcs.assert_match(
      build_management.build_cache().save(
        model=mock_model,
        name="foo/bar",
      ),
      expected_path,
    )
    mock_model.save.assert_called_with(filepath=expected_path)

    # Default, by stack reflection lookup.
    mock_model = unittest.mock.Mock()
    expected_path = os.path.join(
      build_management.build_cache().build_root(),
      reflection.module_name_as_relative_path(reflection.this_module()),
      "foo/bar",
    )
    hamcrest_funcs.assert_match(
      build_management.build_cache().save(
        model=mock_model,
        name="foo/bar",
        relative=True,
      ),
      expected_path,
    )
    mock_model.save.assert_called_with(filepath=expected_path)

    # By module lookup.
    mock_model = unittest.mock.Mock()
    expected_path = os.path.join(
      build_management.build_cache().build_root(),
      reflection.module_name_as_relative_path(build_management),
      "foo/bar",
    )
    hamcrest_funcs.assert_match(
      build_management.build_cache().save(
        model=mock_model,
        name="foo/bar",
        module=build_management,
      ),
      expected_path,
    )
    build_management.build_cache().save(
      model=mock_model,
      name="foo/bar",
      module=build_management,
    )
    mock_model.save.assert_called_with(filepath=expected_path)

    # By module lookup.
    mock_model = unittest.mock.Mock()
    expected_path = os.path.join(
      build_management.build_cache().build_root(),
      reflection.module_name_as_relative_path(reflection.this_module()),
      "foo/bar",
    )
    hamcrest_funcs.assert_match(
      reflection_testlib.apply(
        build_management.build_cache().save,
        model=mock_model,
        name="foo/bar",
        relative=True,
        stack_depth=1,
      ),
      expected_path,
    )
    mock_model.save.assert_called_with(filepath=expected_path)

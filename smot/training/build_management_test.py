import os
import unittest.mock

import testfixtures

from smot.common.runtime import reflection, reflection_testlib
from smot.testlib import eggs
from smot.training import build_management


class CacheRootTest(unittest.TestCase):
    def test(self) -> None:
        eggs.assert_match(
            build_management.model_build_dir(),
            reflection.repository_source_root() + "/build/models",
        )


class ModelBuildTargetTest(unittest.TestCase):
    def test(self) -> None:
        target = build_management.ModelBuildTarget(
            build_root="foo",
            target_id="bar",
        )

        eggs.assert_match(target.build_root(), "foo")
        eggs.assert_match(target.target_id(), "bar")
        eggs.assert_match(target.model_save_path(), "foo/bar")

    def test_save(self) -> None:
        with testfixtures.TempDirectory() as temp_dir:
            target = build_management.ModelBuildTarget(
                build_root=temp_dir.getpath("foo"),
                target_id="bar",
            )

            mock_model = unittest.mock.Mock()
            expected_path = temp_dir.getpath("foo/bar")

            eggs.assert_match(
                target.save_model(model=mock_model),
                expected_path,
            )
            mock_model.save.assert_called_with(filepath=expected_path)

    def test_load(self) -> None:
        with testfixtures.TempDirectory() as temp_dir:
            target = build_management.ModelBuildTarget(
                build_root=temp_dir.getpath("foo"),
                target_id="bar",
            )

            mock_model = unittest.mock.Mock()
            mock_loader = unittest.mock.Mock(return_value=mock_model)

            eggs.assert_match(
                target.load_model(_loader=mock_loader),
                mock_model,
            )
            mock_loader.assert_called_with(filepath=target.model_save_path())


class ModelBuildCacheTest(unittest.TestCase):
    def test_targets(self) -> None:
        target = build_management.build_cache().target(name="foo/bar")
        eggs.assert_match(
            target.build_root(), build_management.build_cache().build_root()
        )
        eggs.assert_match(target.target_id(), "foo/bar")
        eggs.assert_match(
            target.model_save_path(),
            os.path.join(
                build_management.build_cache().build_root(),
                "foo/bar",
            ),
        )

        # Default, by stack reflection lookup.
        target = build_management.build_cache().target(
            name="foo/bar",
            relative=True,
        )
        eggs.assert_match(
            target.build_root(), build_management.build_cache().build_root()
        )
        eggs.assert_match(
            target.target_id(),
            os.path.join(
                reflection.module_name_as_relative_path(reflection.this_module()),
                "foo/bar",
            ),
        )
        eggs.assert_match(
            target.model_save_path(),
            os.path.join(
                build_management.build_cache().build_root(),
                reflection.module_name_as_relative_path(reflection.this_module()),
                "foo/bar",
            ),
        )

        # By module lookup.
        target = build_management.build_cache().target(
            name="foo/bar",
            module=build_management,
        )
        eggs.assert_match(
            target.build_root(), build_management.build_cache().build_root()
        )
        eggs.assert_match(
            target.target_id(),
            os.path.join(
                reflection.module_name_as_relative_path(build_management),
                "foo/bar",
            ),
        )
        eggs.assert_match(
            target.model_save_path(),
            os.path.join(
                build_management.build_cache().build_root(),
                reflection.module_name_as_relative_path(build_management),
                "foo/bar",
            ),
        )

        # By module lookup.
        # Use the testlib func to alter the stack.
        target = reflection_testlib.apply(
            build_management.build_cache().target,
            name="foo/bar",
            relative=True,
            stack_depth=1,
        )
        eggs.assert_match(
            target.build_root(), build_management.build_cache().build_root()
        )
        eggs.assert_match(
            target.target_id(),
            os.path.join(
                reflection.module_name_as_relative_path(reflection.this_module()),
                "foo/bar",
            ),
        )
        eggs.assert_match(
            target.model_save_path(),
            os.path.join(
                build_management.build_cache().build_root(),
                reflection.module_name_as_relative_path(reflection.this_module()),
                "foo/bar",
            ),
        )

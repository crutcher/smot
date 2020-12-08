import unittest

from smot.common.runtime import reflection
from smot.training import build_management


class CacheRootTest(unittest.TestCase):

  def test(self) -> None:
    self.assertEqual(
      build_management.model_build_dir(),
      reflection.repository_source_root() + '/build/models',
    )

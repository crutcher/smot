import unittest

from smot.common.runtime import reflection
from smot.training import model_cache


class CacheRootTest(unittest.TestCase):

  def test(self) -> None:
    self.assertEqual(
      model_cache.default_cache_root(),
      reflection.repository_source_root() + '/training/cache',
    )

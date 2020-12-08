import os
from types import ModuleType
from typing import Optional

import tensorflow as tf

from smot.common.runtime import reflection


def model_build_dir() -> str:
  """
  Default cache root.
  """
  return os.path.join(
    reflection.repository_source_root(),
    'build/models',
  )


class ModelBuildCache:
  """
  Handle for managing reading and writing the model cache.
  """
  _cache_root: str

  def __init__(self, *, cache_root: Optional[str] = None):
    if cache_root is None:
      cache_root = model_build_dir()

    self._cache_root = cache_root

  def model_path(
    self,
    *,
    name: str,
    module: Optional[ModuleType] = None,
    stack_depth: int = 1,
    makedirs: bool = False,
  ) -> str:
    """
    Model path.

    :param name: the model name.
    :param module: (optional) the module by reference.
    :param stack_depth: the stack depth to module, relative to the caller.
    :param makedirs: create the containing directories?
    :return: the path.
    """
    dir = os.path.join(
      self._cache_root,
      reflection.module_name_as_relative_path(
        reflection.calling_module(
          module=module,
          stack_depth=stack_depth + 1,
        ),
      ),
    )

    if makedirs:
      os.makedirs(dir, exist_ok=True)

    return os.path.join(dir, name)

  def save(
    self,
    model: tf.keras.Model,
    name: str,
    *,
    module: Optional[ModuleType] = None,
    stack_depth: int = 1,
  ) -> str:
    """
    Save a tensorflow keras model.

    :param model: the model.
    :param name: the model name.
    :param module: (optional) the module by reference.
    :param stack_depth: the stack depth to module, relative to the caller.
    :return: the path.
    """
    path = self.model_path(
      module=module,
      stack_depth=stack_depth,
      name=name,
      makedirs=True,
    )

    model.save(filepath=path)

    return path

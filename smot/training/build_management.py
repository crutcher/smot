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
  _build_root: str

  def __init__(self, *, cache_root: Optional[str] = None):
    if cache_root is None:
      cache_root = model_build_dir()

    self._build_root = cache_root

  def build_root(self) -> str:
    """The root directory of builds."""
    return self._build_root

  def model_path(
    self,
    *,
    name: str,
    relative: bool = False,
    module: Optional[ModuleType] = None,
    stack_depth: int = 0,
  ) -> str:
    """
    Model path.

    :param name: the model name.
    :param relative: should we extend the name relative to the module?
    :param module: (optional) the module by reference, implies ``relative``.
    :param stack_depth: the stack depth to module, relative to the caller.
    :return: the path.
    """
    relative_path = name

    if relative or module:
      relative_path = os.path.join(
        reflection.module_name_as_relative_path(
          reflection.calling_module(
            module=module,
            stack_depth=stack_depth + 1,
          ),
        ),
        relative_path,
      )

    return os.path.join(
      self._build_root,
      relative_path,
    )

  def save(
    self,
    *,
    model: tf.keras.Model,
    name: str,
    relative: bool = False,
    module: Optional[ModuleType] = None,
    stack_depth: int = 0,
  ) -> str:
    """
    Save a tensorflow keras model.

    :param model: the model.
    :param name: the model name.
    :param relative: should we extend the name relative to the module?
    :param module: (optional) the module by reference, implies ``relative``.
    :param stack_depth: the stack depth to module, relative to the caller.
    :return: the path.
    """
    path = self.model_path(
      module=module,
      relative=relative,
      stack_depth=stack_depth + 1,
      name=name,
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    model.save(filepath=path)

    return path


_model_build_cache: Optional[ModelBuildCache] = None


def build_cache() -> ModelBuildCache:
  """
  Return a default ModelBuildCache instance.

  :return: a ModelBuildCache.
  """
  global _model_build_cache
  if not _model_build_cache:
    _model_build_cache = ModelBuildCache()

  return _model_build_cache

import os
from types import ModuleType
from typing import Callable, Optional

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


class ModelBuildTarget:
  """
  Build target for paths to model files.
  """
  _build_root: str
  _target_id: str

  def __init__(
    self,
    *,
    build_root: str,
    target_id: str,
  ):
    """
    :param build_root: the build root.
    :param target_id: the target id.
    """
    self._build_root = build_root
    self._target_id = target_id

  def build_root(self) -> str:
    """The build root of this target."""
    return self._build_root

  def target_id(self) -> str:
    """The id of this target."""
    return self._target_id

  def model_save_path(self) -> str:
    """
    Path to the model save files.
    """
    return os.path.join(
      self._build_root,
      self._target_id,
    )

  def save_model(
    self,
    model: tf.keras.Model,
  ) -> str:
    """
    Save a tensorflow keras model.

    :param model: the model.
    :return: the path.
    """
    path = self.model_save_path()

    os.makedirs(os.path.dirname(path), exist_ok=True)

    model.save(filepath=path)

    return path

  def load_model(
    self,
    *,
    _loader: Callable[..., tf.keras.Model] = tf.keras.models.load_model,
  ) -> tf.keras.Model:
    """
    Load a tf.keras.Model from the target files.

    :param _loader: test param for over-riding the load function.
    :return: a Model.
    """
    return _loader(filepath=self.model_save_path())


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

  def target(
    self,
    *,
    name: str,
    relative: bool = False,
    module: Optional[ModuleType] = None,
    stack_depth: int = 0,
  ) -> ModelBuildTarget:
    """
    Build a ModelBuildTarget.

    :param name: the model name.
    :param relative: should we extend the name relative to the module?
    :param module: (optional) the module by reference, implies ``relative``.
    :param stack_depth: the stack depth to module, relative to the caller.
    :return: the ModelBuildTarget.
    """
    target_id = name

    if relative or module:
      target_id = os.path.join(
        reflection.module_name_as_relative_path(
          reflection.calling_module(
            module=module,
            stack_depth=stack_depth + 1,
          ),
        ),
        target_id,
      )

    return ModelBuildTarget(
      build_root=self._build_root,
      target_id=target_id,
    )


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

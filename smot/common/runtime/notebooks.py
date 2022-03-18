import os

from smot.common.runtime import build_paths, reflection


def notebook_dir() -> str:
    return os.path.dirname(os.path.realpath("__file__"))


def notebook_relative_dir() -> str:
    return notebook_dir().removeprefix(
        reflection.repository_source_root() + "/",
    )

def output_path(name: str) -> str:
    d = os.path.join(
        build_paths.build_root(),
        notebook_relative_dir(),
    )
    os.makedirs(d, exist_ok=True)
    return os.path.join(
        d,
        name,
    )

def output_dir(name: str) -> str:
    d = output_path(name)
    os.makedirs(d, exist_ok=True)
    return d


def experiment_log_dir() -> str:
    d = output_dir('tensorboard')
    os.environ['experiment_log_dir'] = d
    return d
import os

from smot.common.expect import ExpectPath
from smot.common.runtime.build_paths import data_root


def kaggle_data_root() -> str:
    """
    The configured source root for kaggle data.
    """
    return os.path.join(
        data_root(),
        "kaggle",
    )


def kaggle_data_path(competition: str, file: str) -> str:
    """
    The path to a kaggle file.

    :param competition: the competition module name.
    :param file: the file.
    :return: the full path.
    :raises AssertionError: if the file is not present.
    """
    return ExpectPath.is_file(
        os.path.join(
            kaggle_data_root(),
            competition,
            file,
        )
    )


import os

from smot.common.expect import ExpectPath
from smot.common.runtime import reflection

def data_root() -> str:
    """
    The configured source root for cached data.
    """
    return os.path.join(
        reflection.repository_source_root(),
        "build/data_cache",
    )

def notebook_output_path(name: str) -> str:
    d = os.path.join(
        reflection.repository_source_root(),
        "build/notebook",
    )
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
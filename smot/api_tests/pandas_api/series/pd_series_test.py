from typing import cast
import unittest

import hamcrest
import numpy as np
import pandas as pd

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, np_eggs


@api_link(
    target="pandas.Series",
    ref="https://pandas.pydata.org/docs/reference/api/pandas.Series.html",
)
class SeriesTest(unittest.TestCase):
    def test_simple(self) -> None:
        s = pd.Series(["a", "b", "c"])

        eggs.assert_match(
            s.axes,
            hamcrest.contains_exactly(
                hamcrest.contains_exactly(
                    *pd.RangeIndex(0, 3),
                )
            ),
        )
        np_eggs.assert_ndarray_equals(
            cast(np.ndarray, s.values),
            np.array(["a", "b", "c"], dtype=object),
        )

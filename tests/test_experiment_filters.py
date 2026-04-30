from __future__ import annotations

import pandas as pd

from dvgrpsig.reporting import filter_formal_tracks


def test_filter_formal_tracks_removes_toy_rows():
    frame = pd.DataFrame(
        [
            {"track": "toy", "value": 1},
            {"track": "GPV-S", "value": 2},
            {"track": "GPV-M", "value": 3},
        ]
    )
    filtered = filter_formal_tracks(frame)
    assert filtered["track"].tolist() == ["GPV-S", "GPV-M"]


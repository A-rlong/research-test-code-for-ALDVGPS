from __future__ import annotations

import pandas as pd


FORMAL_TRACKS = {"GPV-S", "GPV-M", "GPV-L"}


def filter_formal_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["track"].isin(FORMAL_TRACKS)].reset_index(drop=True)


import pandas as pd
from app import _enforce_capacity

class DummyLoc:
    def __init__(self, side_id, section, tote_id):
        self.side_id = side_id
        self.section = section
        self.tote_id = tote_id

if __name__ == "__main__":
    # simple sanity test of capacity enforcement
    loc_cols = ["side_id", "section", "tote_id"]
    locs = [DummyLoc("A001", "A001-AA", f"A001-AA-{i:03d}") for i in range(1, 3)]
    # assign 25 SKUs to first loc, 10 to second -> expect redistribution to cap at 20 in first, 15 in second
    assign = {i: locs[0] for i in range(1, 26)}
    assign.update({i: locs[1] for i in range(26, 36)})
    df = _enforce_capacity(assign, locs, loc_cols, capacity=20)
    print("rows:", len(df))
    print(df.head())

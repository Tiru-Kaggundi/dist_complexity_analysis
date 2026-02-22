"""
Stitch DGCI&S state-of-origin Excel files (FY25: Apr 2024 - Mar 2025) into one CSV.

- Source folder: dgcis_dist_am25 (contains .xls files, possibly in subfolders).
- First row in each file is description (skip).
- Second row is the header.
- No columns dropped: full period "April, 24 To March, 25" is the FY25 total.
- Output: dists_am25_full.csv in the same folder (dgcis_dist_am25).
"""

import pandas as pd
from pathlib import Path


def load_one_file(path: Path) -> pd.DataFrame:
    """Load one .xls file: skip row 0, use row 1 as header, data from row 2 onward."""
    df = pd.read_excel(path, header=None)
    if df.shape[0] < 2:
        return pd.DataFrame()
    headers = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[1].tolist()]
    data = df.iloc[2:].copy()
    data.columns = headers
    return data


def main():
    base = Path(__file__).parent / "dgcis_dist_am25"
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")

    xls_files = sorted(base.rglob("*.xls"))
    if not xls_files:
        raise FileNotFoundError(f"No .xls files found under {base}")

    frames = []
    for path in xls_files:
        df = load_one_file(path)
        if df.empty:
            continue
        frames.append(df)
        print(f"  Loaded {path.name}: {len(df)} rows")

    if not frames:
        raise ValueError("No data loaded from any file.")

    ref_columns = list(frames[0].columns)
    combined = []
    for df in frames:
        use_cols = [c for c in ref_columns if c in df.columns]
        combined.append(df[use_cols].copy())
    out_df = pd.concat(combined, axis=0, ignore_index=True)

    out_path = base / "dists_am25_full.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Written {len(out_df)} rows, {len(out_df.columns)} columns to {out_path}")
    return out_path


if __name__ == "__main__":
    main()

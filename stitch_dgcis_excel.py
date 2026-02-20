"""
Stitch DGCIS state-of-origin Excel files into one CSV.

- First row in each file is description (skip).
- Second row is the header.
- Remove two columns: December 2025 exports (INR and USD).
- Concatenate all files with the same header.
- Output: dists_2025_full.csv in the same folder.
"""

import os
import pandas as pd
from pathlib import Path

# Exact column names to drop (December 2025 INR and USD)
DECEMBER_2025_COLUMNS = [
    "December, 25 Value(INR)",
    "December, 25 Value(US $)",
]
# Normalized (strip) for matching
DECEMBER_2025_STRIP = [c.strip() for c in DECEMBER_2025_COLUMNS]


def load_one_file(path: Path) -> pd.DataFrame:
    """Load one .xls file: skip row 0, use row 1 as header, drop Dec 2025 columns."""
    df = pd.read_excel(path, header=None)
    if df.shape[0] < 2:
        return pd.DataFrame()
    # Row 1 as headers
    headers = [str(x).strip() if pd.notna(x) else "" for x in df.iloc[1].tolist()]
    # Data from row 2 onward
    data = df.iloc[2:].copy()
    data.columns = headers
    # Drop December 2025 columns (match after strip)
    to_drop = [c for c in data.columns if (c and c.strip() in DECEMBER_2025_STRIP) or c in DECEMBER_2025_COLUMNS]
    if to_drop:
        data = data.drop(columns=to_drop, errors="ignore")
    return data


def main():
    base = Path(__file__).parent / "dgcis_stateoforigin1771519863052"
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")

    xls_files = sorted([f for f in os.listdir(base) if f.endswith(".xls")])
    if not xls_files:
        raise FileNotFoundError(f"No .xls files in {base}")

    frames = []
    for fname in xls_files:
        path = base / fname
        df = load_one_file(path)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No data loaded from any file.")

    # Use first file's columns as reference so all have same column order
    ref_columns = list(frames[0].columns)
    combined = []
    for df in frames:
        # Keep only columns that exist; order like reference
        use_cols = [c for c in ref_columns if c in df.columns]
        combined.append(df[use_cols].copy())
    out_df = pd.concat(combined, axis=0, ignore_index=True)

    out_path = base / "dists_2025_full.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Written {len(out_df)} rows, {len(out_df.columns)} columns to {out_path}")
    return out_path


if __name__ == "__main__":
    main()

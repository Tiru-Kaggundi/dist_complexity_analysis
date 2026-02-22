"""
Build dists_am25_base.csv from dgcis_dist_am25/dists_am25_full.csv for the PCI/ECI pipeline.
Keeps State, District, HS Code, Commodity Description; renames FY25 columns to Export_USD, Export_INR.
Output: dists_am25_base.csv (project root).
"""

from pathlib import Path
import pandas as pd


def main():
    base = Path(__file__).parent
    src = base / "dgcis_dist_am25" / "dists_am25_full.csv"
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}. Run stitch_dgcis_am25.py first.")
    df = pd.read_csv(src)
    # FY25 full-period columns
    usd_col = "April, 24 To March, 25 Value(US $)"
    inr_col = "April, 24 To March, 25 Value(INR)"
    if usd_col not in df.columns:
        raise ValueError(f"Column not found: {usd_col}. Columns: {list(df.columns)}")
    out = df[["State", "District", "HS Code", "Commodity Description", usd_col, inr_col]].copy()
    out = out.rename(columns={usd_col: "Export_USD", inr_col: "Export_INR"})
    # HS Code as 4-digit string for PCI merge (OEC PCI uses HS4 e.g. "0101")
    out["HS Code"] = out["HS Code"].astype(str).str.zfill(4)
    out["Export_USD"] = pd.to_numeric(out["Export_USD"], errors="coerce").fillna(0)
    out["Export_INR"] = pd.to_numeric(out["Export_INR"], errors="coerce").fillna(0)
    out_path = base / "dists_am25_base.csv"
    out.to_csv(out_path, index=False)
    print(f"Written {len(out)} rows to {out_path}")
    print(f"  Total Export_USD (billion): {out['Export_USD'].sum() / 1e9:.2f}")
    return out_path


if __name__ == "__main__":
    main()

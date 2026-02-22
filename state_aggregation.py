"""
Aggregate district-level trade to state level and recompute PCI, RCA, ECI.
Builds states_2025_full.csv with State, HS Code, exports, PCI, RCA, ECI, ECI_std,
VIIRS_Mean (state mean), and GSDP_per_capita.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Reuse PCI/ECI logic from district pipeline
from pci_eci_calculator import (
    load_oec_pci_data,
    add_pci_column,
    calculate_rca,
    calculate_eci,
)


def aggregate_districts_to_states(district_path: str) -> pd.DataFrame:
    """
    Aggregate district-level trade to state level: sum Export_USD and Export_INR
    by State × HS Code; keep first Commodity Description.
    """
    print("Aggregating districts to states...")
    df = pd.read_csv(district_path)
    # Sum by State and HS Code (drop District)
    agg = df.groupby(["State", "HS Code"], as_index=False).agg({
        "Export_USD": "sum",
        "Export_INR": "sum",
        "Commodity Description": "first",
    })
    print(f"  State×HS4 rows: {len(agg)}")
    return agg


def add_state_mpi(
    state_trade: pd.DataFrame,
    mpi_path: str = "data/niti_mpi/mpi_state_data.csv",
) -> pd.DataFrame:
    """
    Add state-level MPI (Multidimensional Poverty Index headcount ratio) from
    NITI Aayog MPI data (e.g. NFHS-5 2019-21). Column MPI_HCR.
    """
    mpi_file = Path(mpi_path)
    if not mpi_file.exists():
        state_trade["MPI_HCR"] = np.nan
        return state_trade
    print("Adding state MPI (NITI Aayog)...")
    mpi = pd.read_csv(mpi_file)
    mpi["State"] = mpi["State"].astype(str).str.strip().str.upper()
    mpi = mpi[["State", "MPI_HCR"]].drop_duplicates(subset=["State"])
    state_upper = state_trade["State"].astype(str).str.strip().str.upper()
    state_trade = state_trade.copy()
    state_trade["MPI_HCR"] = state_upper.map(mpi.set_index("State")["MPI_HCR"])
    return state_trade


def add_state_viirs_and_gsdp(
    state_trade: pd.DataFrame,
    district_path: str,
    gsdp_path: str = "data/gsdp_per_capita.csv",
) -> pd.DataFrame:
    """
    Add state-level VIIRS_Mean (mean of district VIIRS in that state) and
    GSDP_per_capita from data/gsdp_per_capita.csv.
    """
    print("Adding state-level VIIRS and GSDP per capita...")
    dist = pd.read_csv(district_path)
    if "VIIRS_Mean" in dist.columns:
        viirs = dist.groupby("State")["VIIRS_Mean"].mean().reset_index()
        state_trade = state_trade.merge(viirs, on="State", how="left")
    else:
        state_trade["VIIRS_Mean"] = np.nan

    gsdp_path = Path(gsdp_path)
    if gsdp_path.exists():
        gsdp = pd.read_csv(gsdp_path)
        gsdp["State"] = gsdp["State"].astype(str).str.strip().str.upper()
        pc_col = [c for c in gsdp.columns if "gsdp" in c.lower() or "per_capita" in c.lower()]
        if pc_col:
            gsdp = gsdp.rename(columns={pc_col[0]: "GSDP_per_capita"})
        state_upper = state_trade["State"].astype(str).str.strip().str.upper()
        gsdp_lookup = gsdp.set_index("State")["GSDP_per_capita"]
        state_trade["GSDP_per_capita"] = state_upper.map(gsdp_lookup)
    else:
        state_trade["GSDP_per_capita"] = np.nan
    return state_trade


def add_eci_column_state(base_df: pd.DataFrame, eci_df: pd.DataFrame) -> pd.DataFrame:
    """Add ECI/ECI_std to state-level base by merging on State."""
    eci_df = eci_df.rename(columns={"District": "State"}) if "District" in eci_df.columns else eci_df
    # Remove duplicate columns (e.g. State twice when district_col='State')
    eci_df = eci_df.loc[:, ~eci_df.columns.duplicated()]
    cols = ["State", "ECI", "ECI_std"] if "ECI_std" in eci_df.columns else ["State", "ECI"]
    cols = [c for c in cols if c in eci_df.columns]
    result = base_df.merge(eci_df[cols], on="State", how="left")
    return result


def build_states_2025_full(
    district_path: str = "dists_2025_full.csv",
    output_path: str = "states_2025_full.csv",
    pci_path: Optional[str] = None,
    gsdp_path: str = "data/gsdp_per_capita.csv",
    min_export_usd: float = 1000,
    min_products_for_eci: int = 5,
    use_oec_api: bool = False,
) -> pd.DataFrame:
    """
    Build state-level dataset: aggregate trade, add PCI, compute RCA and ECI
    at state level, add state VIIRS mean and GSDP per capita.
    """
    print("="*60)
    print("Building states_2025_full.csv")
    print("="*60)

    # 1) Aggregate to state × HS4
    state_trade = aggregate_districts_to_states(district_path)
    # Filter min export (same as district pipeline)
    state_trade = state_trade[state_trade["Export_USD"] >= min_export_usd].copy()
    state_trade["HS Code"] = state_trade["HS Code"].astype(str).str.strip().str.zfill(4)
    state_trade = state_trade[state_trade["HS Code"].str.match(r"^\d{4}$")].copy()
    # Ensure HS Code is string for PCI merge
    state_trade["HS Code"] = state_trade["HS Code"].astype(str)

    # 2) Load PCI and add to state trade
    pci_df = load_oec_pci_data(path=pci_path or "data/oec_pci.csv", use_api=use_oec_api, cache_path="data/oec_pci.csv")
    pci_df["HS4"] = pci_df["HS4"].astype(str).str.strip().str.zfill(4)
    state_trade = add_pci_column(state_trade, pci_df)

    # 3) RCA at state level (entity = State)
    state_trade = calculate_rca(
        state_trade,
        district_col="State",
        hs4_col="HS Code",
        value_col="Export_USD",
    )

    # 4) ECI at state level
    eci_df = calculate_eci(
        state_trade,
        pci_df,
        district_col="State",
        hs4_col="HS Code",
        min_products=min_products_for_eci,
        standardize=True,
    )
    # ECI_df has State (we passed district_col='State') and State column from state_map - may have duplicate State
    if "District" in eci_df.columns:
        eci_df = eci_df.rename(columns={"District": "State"})
    if "State" not in eci_df.columns and "District" in eci_df.columns:
        eci_df["State"] = eci_df["District"]
    state_trade = add_eci_column_state(state_trade, eci_df)

    # 5) State VIIRS and GSDP
    state_trade = add_state_viirs_and_gsdp(state_trade, district_path, gsdp_path=gsdp_path)

    # 6) State MPI (NITI Aayog)
    state_trade = add_state_mpi(state_trade, mpi_path="data/niti_mpi/mpi_state_data.csv")

    # 7) Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    state_trade.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(state_trade)} rows)")
    return state_trade


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="dists_2025_full.csv", help="District-level CSV")
    p.add_argument("--output", default="states_2025_full.csv", help="Output state-level CSV")
    p.add_argument("--gsdp", default="data/gsdp_per_capita.csv", help="GSDP per capita CSV")
    p.add_argument("--no-api", action="store_true", help="Do not use OEC API for PCI")
    args = p.parse_args()
    build_states_2025_full(
        district_path=args.input,
        output_path=args.output,
        gsdp_path=args.gsdp,
        use_oec_api=not args.no_api,
    )

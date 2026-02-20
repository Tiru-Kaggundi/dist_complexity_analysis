"""
Plot ECI vs MPI at district level with regression line, equation, and R².
Uses districts that have both ECI and MPI; merges MPI from data/niti_mpi if needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from rapidfuzz import fuzz, process


def _normalize(s: pd.Series) -> pd.Series:
    """Upper-case, strip, collapse spaces."""
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def load_district_eci_mpi(
    trade_path: str = "dists_2025_full.csv",
    mpi_path: str = "data/niti_mpi/mpi_district_data.csv",
    fuzzy_threshold: int = 85,
) -> pd.DataFrame:
    """
    Load trade data (district-level ECI) and MPI data, merge on State+District.
    Exact match first, then fuzzy match for remaining; return one row per district
    with ECI and MPI_HCR (only where both exist).
    """
    trade = pd.read_csv(trade_path)
    eci_dist = (
        trade.groupby(["State", "District"], as_index=False)
        .agg(ECI=("ECI", "first"), ECI_std=("ECI_std", "first"), VIIRS_Mean=("VIIRS_Mean", "first"))
        .dropna(subset=["ECI"])
    )
    eci_dist["State_norm"] = _normalize(eci_dist["State"])
    eci_dist["District_norm"] = _normalize(eci_dist["District"])

    mpi = pd.read_csv(mpi_path)
    mpi["State_norm"] = _normalize(mpi["State"])
    mpi["District_norm"] = _normalize(mpi["District"])

    # Exact merge
    merged = eci_dist.merge(
        mpi[["State_norm", "District_norm", "MPI_HCR"]],
        on=["State_norm", "District_norm"],
        how="left",
    )
    unmatched = merged["MPI_HCR"].isna()
    if unmatched.any():
        # Fuzzy match within state for unmatched
        for idx in merged.index[unmatched]:
            row = merged.loc[idx]
            s, d = row["State_norm"], row["District_norm"]
            if pd.isna(s) or pd.isna(d):
                continue
            cand = mpi[mpi["State_norm"] == s]
            if cand.empty:
                continue
            match = process.extractOne(d, cand["District_norm"].tolist(), scorer=fuzz.token_sort_ratio)
            if match and match[1] >= fuzzy_threshold:
                mrow = cand[cand["District_norm"] == match[0]].iloc[0]
                merged.loc[idx, "MPI_HCR"] = mrow["MPI_HCR"]

    merged = merged.dropna(subset=["ECI", "MPI_HCR"])
    return merged[["State", "District", "ECI", "ECI_std", "MPI_HCR", "VIIRS_Mean"]]


def _plot_x_vs_mpi(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: str,
    title: str,
) -> None:
    """Scatter x_col vs MPI_HCR with regression line, equation, and R²."""
    df_plot = df.dropna(subset=[x_col, "MPI_HCR"])
    x = df_plot[x_col].values
    y = df_plot["MPI_HCR"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_sq = r_value ** 2
    line_y = slope * x + intercept
    eq_text = f"MPI_HCR = {intercept:.2f} {slope:+.2f} × {x_col}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.6, s=25, c="steelblue", edgecolors="none", label="Districts")
    ax.plot(x, line_y, color="coral", linewidth=2, label="Best fit")
    ax.set_xlabel(x_label)
    ax.set_ylabel("MPI Headcount Ratio (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} districts, R² = {r_sq:.4f}, equation: {eq_text}")


def plot_eci_vs_mpi(
    df: pd.DataFrame,
    out_path: str = "eci_vs_mpi.png",
    title: str = "District-level ECI vs MPI (Headcount Ratio)",
) -> None:
    """Scatter ECI vs MPI_HCR with regression line, equation, and R²."""
    _plot_x_vs_mpi(df, "ECI", "Economic Complexity Index (ECI)", out_path, title)


def plot_eci_std_vs_mpi(
    df: pd.DataFrame,
    out_path: str = "eci_std_vs_mpi.png",
    title: str = "District-level ECI (standardized) vs MPI (Headcount Ratio)",
) -> None:
    """Scatter ECI_std vs MPI_HCR with regression line, equation, and R²."""
    _plot_x_vs_mpi(df, "ECI_std", "ECI (standardized)", out_path, title)


def load_district_eci_viirs(
    trade_path: str = "dists_2025_full.csv",
) -> pd.DataFrame:
    """
    Load trade data (district-level ECI and VIIRS_Mean).
    Return one row per district with ECI and VIIRS_Mean (only where both exist).
    """
    trade = pd.read_csv(trade_path)
    eci_dist = (
        trade.groupby(["State", "District"], as_index=False)
        .agg(ECI=("ECI", "first"), ECI_std=("ECI_std", "first"), VIIRS_Mean=("VIIRS_Mean", "first"))
        .dropna(subset=["ECI", "VIIRS_Mean"])
    )
    return eci_dist[["State", "District", "ECI", "ECI_std", "VIIRS_Mean"]]


def plot_eci_vs_viirs(
    df: pd.DataFrame,
    out_path: str = "eci_vs_viirs.png",
    title: str = "District-level ECI vs VIIRS Mean Luminosity",
) -> None:
    """Scatter ECI vs VIIRS_Mean with regression line, equation, and R²."""
    _plot_x_vs_y(df, "ECI", "VIIRS_Mean", "Economic Complexity Index (ECI)", 
                 "VIIRS Mean Luminosity", out_path, title)


def plot_eci_std_vs_viirs(
    df: pd.DataFrame,
    out_path: str = "eci_std_vs_viirs.png",
    title: str = "District-level ECI (standardized) vs VIIRS Mean Luminosity",
) -> None:
    """Scatter ECI_std vs VIIRS_Mean with regression line, equation, and R²."""
    _plot_x_vs_y(df, "ECI_std", "VIIRS_Mean", "ECI (standardized)", 
                 "VIIRS Mean Luminosity", out_path, title)


def _plot_x_vs_y(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    out_path: str,
    title: str,
) -> None:
    """Scatter x_col vs y_col with regression line, equation, and R²."""
    df_plot = df.dropna(subset=[x_col, y_col])
    x = df_plot[x_col].values
    y = df_plot[y_col].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_sq = r_value ** 2
    line_y = slope * x + intercept
    eq_text = f"{y_col} = {intercept:.2f} {slope:+.2f} × {x_col}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.6, s=25, c="steelblue", edgecolors="none", label="Districts")
    ax.plot(x, line_y, color="coral", linewidth=2, label="Best fit")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right")

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} districts, R² = {r_sq:.4f}, equation: {eq_text}")


def main():
    base = Path(__file__).parent
    trade_path = base / "dists_2025_full.csv"
    mpi_path = base / "data" / "niti_mpi" / "mpi_district_data.csv"

    if not trade_path.exists():
        raise FileNotFoundError(f"Trade data not found: {trade_path}")
    if not mpi_path.exists():
        raise FileNotFoundError(f"MPI data not found: {mpi_path}")

    # ECI vs MPI plots
    df_mpi = load_district_eci_mpi(str(trade_path), str(mpi_path))
    print(f"Districts with both ECI and MPI: {len(df_mpi)}")
    plot_eci_vs_mpi(df_mpi, out_path=str(base / "eci_vs_mpi.png"))
    plot_eci_std_vs_mpi(df_mpi, out_path=str(base / "eci_std_vs_mpi.png"))
    
    # ECI vs VIIRS plots
    df_viirs = load_district_eci_viirs(str(trade_path))
    print(f"\nDistricts with both ECI and VIIRS: {len(df_viirs)}")
    plot_eci_vs_viirs(df_viirs, out_path=str(base / "eci_vs_viirs.png"))
    plot_eci_std_vs_viirs(df_viirs, out_path=str(base / "eci_std_vs_viirs.png"))


if __name__ == "__main__":
    main()

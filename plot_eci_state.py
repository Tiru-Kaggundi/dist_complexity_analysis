"""
State-level plots: ECI vs GSDP per capita, ECI vs VIIRS, ECI vs log(VIIRS).
Uses states_2025_full.csv (one row per state per HS4; we take one row per state for ECI/ECI_std/VIIRS/GSDP).
ECI vs GSDP plots: linear fit, bubble size = total export USD, state labels, bubble-size legend.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
try:
    from rapidfuzz import fuzz, process
except ImportError:
    process = None
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# States excluded as outliers for ECI vs GSDP "_removed" plots
EXCLUDED_STATES_GSDP = {
    "TRIPURA",
    "NAGALAND",
    "MEGHALAYA",
    "ANDAMAN AND NICOBAR ISLANDS",
    "SIKKIM",
    "CHANDIGARH",
}

# Widescreen 16:9, high resolution for blog
FIG_SIZE_16_9 = (12, 6.75)
DPI_HIGH = 400

# Bubble size: area ∝ annual export USD. s in points² so area ∝ s. s = (export_usd/1e9) * AREA_PER_BN
# Cap set so largest state (~116 bn) is not clipped; 116*70 = 8120
AREA_PER_BN = 70   # points² per billion USD
BUBBLE_SIZE_MIN, BUBBLE_SIZE_MAX = 25, 9000
BUBBLE_REF_EXPORTS = [10e9]  # legend: single circle for 10 bn USD


def _normalize(s: pd.Series) -> pd.Series:
    """Upper-case, strip, collapse spaces."""
    return s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)


def compute_state_mpi_export_weighted(
    trade_path: str,
    mpi_path: str,
    fuzzy_threshold: int = 85,
) -> pd.DataFrame:
    """
    State-level MPI as export-weighted average of district MPIs.
    Returns DataFrame with State (UPPER), MPI_export_weighted.
    """
    trade = pd.read_csv(trade_path)
    dist_export = (
        trade.groupby(["State", "District"], as_index=False)["Export_USD"]
        .sum()
        .dropna(subset=["Export_USD"])
    )
    dist_export = dist_export[dist_export["Export_USD"] > 0]
    dist_export["State_norm"] = _normalize(dist_export["State"])
    dist_export["District_norm"] = _normalize(dist_export["District"])

    mpi = pd.read_csv(mpi_path)
    if "MPI_HCR" not in mpi.columns:
        # Try to find MPI column
        for c in mpi.columns:
            if "mpi" in c.lower() or "hcr" in c.lower():
                mpi = mpi.rename(columns={c: "MPI_HCR"})
                break
    mpi["State_norm"] = _normalize(mpi["State"])
    mpi["District_norm"] = _normalize(mpi["District"])

    merged = dist_export.merge(
        mpi[["State_norm", "District_norm", "MPI_HCR"]],
        on=["State_norm", "District_norm"],
        how="left",
    )
    # Fuzzy match for unmatched districts
    if process is not None and merged["MPI_HCR"].isna().any():
        unmatched = merged["MPI_HCR"].isna()
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

    merged = merged.dropna(subset=["MPI_HCR", "Export_USD"])
    # Export-weighted average by state: MPI_state = sum(MPI_d * Export_d) / sum(Export_d)
    state_mpi = (
        merged.groupby("State", group_keys=False)
        .apply(lambda g: (g["MPI_HCR"] * g["Export_USD"]).sum() / g["Export_USD"].sum(), include_groups=False)
        .reset_index(name="MPI_export_weighted")
    )
    state_mpi["State"] = state_mpi["State"].astype(str).str.strip().str.upper()
    return state_mpi


def _bubble_sizes_and_legend(export_usd: np.ndarray):
    """Return sizes (area ∝ export USD) and legend refs. Matplotlib s = area in points²."""
    # area ∝ export: s = (export_usd / 1e9) * AREA_PER_BN, clipped
    sizes = np.clip(
        (np.asarray(export_usd, dtype=float) / 1e9) * AREA_PER_BN,
        BUBBLE_SIZE_MIN,
        BUBBLE_SIZE_MAX,
    )
    ref_sizes = [min((ex / 1e9) * AREA_PER_BN, BUBBLE_SIZE_MAX) for ex in BUBBLE_REF_EXPORTS]
    return sizes, BUBBLE_REF_EXPORTS, ref_sizes


def load_state_level(states_path: str = "states_2025_full.csv") -> pd.DataFrame:
    """Load state-level dataset: one row per state with ECI, ECI_std, VIIRS_Mean, GSDP_per_capita, Total_Export_USD."""
    df = pd.read_csv(states_path)
    # Total export per state (sum over HS codes)
    total_exports = df.groupby("State")["Export_USD"].sum().reset_index()
    total_exports = total_exports.rename(columns={"Export_USD": "Total_Export_USD"})
    # One row per state for ECI, VIIRS, GSDP
    agg = df.groupby("State", as_index=False).agg({
        "ECI": "first",
        "ECI_std": "first",
        "VIIRS_Mean": "first",
        "GSDP_per_capita": "first",
    })
    agg = agg.merge(total_exports, on="State", how="left")
    return agg


def _short_state_name(state: str) -> str:
    """Shorten state name for labels (e.g. ANDHRA PRADESH -> Andhra Pradesh, TAMIL NADU -> Tamil Nadu)."""
    return state.title()


def _add_state_labels(ax, df_plot: pd.DataFrame, x_col: str, y_col: str, scatter_artist=None, adjust_overlap: bool = True):
    """Annotate each point with state name; use adjustText to avoid overlap if available."""
    x_vals = df_plot[x_col].values
    y_vals = df_plot[y_col].values
    texts = []
    for i, row in df_plot.iterrows():
        t = ax.annotate(
            _short_state_name(row["State"]),
            (row[x_col], row[y_col]),
            fontsize=7,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )
        texts.append(t)
    if adjust_overlap and HAS_ADJUST_TEXT and texts:
        kwargs = dict(
            ax=ax,
            x=x_vals,
            y=y_vals,
            force_text=(0.6, 1.0),
            force_pull=(0.01, 0.02),
            expand=(1.2, 1.5),
            max_move=(100, 100),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
        if scatter_artist is not None:
            kwargs["objects"] = [scatter_artist]
        adjust_text(texts, **kwargs)


def _plot_eci_vs_gsdp(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: str,
    title: str,
) -> None:
    """Scatter x (ECI or ECI_std) vs GSDP with linear fit, bubble size = total export USD, state labels, size legend."""
    y_col = "GSDP_per_capita"
    df_plot = df.dropna(subset=[x_col, y_col, "Total_Export_USD"]).copy()
    df_plot = df_plot[df_plot["Total_Export_USD"] > 0]
    if len(df_plot) == 0:
        print(f"Skipping {out_path}: no valid data")
        return
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    # Linear fit
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_sq = r_value ** 2
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    eq_text = f"GSDP = {intercept:.0f} {slope:+.0f}×x"

    export_usd = df_plot["Total_Export_USD"].values
    sizes, ref_exports, ref_sizes = _bubble_sizes_and_legend(export_usd)

    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    sc = ax.scatter(x, y, s=sizes, alpha=0.6, c="steelblue", edgecolors="navy", linewidths=0.8, zorder=2)
    ax.plot(x_line, y_line, color="coral", linewidth=2, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("GSDP per capita (₹)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _add_state_labels(ax, df_plot, x_col, y_col, scatter_artist=sc)

    legend_handles = [
        plt.scatter([], [], s=sz, c="steelblue", edgecolors="navy", alpha=0.6, label=f"{int(ex/1e9)} bn USD")
        for ex, sz in zip(ref_exports, ref_sizes)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_HIGH)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} states, R² = {r_sq:.4f}, equation: {eq_text}")


def _plot_eci_vs_log_gsdp(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: str,
    title: str,
) -> None:
    """Scatter x (ECI or ECI_std) vs log(GSDP) with linear fit, bubble size = total export USD, state labels, size legend."""
    df_plot = df.dropna(subset=[x_col, "GSDP_per_capita", "Total_Export_USD"]).copy()
    df_plot = df_plot[df_plot["GSDP_per_capita"] > 0]
    df_plot = df_plot[df_plot["Total_Export_USD"] > 0]
    df_plot["log_GSDP_per_capita"] = np.log(df_plot["GSDP_per_capita"])
    y_col = "log_GSDP_per_capita"
    if len(df_plot) == 0:
        print(f"Skipping {out_path}: no valid data")
        return
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_sq = r_value ** 2
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    eq_text = f"log(GSDP) = {intercept:.3f} {slope:+.3f}×x"

    export_usd = df_plot["Total_Export_USD"].values
    sizes, ref_exports, ref_sizes = _bubble_sizes_and_legend(export_usd)

    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    sc = ax.scatter(x, y, s=sizes, alpha=0.6, c="steelblue", edgecolors="navy", linewidths=0.8, zorder=2)
    ax.plot(x_line, y_line, color="coral", linewidth=2, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("log(GSDP per capita, ₹)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _add_state_labels(ax, df_plot, x_col, y_col, scatter_artist=sc)

    legend_handles = [
        plt.scatter([], [], s=sz, c="steelblue", edgecolors="navy", alpha=0.6, label=f"{int(ex/1e9)} bn USD")
        for ex, sz in zip(ref_exports, ref_sizes)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_HIGH)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} states, R² = {r_sq:.4f}, equation: {eq_text}")


def _plot_eci_vs_mpi_state(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: str,
    title: str,
) -> None:
    """Scatter x (ECI or ECI_std) vs state export-weighted MPI; same style as GSDP plots."""
    y_col = "MPI_export_weighted"
    df_plot = df.dropna(subset=[x_col, y_col, "Total_Export_USD"]).copy()
    df_plot = df_plot[df_plot["Total_Export_USD"] > 0]
    if len(df_plot) == 0:
        print(f"Skipping {out_path}: no valid data")
        return
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_sq = r_value ** 2
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    eq_text = f"MPI = {intercept:.2f} {slope:+.2f}×x"

    export_usd = df_plot["Total_Export_USD"].values
    sizes, ref_exports, ref_sizes = _bubble_sizes_and_legend(export_usd)

    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    sc = ax.scatter(x, y, s=sizes, alpha=0.6, c="steelblue", edgecolors="navy", linewidths=0.8, zorder=2)
    ax.plot(x_line, y_line, color="coral", linewidth=2, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("MPI Headcount Ratio (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _add_state_labels(ax, df_plot, x_col, y_col, scatter_artist=sc)

    legend_handles = [
        plt.scatter([], [], s=sz, c="steelblue", edgecolors="navy", alpha=0.6, label=f"{int(ex/1e9)} bn USD")
        for ex, sz in zip(ref_exports, ref_sizes)
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)  # downward slope: legend bottom-left

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_HIGH)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} states, R² = {r_sq:.4f}, equation: {eq_text}")


def _plot_eci_vs_viirs(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    out_path: str,
    title: str,
    eq_label: str,
    filter_viirs_outlier: bool = True,
) -> None:
    """ECI vs VIIRS (or log VIIRS): linear fit, bubble = total export USD, state labels, 1B/10B legend. Optionally drop VIIRS < -3."""
    x_col = "ECI"
    x_label = "Economic Complexity Index (ECI)"
    df_plot = df.dropna(subset=[x_col, y_col, "Total_Export_USD"]).copy()
    df_plot = df_plot[df_plot["Total_Export_USD"] > 0]
    if filter_viirs_outlier and "VIIRS_Mean" in df_plot.columns:
        # Remove outlier with VIIRS_Mean < -3
        before = len(df_plot)
        df_plot = df_plot[df_plot["VIIRS_Mean"] >= -3]
        dropped = before - len(df_plot)
        if dropped > 0:
            print(f"  VIIRS: dropped {dropped} state(s) with VIIRS_Mean < -3")
    if len(df_plot) == 0:
        print(f"Skipping {out_path}: no valid data")
        return
    x = df_plot[x_col].values
    y = df_plot[y_col].values
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r_sq = r_value ** 2
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    eq_text = f"{eq_label} = {intercept:.3f} {slope:+.3f}×ECI"

    export_usd = df_plot["Total_Export_USD"].values
    sizes, ref_exports, ref_sizes = _bubble_sizes_and_legend(export_usd)

    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    sc = ax.scatter(x, y, s=sizes, alpha=0.6, c="steelblue", edgecolors="navy", linewidths=0.8, zorder=2)
    ax.plot(x_line, y_line, color="coral", linewidth=2, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _add_state_labels(ax, df_plot, x_col, y_col, scatter_artist=sc)

    legend_handles = [
        plt.scatter([], [], s=sz, c="steelblue", edgecolors="navy", alpha=0.6, label=f"{int(ex/1e9)} bn USD")
        for ex, sz in zip(ref_exports, ref_sizes)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    textstr = f"{eq_text}\n$R^2$ = {r_sq:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_HIGH)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  N = {len(df_plot)} states, R² = {r_sq:.4f}, equation: {eq_text}")


def main():
    base = Path(__file__).parent
    states_path = base / "states_2025_full.csv"
    if not states_path.exists():
        raise FileNotFoundError(f"State data not found: {states_path}. Run state_aggregation.py first.")
    df = load_state_level(str(states_path))
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    print(f"States with ECI: {df['ECI'].notna().sum()}, with GSDP: {df['GSDP_per_capita'].notna().sum()}, with VIIRS: {df['VIIRS_Mean'].notna().sum()}")

    # State-level MPI for ECI vs MPI plots: prefer NITI Aayog report (mpi_state_data.csv), else export-weighted from districts
    mpi_state_report_path = base / "data" / "niti_mpi" / "mpi_state_data.csv"
    trade_path = base / "dists_2025_full.csv"
    mpi_district_path = base / "data" / "niti_mpi" / "mpi_district_data.csv"
    if mpi_state_report_path.exists():
        state_mpi = pd.read_csv(mpi_state_report_path)
        state_mpi["State"] = state_mpi["State"].astype(str).str.strip().str.upper()
        state_mpi = state_mpi.rename(columns={"MPI_HCR": "MPI_export_weighted"})[["State", "MPI_export_weighted"]]
        df = df.merge(state_mpi, on="State", how="left")
        n_mpi = df["MPI_export_weighted"].notna().sum()
        print(f"States with MPI (from NITI report): {n_mpi}")
        mpi_label = "MPI (Headcount Ratio, %)"
    elif trade_path.exists() and mpi_district_path.exists():
        state_mpi = compute_state_mpi_export_weighted(str(trade_path), str(mpi_district_path))
        df = df.merge(state_mpi, on="State", how="left")
        n_mpi = df["MPI_export_weighted"].notna().sum()
        print(f"States with MPI (export-weighted from districts): {n_mpi}")
        mpi_label = "MPI (export-weighted, %)"
    else:
        mpi_label = None
    if mpi_label and df["MPI_export_weighted"].notna().any():
        _plot_eci_vs_mpi_state(
            df, "ECI",
            "Economic Complexity Index (ECI)",
            str(base / "state_eci_vs_mpi.png"),
            f"State-level ECI vs {mpi_label}",
        )
        _plot_eci_vs_mpi_state(
            df, "ECI_std",
            "ECI (standardized)",
            str(base / "state_eci_std_vs_mpi.png"),
            f"State-level ECI (standardized) vs {mpi_label}",
        )
        df_removed = df[~df["State"].str.upper().isin(EXCLUDED_STATES_GSDP)].copy()
        _plot_eci_vs_mpi_state(
            df_removed, "ECI",
            "Economic Complexity Index (ECI)",
            str(base / "state_eci_vs_mpi_removed.png"),
            f"State-level ECI vs {mpi_label} (outliers removed)",
        )
        _plot_eci_vs_mpi_state(
            df_removed, "ECI_std",
            "ECI (standardized)",
            str(base / "state_eci_std_vs_mpi_removed.png"),
            f"State-level ECI (standardized) vs {mpi_label} (outliers removed)",
        )
    elif not mpi_label:
        print("Skipping state ECI vs MPI plots (no mpi_state_data.csv or district MPI/trade data)")

    # ECI vs GSDP per capita (linear fit, bubble = total export USD, state labels, size legend)
    _plot_eci_vs_gsdp(
        df, "ECI",
        "Economic Complexity Index (ECI)",
        str(base / "state_eci_vs_gsdp.png"),
        "State-level ECI vs GSDP per capita",
    )
    _plot_eci_vs_gsdp(
        df, "ECI_std",
        "ECI (standardized)",
        str(base / "state_eci_std_vs_gsdp.png"),
        "State-level ECI (standardized) vs GSDP per capita",
    )
    # Same ECI vs GSDP plots with outlier states removed
    df_removed = df[~df["State"].str.upper().isin(EXCLUDED_STATES_GSDP)].copy()
    _plot_eci_vs_gsdp(
        df_removed, "ECI",
        "Economic Complexity Index (ECI)",
        str(base / "state_eci_vs_gsdp_removed.png"),
        "State-level ECI vs GSDP per capita (outliers removed)",
    )
    _plot_eci_vs_gsdp(
        df_removed, "ECI_std",
        "ECI (standardized)",
        str(base / "state_eci_std_vs_gsdp_removed.png"),
        "State-level ECI (standardized) vs GSDP per capita (outliers removed)",
    )
    # ECI vs log(GSDP per capita), outliers removed
    _plot_eci_vs_log_gsdp(
        df_removed, "ECI",
        "Economic Complexity Index (ECI)",
        str(base / "state_eci_vs_log_gsdp_removed.png"),
        "State-level ECI vs log(GSDP per capita) (outliers removed)",
    )
    _plot_eci_vs_log_gsdp(
        df_removed, "ECI_std",
        "ECI (standardized)",
        str(base / "state_eci_std_vs_log_gsdp_removed.png"),
        "State-level ECI (standardized) vs log(GSDP per capita) (outliers removed)",
    )
    # ECI vs VIIRS (outlier VIIRS < -3 removed; bubbles + state labels; 1B/10B legend)
    _plot_eci_vs_viirs(
        df, y_col="VIIRS_Mean", y_label="VIIRS Mean Luminosity",
        out_path=str(base / "state_eci_vs_viirs.png"),
        title="State-level ECI vs VIIRS Mean Luminosity",
        eq_label="VIIRS",
        filter_viirs_outlier=True,
    )
    # ECI vs log VIIRS (same: drop VIIRS < -3, then take log of positive values)
    df_viirs = df.dropna(subset=["ECI", "VIIRS_Mean", "Total_Export_USD"]).copy()
    df_viirs = df_viirs[df_viirs["VIIRS_Mean"] >= -3]
    df_viirs = df_viirs[df_viirs["VIIRS_Mean"] > 0]
    df_viirs["log_VIIRS_Mean"] = np.log(df_viirs["VIIRS_Mean"])
    _plot_eci_vs_viirs(
        df_viirs, y_col="log_VIIRS_Mean", y_label="log(VIIRS Mean Luminosity)",
        out_path=str(base / "state_eci_vs_log_viirs.png"),
        title="State-level ECI vs log(VIIRS Mean Luminosity)",
        eq_label="log(VIIRS)",
        filter_viirs_outlier=False,  # already filtered above
    )


if __name__ == "__main__":
    main()

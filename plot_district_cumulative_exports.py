"""
Cumulative exports from districts vs number of districts (FY25).
Uses dgcis_dist_am25/dists_am25_full.csv (Apr 2024 - Mar 2025). Districts sorted by export (highest first),
cumulative USD billions on Y, with annotations for 80% contribution and tail.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Same widescreen/res as state plots
FIG_SIZE_16_9 = (12, 6.75)
DPI_HIGH = 200


def load_district_exports(path: str = None) -> pd.DataFrame:
    """Sum export USD by (State, District), sort descending."""
    if path is None:
        base = Path(__file__).parent
        path = str(base / "dgcis_dist_am25" / "dists_am25_full.csv")
    df = pd.read_csv(path)
    # USD column: prefer FY25 full-period (April to March), else Export_USD, else first US $ column
    usd_col = None
    for c in df.columns:
        if "April" in c and "March" in c and ("US $" in c or "USD" in c.upper()):
            usd_col = c
            break
    if usd_col is None:
        for c in df.columns:
            if c == "Export_USD":
                usd_col = c
                break
            if "USD" in c.upper() or "US $" in c:
                usd_col = c
                break
    if usd_col is None:
        raise ValueError(f"No USD column found. Columns: {list(df.columns)}")
    dist = (
        df.groupby(["State", "District"], as_index=False)[usd_col]
        .sum()
        .sort_values(usd_col, ascending=False)
        .reset_index(drop=True)
    )
    dist = dist.rename(columns={usd_col: "Export_USD"})
    dist["rank"] = np.arange(1, len(dist) + 1)
    dist["cum_export_usd"] = dist["Export_USD"].cumsum()
    return dist


def main():
    base = Path(__file__).parent
    data_path = base / "dgcis_dist_am25" / "dists_am25_full.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"District data not found: {data_path}. Run stitch_dgcis_am25.py first.")
    print("Using FY25 district data:", data_path.relative_to(base))
    dist = load_district_exports(str(data_path))
    total_usd = dist["Export_USD"].sum()
    total_b = total_usd / 1e9
    n_dist = len(dist)
    cum_b = dist["cum_export_usd"].values / 1e9
    rank = dist["rank"].values

    # Annotations: 80% of total, and tail contributing <1%
    target_80 = 0.80 * total_usd
    idx_80 = (dist["cum_export_usd"] >= target_80).idxmax()
    n_80 = dist.loc[idx_80, "rank"]
    cum_80_b = dist.loc[idx_80, "cum_export_usd"] / 1e9

    target_99 = 0.99 * total_usd
    idx_99 = (dist["cum_export_usd"] >= target_99).idxmax()
    n_99 = dist.loc[idx_99, "rank"]
    tail_count = n_dist - n_99  # districts that together contribute <1%

    # One marker per district (single continuous line, no skip)
    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    ax.plot(rank, cum_b, "b-", linewidth=1.5, marker="o", markersize=3, markevery=1)
    ax.set_xlabel("Number of Districts", fontsize=12)
    ax.set_ylabel("Cumulative Exports (USD Billions)", fontsize=12)
    ax.set_title("Cumulative Exports from Districts Vs Number of Districts - FY25 (Apr 2024 - Mar 2025)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0, max(rank) * 1.02)
    # Y-axis: full cumulative range so total is visible (use data max, not a cap)
    y_max = cum_b.max()
    ax.set_ylim(0, y_max * 1.02)
    ax.grid(True, alpha=0.3)

    # 80% annotation: vertical line at n_80, horizontal to cum_80_b
    ax.axvline(n_80, color="red", linestyle="--", linewidth=1)
    ax.axhline(cum_80_b, color="red", linestyle="--", linewidth=1)
    ax.annotate(
        f"{int(n_80)} districts contribute\naround 80% of total exports",
        xy=(n_80, cum_80_b),
        xytext=(n_80 + 80, cum_80_b - total_b * 0.15),
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Tail <1% annotation: vertical at n_99, horizontal at cum there; place text inside plot (e.g. lower-right area)
    cum_99_b = dist.loc[idx_99, "cum_export_usd"] / 1e9
    ax.axvline(n_99, color="red", linestyle="--", linewidth=1)
    ax.axhline(cum_99_b, color="red", linestyle="--", linewidth=1)
    # Position text box inside plot: to the right of the curve, in the lower half
    ax.annotate(
        f"From here on, the next {int(tail_count)} districts cumulatively\ncontribute less than 1% of total annual exports of India",
        xy=(n_99, cum_99_b),
        xytext=(n_99 + 60, total_b * 0.25),
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Note on districts with no exports (if we assume ~750 total districts in India)
    approx_total_districts = 770  # approximate number of districts in India (2020s)
    no_export = max(0, approx_total_districts - n_dist)
    if no_export > 0:
        ax.text(
            0.98,
            0.02,
            f"No exports registered from around {no_export} districts",
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    out_path = base / "district_cumulative_exports_2025.png"
    fig.savefig(out_path, dpi=DPI_HIGH)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"  Total: {total_b:.1f} USD billions across {n_dist} districts")
    print(f"  {int(n_80)} districts → ~80% of exports; over {int(tail_count)} districts → <1% cumulatively")


if __name__ == "__main__":
    main()

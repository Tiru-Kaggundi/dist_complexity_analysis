"""
Build a district–sector cluster table from the enriched trade data.

Input:
- dists_am25_enriched.csv (from pci_eci_calculator.process_and_enrich_dataset)
  with columns including:
    - State, District
    - HS Code (4-digit HS4 as string)
    - Export_USD, Export_INR
    - PCI, ECI

Output:
- cluster_maps/district_sector_clusters.csv with columns:
    State, District, sector, export_usd, export_inr, avg_pci, eci

Each row is a (district, sector) combination, where sector is defined by HS2
prefix groupings in cluster_config.SECTORS.
"""

from pathlib import Path
from typing import List

import pandas as pd

from cluster_maps.cluster_config import (
    DISTRICT_EXPORTS_ENRICHED,
    DISTRICT_SECTOR_CLUSTERS,
    SECTORS,
)


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            f"- Run the PCI/ECI pipeline first to generate it.\n"
            f"  Example:\n"
            f"    python pci_eci_calculator.py --input dists_am25_base.csv "
            f"--output dists_am25_enriched.csv --no-api --pci-file data/oec_pci.csv"
        )


def build_district_sector_clusters(
    input_path: Path = DISTRICT_EXPORTS_ENRICHED,
    output_path: Path = DISTRICT_SECTOR_CLUSTERS,
) -> Path:
    """
    Aggregate enriched district–product data into district–sector clusters.
    """
    _require_file(input_path, "Enriched district export data")

    print(f"Loading enriched district exports from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")

    required_cols: List[str] = [
        "State",
        "District",
        "HS Code",
        "Export_USD",
        "Export_INR",
        "PCI",
        "ECI",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # Normalise HS code and derive HS2 prefix.
    df["HS Code"] = df["HS Code"].astype(str).str.strip().str.zfill(4)
    df["HS2"] = df["HS Code"].str[:2]

    # Ensure numeric types for exports.
    df["Export_USD"] = pd.to_numeric(df["Export_USD"], errors="coerce").fillna(0.0)
    df["Export_INR"] = pd.to_numeric(df["Export_INR"], errors="coerce").fillna(0.0)

    # ECI is at district level; summarise once per district.
    eci_summary = (
        df.groupby(["State", "District"], as_index=False)["ECI"]
        .first()
        .rename(columns={"ECI": "eci"})
    )

    sector_frames = []

    for key, spec in SECTORS.items():
        hs2_prefixes = spec.get("hs2_prefixes", [])
        label = spec.get("label", key)
        if not hs2_prefixes:
            print(f"Skipping sector '{key}' (no HS2 prefixes defined).")
            continue

        mask = df["HS2"].isin(hs2_prefixes)
        sub = df[mask].copy()
        print(f"Sector '{label}' ({key}): {len(sub)} rows after HS2 filter.")
        if sub.empty:
            continue

        # Aggregate exports by district for this sector.
        agg = (
            sub.groupby(["State", "District"], as_index=False)[
                ["Export_USD", "Export_INR"]
            ]
            .sum()
            .rename(
                columns={
                    "Export_USD": "export_usd",
                    "Export_INR": "export_inr",
                }
            )
        )

        # Export-weighted average PCI per district-sector (if PCI present).
        sub_valid = sub.dropna(subset=["PCI"])
        if not sub_valid.empty:
            weighted_num = (
                sub_valid.assign(
                    _num=lambda d: d["PCI"] * d["Export_USD"],
                    _den=lambda d: d["Export_USD"],
                )
                .groupby(["State", "District"], as_index=False)[["_num", "_den"]]
                .sum()
            )
            weighted_num["avg_pci"] = weighted_num["_num"] / weighted_num["_den"]
            weighted_num = weighted_num[["State", "District", "avg_pci"]]
            agg = agg.merge(weighted_num, on=["State", "District"], how="left")
        else:
            agg["avg_pci"] = float("nan")

        agg["sector"] = key
        sector_frames.append(agg)

    if not sector_frames:
        raise ValueError("No sector data was generated; check SECTORS config and HS codes.")

    result = pd.concat(sector_frames, axis=0, ignore_index=True)
    # Attach ECI for each district.
    result = result.merge(eci_summary, on=["State", "District"], how="left")

    # Store columns in a consistent order.
    result = result[
        ["State", "District", "sector", "export_usd", "export_inr", "avg_pci", "eci"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Written {len(result)} rows to {output_path}")
    return output_path


def main() -> None:
    build_district_sector_clusters()


if __name__ == "__main__":
    main()


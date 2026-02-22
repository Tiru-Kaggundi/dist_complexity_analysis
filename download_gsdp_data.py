"""
Download or update state-wise GSDP per capita (RBI / latest).

RBI Handbook of Statistics on Indian States, Table 19: Per Capita Net State
Domestic Product (Current Prices).
Source: https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23468
Direct XLSX: https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/19T%5F11122025B8CC230E4A34431999B4D6A107707BCA.XLSX

Uses 2024-25 where available, else 2023-24, else 2022-23 (Gujarat).
"""

from pathlib import Path
import pandas as pd

# RBI Table 19 – Per Capita NSDP (Current Prices)
# 2024-25 where available; 2023-24 for A&N, Chandigarh, Goa, Ladakh, Manipur, Mizoram, Nagaland, Sikkim; 2022-23 for Gujarat
# Source: RBI Table 19, Dec 11, 2025
RBI_TABLE19 = [
    ("ANDAMAN AND NICOBAR ISLANDS", 276_000, "2023-24"),
    ("ANDHRA PRADESH", 266_240, "2024-25"),
    ("ARUNACHAL PRADESH", 246_813, "2024-25"),
    ("ASSAM", 159_185, "2024-25"),
    ("BIHAR", 69_321, "2024-25"),
    ("CHANDIGARH", 453_457, "2023-24"),
    ("CHHATTISGARH", 162_870, "2024-25"),
    ("DELHI", 493_024, "2024-25"),
    ("GOA", 585_953, "2023-24"),
    ("GUJARAT", 272_451, "2022-23"),  # 2023-24/2024-25 not in table
    ("HARYANA", 353_182, "2024-25"),
    ("HIMACHAL PRADESH", 256_137, "2024-25"),
    ("JAMMU AND KASHMIR", 154_826, "2024-25"),
    ("JHARKHAND", 116_663, "2024-25"),
    ("KARNATAKA", 380_906, "2024-25"),
    ("KERALA", 308_338, "2024-25"),
    ("LADAKH", 242_360, "2023-24"),
    ("MADHYA PRADESH", 152_615, "2024-25"),
    ("MAHARASHTRA", 309_340, "2024-25"),
    ("MANIPUR", 119_938, "2023-24"),
    ("MEGHALAYA", 157_141, "2024-25"),
    ("MIZORAM", 234_996, "2023-24"),
    ("NAGALAND", 154_828, "2023-24"),
    ("ODISHA", 168_966, "2024-25"),
    ("PUDUCHERRY", 281_478, "2024-25"),
    ("PUNJAB", 221_197, "2024-25"),
    ("RAJASTHAN", 185_053, "2024-25"),
    ("SIKKIM", 587_743, "2023-24"),
    ("TAMIL NADU", 361_619, "2024-25"),
    ("TELANGANA", 387_623, "2024-25"),
    ("TRIPURA", 192_842, "2024-25"),
    ("UTTAR PRADESH", 108_572, "2024-25"),
    ("UTTARAKHAND", 274_064, "2024-25"),
    ("WEST BENGAL", 163_467, "2024-25"),
]


def get_gsdp_path() -> Path:
    return Path(__file__).parent / "data" / "gsdp_per_capita.csv"


def save_rbi_table19_csv(out_path: Path) -> pd.DataFrame:
    """Save RBI Table 19 per capita NSDP to CSV (2024-25 / 2023-24 / 2022-23)."""
    df = pd.DataFrame(RBI_TABLE19, columns=["State", "GSDP_per_capita", "Year"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_2024 = (df["Year"] == "2024-25").sum()
    n_2023 = (df["Year"] == "2023-24").sum()
    print(f"Saved GSDP per capita for {len(df)} states to {out_path}")
    print(f"  Year: 2024-25 = {n_2024}, 2023-24 = {n_2023}, 2022-23 = 1 (Gujarat)")
    return df


def load_gsdp_per_capita(path: Path = None) -> pd.DataFrame:
    """Load state-wise GSDP per capita; normalize State to UPPER."""
    path = path or get_gsdp_path()
    if not path.exists():
        save_rbi_table19_csv(path)
    df = pd.read_csv(path)
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.strip().str.upper()
    for c in df.columns:
        if ("gsdp" in c.lower() or "per_capita" in c.lower()) and c != "GSDP_per_capita":
            df = df.rename(columns={c: "GSDP_per_capita"})
            break
    return df


if __name__ == "__main__":
    path = get_gsdp_path()
    save_rbi_table19_csv(path)
    df = load_gsdp_per_capita(path)
    print(df.head(12))

from pathlib import Path

"""
Shared configuration for cluster mapping tools and app.

- Points to the enriched district export data with PCI/ECI.
- Defines sector groupings based on HS2 prefixes.
- Defines default paths and column names for the India district shapefile.

NOTE: You must supply a Census-2011-compatible India districts shapefile at
`data/geo/india_districts_2011.shp` (or update `INDIA_DISTRICTS_SHP` below).
The Streamlit app will show a clear error if the file is missing.
"""

BASE_DIR = Path(__file__).resolve().parent.parent

# Input from PCI/ECI pipeline
DISTRICT_EXPORTS_ENRICHED = BASE_DIR / "dists_am25_enriched.csv"

# Output of sector aggregation
DISTRICT_SECTOR_CLUSTERS = BASE_DIR / "cluster_maps" / "district_sector_clusters.csv"

# Path to India districts shapefile (Census 2011-compatible), to be provided by user
INDIA_DISTRICTS_SHP = BASE_DIR / "data" / "geo" / "india_districts_2011.shp"

# Shapefile attribute columns used for joining to the trade data.
# For the SHRUG pc11 district shapefile, typical columns are:
#   - pc11_s_id  (state ID)
#   - pc11_d_id  (district ID)
#   - d_name     (district name)
# Adjust these if your shapefile uses different names.
SHAPEFILE_STATE_COL = "pc11_s_id"
SHAPEFILE_DISTRICT_COL = "d_name"


# Sector definitions: HS2 prefixes grouped into broad sectors.
# HS Code in the trade data is 4-digit HS4; HS2 is the first two digits.
# Each HS2 code should appear in at most one sector to avoid double counting.
SECTORS = {
    # Existing sectors
    "footwear": {
        "label": "Footwear (HS 64)",
        "hs2_prefixes": ["64"],
        "cmap": "Blues",
    },
    "auto": {
        "label": "Automotive (HS 87)",
        "hs2_prefixes": ["87"],
        "cmap": "Reds",
    },
    "leather": {
        "label": "Leather & Leather Goods (HS 41–43)",
        "hs2_prefixes": ["41", "42", "43"],
        "cmap": "Greens",
    },
    "pharma": {
        "label": "Pharmaceuticals (HS 30)",
        "hs2_prefixes": ["30"],
        "cmap": "Purples",
    },
    # New sectors (OEC-aligned and India-relevant)
    "gems_jewellery": {
        "label": "Gems & Jewellery (HS 71)",
        "hs2_prefixes": ["71"],
        "cmap": "YlOrBr",
    },
    "chemicals": {
        "label": "Chemical Products (HS 28–29, 31–36, 38)",
        "hs2_prefixes": ["28", "29", "31", "32", "33", "34", "35", "36", "38"],
        "cmap": "PuBuGn",
    },
    "food_grains": {
        "label": "Grains & Cereals (HS 10–11)",
        "hs2_prefixes": ["10", "11"],
        "cmap": "YlGn",
    },
    "food_fresh": {
        "label": "Fresh Fruit & Vegetables (HS 07–08)",
        "hs2_prefixes": ["07", "08"],
        "cmap": "Greens",
    },
    "food_processed": {
        "label": "Processed Food & Beverages (HS 16–22, 24)",
        "hs2_prefixes": ["16", "17", "18", "19", "20", "21", "22", "24"],
        "cmap": "YlOrRd",
    },
    "electronics": {
        "label": "Electronics & Electrical (HS 85)",
        "hs2_prefixes": ["85"],
        "cmap": "PuRd",
    },
    "toys": {
        "label": "Toys, Games & Sports Goods (HS 95)",
        "hs2_prefixes": ["95"],
        "cmap": "PuBu",
    },
    "furniture": {
        "label": "Furniture & Furnishings (HS 94)",
        "hs2_prefixes": ["94"],
        "cmap": "OrRd",
    },
    "machinery": {
        "label": "Machinery & Machines (HS 84)",
        "hs2_prefixes": ["84"],
        "cmap": "BuPu",
    },
    "optical_medical": {
        "label": "Optical & Medical Equipment (HS 90)",
        "hs2_prefixes": ["90"],
        "cmap": "GnBu",
    },
    "textiles_garments": {
        "label": "Textiles & Apparel (HS 50–63)",
        "hs2_prefixes": [
            "50",
            "51",
            "52",
            "53",
            "54",
            "55",
            "56",
            "57",
            "58",
            "59",
            "60",
            "61",
            "62",
            "63",
        ],
        "cmap": "PuBuGn",
    },
    "plastics_rubber": {
        "label": "Plastics & Rubber (HS 39–40)",
        "hs2_prefixes": ["39", "40"],
        "cmap": "Oranges",
    },
    "metals": {
        "label": "Metals & Metal Products (HS 72–76, 78–83)",
        "hs2_prefixes": [
            "72",
            "73",
            "74",
            "75",
            "76",
            "78",
            "79",
            "80",
            "81",
            "82",
            "83",
        ],
        "cmap": "Greys",
    },
    "mineral_products": {
        "label": "Minerals, Ores & Fuels (HS 25–27)",
        "hs2_prefixes": ["25", "26", "27"],
        "cmap": "YlOrBr",
    },
    "wood_paper": {
        "label": "Wood, Paper & Printing (HS 44–49)",
        "hs2_prefixes": ["44", "45", "46", "47", "48", "49"],
        "cmap": "YlGnBu",
    },
    "agri_livestock_other": {
        "label": "Other Agricultural Products (HS 01–06, 09, 12–15, 23)",
        "hs2_prefixes": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "09",
            "12",
            "13",
            "14",
            "15",
            "23",
        ],
        "cmap": "PiYG",
    },
    "transport_other": {
        "label": "Other Transport Equipment (HS 86, 88–89)",
        "hs2_prefixes": ["86", "88", "89"],
        "cmap": "RdPu",
    },
    "other_manufactures": {
        "label": "Other Manufactured Goods (HS 65–67, 68–70, 91–93, 96–97)",
        "hs2_prefixes": [
            "65",
            "66",
            "67",
            "68",
            "69",
            "70",
            "91",
            "92",
            "93",
            "96",
            "97",
        ],
        "cmap": "BrBG",
    },
}


# Default metrics that the map can show per district for a given sector.
DEFAULT_METRICS = {
    "Export USD": "export_usd",
    "Export INR": "export_inr",
    "Average PCI": "avg_pci",
    "ECI": "eci",
}


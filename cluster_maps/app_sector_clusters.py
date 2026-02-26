"""
Streamlit app: visualise district-level export clusters by sector on an India map.

Dependencies:
- pandas, geopandas
- folium, streamlit

Data requirements:
- Enriched trade data has already been processed into
  `cluster_maps/district_sector_clusters.csv` via build_district_sector_clusters.py.
- An India districts shapefile (Census 2011-compatible) is available at
  `data/geo/india_districts_2011.shp` (or update the path in cluster_config.py).
"""

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import streamlit as st
from folium import Choropleth, GeoJson, GeoJsonTooltip, LayerControl, Map
from streamlit.components.v1 import html

# Ensure project root is on sys.path so `cluster_maps` can be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cluster_maps.cluster_config import (
    DISTRICT_SECTOR_CLUSTERS,
    INDIA_DISTRICTS_SHP,
    SECTORS,
    SHAPEFILE_DISTRICT_COL,
    SHAPEFILE_STATE_COL,
    DEFAULT_METRICS,
)


st.set_page_config(
    page_title="India District Export Clusters",
    layout="wide",
)


# Backwards-compatible cache decorator for older Streamlit versions.
if hasattr(st, "cache_data"):
    def cache_data_fn(**kwargs):
        return st.cache_data(**kwargs)
else:
    def cache_data_fn(**kwargs):
        # In older Streamlit, allow_output_mutation avoids strict immutability checks.
        return st.cache(allow_output_mutation=True, **kwargs)


def load_sector_clusters(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"District-sector cluster table not found: {path}\n"
            f"Run cluster_maps/build_district_sector_clusters.py first."
        )
    df = pd.read_csv(path)
    # Normalise join keys
    df["state_key"] = df["State"].astype(str).str.strip().str.upper()
    df["district_key"] = df["District"].astype(str).str.strip().str.upper()
    return df


@cache_data_fn(show_spinner=False)
def load_district_geometries(
    shp_path: Path,
    state_col: str,
    district_col: str,
) -> gpd.GeoDataFrame:
    if not shp_path.exists():
        raise FileNotFoundError(
            f"India districts shapefile not found: {shp_path}\n"
            f"Please place a Census 2011-compatible districts shapefile there, "
            f"or update INDIA_DISTRICTS_SHP in cluster_maps/cluster_config.py."
        )
    gdf = gpd.read_file(shp_path)

    if state_col not in gdf.columns or district_col not in gdf.columns:
        raise ValueError(
            f"Expected columns '{state_col}' and '{district_col}' not found in shapefile.\n"
            f"Available columns: {list(gdf.columns)}\n"
            f"Update SHAPEFILE_STATE_COL / SHAPEFILE_DISTRICT_COL in cluster_maps/cluster_config.py "
            f"to match your shapefile."
        )

    gdf["state_key"] = gdf[state_col].astype(str).str.strip().str.upper()
    gdf["district_key"] = gdf[district_col].astype(str).str.strip().str.upper()

    # Ensure WGS84 for web mapping
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Return a copy so downstream mutations don't affect the cached object
    return gdf.copy()


def make_choropleth_map(
    gdf: gpd.GeoDataFrame,
    sector_key: str,
    metric_label: str,
    metric_col: str,
) -> Map:
    sector_conf = SECTORS.get(sector_key, {})
    cmap = sector_conf.get("cmap", "Blues")
    sector_label = sector_conf.get("label", sector_key)

    # Basic empty map if there is no geometry
    if gdf.empty or gdf.geometry.is_empty.all():
        return Map(location=[22.5, 80], zoom_start=4)

    # Compute bounds and center for initial view (focus on India extent)
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB positron")

    # Fit the map to India's bounds so we don't see the whole world.
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Choropleth based on metric
    data = gdf.dropna(subset=[metric_col]).copy()
    legend_name = f"{sector_label} – {metric_label}"

    # Use scaled units for exports to make the legend more readable.
    plot_col = metric_col
    if metric_col == "export_usd":
        # Show exports in millions of USD
        plot_col = "export_usd_million"
        data[plot_col] = (data[metric_col] / 1e6).round(2)
        legend_name = f"{sector_label} – Export (USD millions)"
    elif metric_col == "export_inr":
        # Show exports in billions of INR
        plot_col = "export_inr_billion"
        data[plot_col] = (data[metric_col] / 1e9).round(2)
        legend_name = f"{sector_label} – Export (INR billions)"

    if not data.empty:
        Choropleth(
            geo_data=data,
            data=data,
            columns=["district_key", plot_col],
            key_on="feature.properties.district_key",
            fill_color=cmap,
            fill_opacity=0.8,
            line_opacity=0.0,  # hide strong internal borders from the choropleth layer
            legend_name=legend_name,
        ).add_to(m)

    # Tooltip with key attributes
    tooltip_fields = [
        "State",
        "District",
        "sector",
        "export_usd",
        "export_inr",
        "avg_pci",
        "eci",
    ]
    tooltip_aliases = [
        "State:",
        "District:",
        "Sector:",
        "Export (USD):",
        "Export (INR):",
        "Avg PCI:",
        "ECI:",
    ]
    available_fields = [f for f in tooltip_fields if f in gdf.columns]

    GeoJson(
        gdf,
        style_function=lambda feature: {
            # Keep polygon borders invisible; colour alone highlights districts.
            "fillOpacity": 0.0,
            "color": "transparent",
            "weight": 0,
        },
        tooltip=GeoJsonTooltip(
            fields=available_fields,
            aliases=[a for f, a in zip(tooltip_fields, tooltip_aliases) if f in available_fields],
            localize=True,
        ),
    ).add_to(m)

    LayerControl().add_to(m)
    return m


def _interpretation_text(sector_label: str, metric_label: str) -> str:
    """One or two lines for laypersons on how to read the map for the chosen sector and metric."""
    if metric_label == "Export USD":
        return (
            f"**How to read:** Darker shades mean higher export value (US$) for **{sector_label}** from that district. "
            "Hover over a district to see exact numbers."
        )
    if metric_label == "Export INR":
        return (
            f"**How to read:** Darker shades mean higher export value (₹) for **{sector_label}** from that district. "
            "Hover over a district to see exact numbers."
        )
    if metric_label == "Average PCI":
        return (
            "**PCI (Product Complexity Index)** measures how sophisticated a product is. "
            f"For **{sector_label}**, darker shades mean the district exports more complex products in this sector."
        )
    if metric_label == "ECI":
        return (
            "**ECI (Economic Complexity Index)** measures how diversified and sophisticated a district’s overall exports are. "
            "Darker shades mean the district has a more complex export basket (not limited to the sector you selected)."
        )
    return f"Colour shows **{metric_label}** for **{sector_label}**. Hover over districts for details."


def main() -> None:
    st.title("India District Export Clusters by Sector")
    st.markdown(
        "Data based on **Financial Year 2024–25 (Apr 2024 – Mar 2025)**.\n\n"
        "**PCI** (Product Complexity Index) measures how sophisticated a product is; "
        "**ECI** (Economic Complexity Index) measures how diversified and sophisticated a district’s overall exports are. "
        "Use the sidebar to pick a sector and metric; hover over districts for details."
    )

    try:
        clusters = load_sector_clusters(DISTRICT_SECTOR_CLUSTERS)
    except Exception as e:
        st.error(str(e))
        return

    try:
        gdf = load_district_geometries(
            INDIA_DISTRICTS_SHP,
            SHAPEFILE_STATE_COL,
            SHAPEFILE_DISTRICT_COL,
        )
    except Exception as e:
        st.error(str(e))
        return

    # Sidebar controls
    sector_keys = list(SECTORS.keys())
    sector_labels = [SECTORS[k]["label"] for k in sector_keys]
    sector_choice_label = st.sidebar.selectbox(
        "Select sector", options=sector_labels, index=0
    )
    # Map back from label to key
    label_to_key = {SECTORS[k]["label"]: k for k in sector_keys}
    sector_key = label_to_key[sector_choice_label]

    metric_label = st.sidebar.selectbox(
        "Metric", options=list(DEFAULT_METRICS.keys()), index=0
    )
    metric_col = DEFAULT_METRICS[metric_label]

    # Filter clusters for selected sector and prepare join keys
    sector_df = clusters[clusters["sector"] == sector_key].copy()
    if sector_df.empty:
        st.warning(f"No data available for sector: {sector_choice_label}")
        return

    # Dynamic interpretation for the chosen sector and metric
    st.info(_interpretation_text(sector_choice_label, metric_label))

    # Join with geometries on normalised district names only.
    # The SHRUG district shapefile does not include state names, so we
    # rely on district names for the match; state names come from the
    # trade data side.
    merged = gdf.merge(
        sector_df,
        on=["district_key"],
        how="left",
        suffixes=("_geo", ""),
    )

    # Basic diagnostics
    total_districts = len(gdf)
    matched = merged[metric_col].notna().sum()
    st.sidebar.markdown(
        f"**Districts with data:** {matched} / {total_districts}"
    )

    # Map
    st.subheader(f"Map – {sector_choice_label} ({metric_label})")
    folium_map = make_choropleth_map(merged, sector_key, metric_label, metric_col)
    # Render Folium map as HTML inside Streamlit without streamlit-folium dependency
    map_html = folium_map.get_root().render()
    html(map_html, height=650, scrolling=False)

    # Top districts table
    st.subheader(f"Top districts – {sector_choice_label} by {metric_label}")
    top = (
        sector_df.sort_values(metric_col, ascending=False)
        .loc[:, ["State", "District", "sector", "export_usd", "export_inr", "avg_pci", "eci"]]
        .head(30)
    )
    top_display = top.copy()
    # Scale exports for readability: USD in millions, INR in crore.
    # Format as strings with a single decimal place (no extra trailing zeros).
    top_display["Export USD (Mn)"] = (top_display["export_usd"] / 1e6).map(
        lambda v: f"{v:.1f}"
    )
    top_display["Export INR (Cr)"] = (top_display["export_inr"] / 1e7).map(
        lambda v: f"{v:.1f}"
    )
    top_display = top_display[
        ["State", "District", "sector", "Export USD (Mn)", "Export INR (Cr)", "avg_pci", "eci"]
    ]
    st.dataframe(top_display)

    # Compact SHRUG citation in the footer
    st.caption(
        "District boundaries and SHRUG-based variables: Asher, S., T. Lunt, R. Matsuura, "
        "and P. Novosad (2021), *Development research at high geographic resolution: an "
        "analysis of night-lights, firms, and poverty in India using the SHRUG open data "
        "platform*, World Bank Economic Review."
    )


if __name__ == "__main__":
    main()


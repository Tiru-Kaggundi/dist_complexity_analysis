"""
Download helper for SHRUG district shapefiles (Census 2011-compatible).

This DOES NOT auto-download the data (the SHRUG portal requires
accepting terms of use and, in some cases, authentication). Instead,
it opens the relevant SHRUG pages and gives you clear instructions on
what to download and where to save it so that the cluster_maps app can
use the boundaries.

Target path for the district shapefile:
- data/geo/india_districts_2011.shp  (plus .dbf/.shx/.prj etc.)

After you download and unzip the SHRUG district shapefile, either:
- rename the .shp to india_districts_2011.shp and place all components
  under data/geo/, OR
- update INDIA_DISTRICTS_SHP in cluster_maps/cluster_config.py to
  point at your chosen .shp file.
"""

import time
import webbrowser
from pathlib import Path


def main() -> None:
    base = Path(__file__).parent
    target_dir = base / "data" / "geo"

    print("=" * 72)
    print("SHRUG District Shapefile Download Helper")
    print("=" * 72)
    print(
        f"""
This helper will open the relevant SHRUG pages in your browser.

Goal:
- Obtain a Census 2011-compatible district boundary shapefile from SHRUG
  and place it where the cluster_maps app expects it.

Steps:
1) Visit the SHRUG open-source polygons documentation
   - This describes the district, subdistrict, and village polygons.

2) Visit the SHRUG download portal
   - Look for an open-access district-level shapefile for India
     (Census 2011 districts). The file name may mention "pc11" and
     "district" or "shapefile".

3) Download the district shapefile ZIP and unzip it locally.

4) Move the shapefile components (.shp, .dbf, .shx, .prj, etc.) to:
   - {target_dir}

5) Ensure that the main .shp file path matches the setting in:
   - cluster_maps/cluster_config.py -> INDIA_DISTRICTS_SHP

   By default, the app expects:
   - {target_dir / "india_districts_2011.shp"}

   You can either rename the downloaded .shp to that name, or update
   INDIA_DISTRICTS_SHP to match the actual filename.

Once this is done, restart the Streamlit app:
   source .venv/bin/activate
   python -m streamlit run cluster_maps/app_sector_clusters.py
"""
    )

    target_dir.mkdir(parents=True, exist_ok=True)

    print("\nOpening SHRUG documentation for open-source polygons in 3 seconds...")
    time.sleep(3)
    webbrowser.open("https://docs.devdatalab.org/SHRUG-Construction-Details/shrug-open-source-polygons/")

    print("Opening SHRUG download portal in 3 seconds...")
    time.sleep(3)
    webbrowser.open("https://www.devdatalab.org/shrug_download/")

    print("\n✓ Browser tabs opened. Follow the on-screen instructions above to download the district shapefile.")


if __name__ == "__main__":
    main()


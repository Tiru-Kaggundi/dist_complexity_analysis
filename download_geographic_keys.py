"""
Download SHRUG Geographic Keys

Helper script to download SHRUG geographic keys for mapping district IDs to names.
"""

import webbrowser
import time
from pathlib import Path

def main():
    print("="*70)
    print("SHRUG Geographic Keys Download Helper")
    print("="*70)
    print("""
The VIIRS data uses numeric IDs (pc11_district_id, pc11_state_id) but no district names.
To match districts, we need the SHRUG geographic keys file.

STEP 1: Visit SHRUG Download Portal
  URL: https://www.devdatalab.org/shrug_download/
  
STEP 2: Navigate to Core Keys
  - Look for "Core SHRUG Modules" section
  - Find "Geographic Keys" or "District Keys"
  - Look for file: "shrug_pc11_district_key.dta" or similar
  
STEP 3: Download the File
  - Download the district keys file
  - Save to: data/shrug_geo/
  
STEP 4: After Download
  - Run the luminosity merger again with --geo-keys flag:
    python luminosity_merger.py dists_2025_full.csv dists_2025_full.csv \\
        data/shrug_viirs/viirs_annual_pc11dist.csv \\
        --geo-keys data/shrug_geo/shrug_pc11_district_key.dta
""")
    
    print("\nOpening browser in 3 seconds...")
    time.sleep(3)
    webbrowser.open("https://www.devdatalab.org/shrug_download/")
    print("✓ Browser opened!")

if __name__ == "__main__":
    main()

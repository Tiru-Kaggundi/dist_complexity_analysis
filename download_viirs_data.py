"""
SHRUG VIIRS Data Download Helper

Provides instructions and helper functions for downloading SHRUG VIIRS data.
"""

import os
import sys
from pathlib import Path
import warnings


def print_download_instructions():
    """Print step-by-step instructions for downloading SHRUG VIIRS data."""
    print("="*70)
    print("SHRUG VIIRS Data Download Instructions")
    print("="*70)
    print("""
Step 1: Visit SHRUG Download Portal
  URL: https://www.devdatalab.org/shrug_download/
  
Step 2: Accept License Agreement
  - Read and accept the Creative Commons license
  - Create account if required (free registration)
  
Step 3: Navigate to VIIRS Data
  - Go to: Remote Sensing → Night-time lights → VIIRS
  - Or direct link: https://www.devdatalab.org/shrug_download/
  
Step 4: Download District-Level File
  - Look for file: "shrug_nl_viirs_pc11dist.dta" or similar
  - This contains district-level VIIRS data
  - Download the file
  
Step 5: Save to Project Directory
  - Create directory: data/shrug_viirs/
  - Save the downloaded file there
  - Expected path: data/shrug_viirs/shrug_nl_viirs_pc11dist.dta
  
Step 6: Run the Merger
  - Use: python luminosity_merger.py dists_2025_full.csv dists_2025_full.csv data/shrug_viirs/shrug_nl_viirs_pc11dist.dta
  - Or use the Python API:
    
    from luminosity_merger import process_and_add_luminosity
    process_and_add_luminosity(
        input_path='dists_2025_full.csv',
        output_path='dists_2025_full.csv',
        viirs_path='data/shrug_viirs/shrug_nl_viirs_pc11dist.dta',
        year=2021
    )
""")
    print("="*70)


def check_viirs_file(path: str) -> bool:
    """
    Check if VIIRS file exists and is readable.
    
    Args:
        path: Path to VIIRS file
        
    Returns:
        True if file exists and is readable
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        print(f"❌ File not found: {path}")
        return False
    
    print(f"✅ File found: {path}")
    print(f"   Size: {path_obj.stat().st_size / (1024*1024):.2f} MB")
    
    # Try to read first few rows
    try:
        if path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
        elif path.endswith('.dta'):
            try:
                import pandas as pd
                df = pd.read_stata(path)
                df = df.head(5)
            except ImportError:
                print("   ⚠️  pandas.stata module not available. Install: pip install pyreadstat")
                return False
        else:
            print("   ⚠️  Unknown file format. Trying CSV...")
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
        
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample rows: {len(df)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def main():
    """Main function: print instructions and check for existing files."""
    print_download_instructions()
    
    # Check for common file locations
    common_paths = [
        "data/shrug_viirs/shrug_nl_viirs_pc11dist.dta",
        "data/shrug_viirs/shrug_nl_viirs_pc11dist.csv",
        "data/shrug_viirs/viirs_district.csv",
        "data/shrug_viirs/viirs_district.dta",
    ]
    
    print("\n" + "="*70)
    print("Checking for existing VIIRS files...")
    print("="*70)
    
    found = False
    for path in common_paths:
        if Path(path).exists():
            print(f"\nFound existing file:")
            check_viirs_file(path)
            found = True
            print(f"\n✅ Ready to use! Run:")
            print(f"   python luminosity_merger.py dists_2025_full.csv dists_2025_full.csv {path}")
            break
    
    if not found:
        print("\n⚠️  No VIIRS files found in common locations.")
        print("   Please download SHRUG VIIRS data following the instructions above.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

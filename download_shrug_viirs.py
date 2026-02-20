"""
SHRUG VIIRS Data Download Script

Helps download SHRUG VIIRS night lights data.
Since SHRUG requires manual download, this script:
1. Opens the download page in browser
2. Provides step-by-step instructions
3. Checks for downloaded files
4. Validates the data once downloaded
"""

import os
import sys
import webbrowser
import time
from pathlib import Path
import warnings

try:
    import pandas as pd
except ImportError:
    pd = None
    warnings.warn("pandas not available. Install: pip install pandas")


def open_shrug_download_page():
    """Open SHRUG download page in browser."""
    url = "https://www.devdatalab.org/shrug_download/"
    print(f"Opening SHRUG download page: {url}")
    webbrowser.open(url)
    print("✓ Browser opened. Please follow the instructions below.")


def print_download_instructions():
    """Print detailed download instructions."""
    print("\n" + "="*70)
    print("SHRUG VIIRS Data Download Instructions")
    print("="*70)
    print("""
STEP 1: Navigate to VIIRS Data
  - On the SHRUG download page, look for "Remote Sensing" section
  - Click on "Night-time lights" → "VIIRS"
  - Or search for "viirs" in the search box

STEP 2: Find District-Level File
  - Look for file containing "pc11dist" (district-level)
  - File name might be: "shrug_nl_viirs_pc11dist.dta" or similar
  - This file contains district-level VIIRS data for all years (1992-2021)

STEP 3: Download the File
  - Click the download button/link
  - Save the file to: data/shrug_viirs/
  - Create the directory if it doesn't exist

STEP 4: Verify Download
  - Run this script again with --check flag
  - Or run: python download_shrug_viirs.py --check

Expected file location:
  data/shrug_viirs/shrug_nl_viirs_pc11dist.dta
  OR
  data/shrug_viirs/shrug_nl_viirs_pc11dist.csv
""")


def check_downloaded_file(expected_path: str = None) -> bool:
    """
    Check if VIIRS file has been downloaded.
    
    Args:
        expected_path: Expected file path (optional)
        
    Returns:
        True if file found and valid
    """
    if expected_path:
        paths_to_check = [expected_path]
    else:
        # Common locations
        paths_to_check = [
            "data/shrug_viirs/shrug_nl_viirs_pc11dist.dta",
            "data/shrug_viirs/shrug_nl_viirs_pc11dist.csv",
            "data/shrug_viirs/viirs_district.dta",
            "data/shrug_viirs/viirs_district.csv",
            "data/shrug_viirs/*.dta",
            "data/shrug_viirs/*.csv",
        ]
    
    print("\n" + "="*70)
    print("Checking for downloaded VIIRS files...")
    print("="*70)
    
    found_files = []
    
    # Check specific paths
    for path_str in paths_to_check[:4]:  # First 4 are specific paths
        path = Path(path_str)
        if path.exists():
            found_files.append(path)
            print(f"\n✓ Found: {path}")
            print(f"  Size: {path.stat().st_size / (1024*1024):.2f} MB")
    
    # Check directory for any files
    shrug_dir = Path("data/shrug_viirs")
    if shrug_dir.exists():
        for ext in ['.dta', '.csv']:
            for file in shrug_dir.glob(f"*{ext}"):
                if file not in found_files:
                    found_files.append(file)
                    print(f"\n✓ Found: {file}")
                    print(f"  Size: {file.stat().st_size / (1024*1024):.2f} MB")
    
    if not found_files:
        print("\n⚠️  No VIIRS files found.")
        print("   Please download the file following the instructions above.")
        return False
    
    # Validate the first file found
    if pd:
        return validate_viirs_file(found_files[0])
    else:
        print(f"\n✓ File found: {found_files[0]}")
        print("   (Install pandas to validate contents)")
        return True


def validate_viirs_file(file_path: Path) -> bool:
    """
    Validate VIIRS file structure.
    
    Args:
        file_path: Path to VIIRS file
        
    Returns:
        True if file is valid
    """
    print(f"\nValidating file: {file_path}")
    
    try:
        # Try to read the file
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=100)
        elif file_path.suffix == '.dta':
            try:
                df = pd.read_stata(file_path)
                df = df.head(100)
            except ImportError:
                print("  ⚠️  pandas.stata not available. Install: pip install pyreadstat")
                print("  File appears to be Stata format (.dta)")
                return True  # Assume valid if we can't check
        else:
            # Try CSV
            df = pd.read_csv(file_path, nrows=100)
        
        print(f"  ✓ File readable")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        
        # Check for expected columns
        expected_keywords = ['district', 'viirs', 'year', 'mean', 'pc11']
        found_keywords = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in expected_keywords:
                if keyword in col_lower:
                    found_keywords.append(keyword)
        
        if found_keywords:
            print(f"  ✓ Found relevant columns: {set(found_keywords)}")
        else:
            print(f"  ⚠️  Warning: No obvious VIIRS/district columns found")
            print(f"     This might be the wrong file, or column names are different")
        
        # Check for year column
        year_cols = [c for c in df.columns if 'year' in str(c).lower()]
        if year_cols:
            print(f"  ✓ Year column found: {year_cols}")
            if len(df) > 0:
                years = df[year_cols[0]].unique()[:10]
                print(f"  Sample years: {sorted(years)}")
        
        print(f"\n  ✓ File validation complete")
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False


def create_output_directory():
    """Create output directory if it doesn't exist."""
    output_dir = Path("data/shrug_viirs")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    return output_dir


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download SHRUG VIIRS night lights data"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if file has been downloaded"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Check specific file path"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    create_output_directory()
    
    if args.check or args.file:
        # Check for downloaded file
        file_path = args.file if args.file else None
        if check_downloaded_file(file_path):
            print("\n" + "="*70)
            print("✓ VIIRS file is ready!")
            print("="*70)
            print("\nNext step: Run the luminosity merger:")
            print("  python luminosity_merger.py dists_2025_full.csv dists_2025_full.csv data/shrug_viirs/<filename>")
        else:
            print("\n" + "="*70)
            print("⚠️  File not found or invalid")
            print("="*70)
            print_download_instructions()
    else:
        # Show instructions and open browser
        print_download_instructions()
        
        if not args.no_browser:
            print("\nOpening browser in 3 seconds...")
            print("(Press Ctrl+C to cancel)")
            try:
                time.sleep(3)
                open_shrug_download_page()
            except KeyboardInterrupt:
                print("\nCancelled. Run with --no-browser to skip opening browser.")
        
        print("\n" + "="*70)
        print("After downloading, run:")
        print("  python download_shrug_viirs.py --check")
        print("="*70)


if __name__ == "__main__":
    main()

"""
Script to download NITI Aayog Multidimensional Poverty Index (MPI) district-level data.

Data Sources:
1. GitHub repository (tam0w/poverty_data) - Pre-processed district-level MPI data
2. NITI Aayog SDG Dashboard - Official source
3. Open Government Data Platform (data.gov.in) - State/UT level data
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings


def download_mpi_from_github(output_path: str = "data/niti_mpi/mpi_district_data.csv") -> pd.DataFrame:
    """
    Download district-level MPI data from GitHub repository.
    
    Source: https://github.com/tam0w/poverty_data
    This repository contains district-level MPI Headcount Ratio (HCR) data
    extracted from NITI Aayog's 2023 MPI report.
    
    Args:
        output_path: Path to save the downloaded CSV file
        
    Returns:
        DataFrame with district-level MPI data
    """
    url = "https://raw.githubusercontent.com/tam0w/poverty_data/master/DATESET.csv"
    
    print(f"Downloading MPI data from GitHub repository...")
    print(f"Source: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded successfully to {output_path}")
        
        # Load and process the data
        df = pd.read_csv(output_path)
        
        # The dataset has many columns, but we need: State, District, MPI HCR
        # Check available columns
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Find MPI HCR column (might be named "MPI HCR" or similar)
        mpi_col = None
        for col in df.columns:
            if 'mpi' in col.lower() or 'hcr' in col.lower() or 'poverty' in col.lower():
                mpi_col = col
                break
        
        if mpi_col:
            print(f"Found MPI column: {mpi_col}")
        else:
            warnings.warn("Could not find MPI/HCR column. Please check the dataset manually.")
        
        return df
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download from GitHub: {e}")


def download_mpi_from_ogd_platform(output_path: str = "data/niti_mpi/mpi_state_data.csv") -> Optional[pd.DataFrame]:
    """
    Download MPI data from Open Government Data Platform (data.gov.in).
    
    Note: This provides state/UT level data, not district level.
    URL: https://www.data.gov.in/resource/stateut-wise-details-headcount-ratio-intensity-and-multi-dimensional-poverty-index-mpi
    
    Args:
        output_path: Path to save the downloaded CSV file
        
    Returns:
        DataFrame with state/UT level MPI data, or None if download fails
    """
    # The OGD platform typically requires visiting the page and clicking download
    # This is a placeholder - actual implementation would need to handle the OGD API
    # or scrape the download page
    
    print("\nNote: OGD Platform (data.gov.in) requires manual download.")
    print("Visit: https://www.data.gov.in/resource/stateut-wise-details-headcount-ratio-intensity-and-multi-dimensional-poverty-index-mpi")
    print("Click 'Download' to get the CSV file.")
    
    return None


def process_mpi_data(df: pd.DataFrame, 
                    state_col: str = "State",
                    district_col: str = "District",
                    mpi_col: Optional[str] = None) -> pd.DataFrame:
    """
    Process downloaded MPI data to standardize format.
    
    Args:
        df: Raw MPI DataFrame
        state_col: Name of state column
        district_col: Name of district column
        mpi_col: Name of MPI column (will be auto-detected if None)
        
    Returns:
        Processed DataFrame with standardized columns
    """
    df = df.copy()
    
    # Find MPI column if not specified
    if mpi_col is None:
        for col in df.columns:
            col_lower = col.lower()
            if 'mpi' in col_lower and 'hcr' in col_lower:
                mpi_col = col
                break
            elif 'mpi' in col_lower:
                mpi_col = col
                break
            elif 'hcr' in col_lower or 'headcount' in col_lower:
                mpi_col = col
                break
    
    if mpi_col is None:
        raise ValueError("Could not find MPI column. Available columns: " + ", ".join(df.columns))
    
    # Standardize column names
    result_df = pd.DataFrame({
        "State": df[state_col].astype(str).str.strip(),
        "District": df[district_col].astype(str).str.strip(),
        "MPI_HCR": df[mpi_col]
    })
    
    # Clean MPI_HCR values (remove % sign if present, convert to float)
    if result_df["MPI_HCR"].dtype == 'object':
        result_df["MPI_HCR"] = result_df["MPI_HCR"].astype(str).str.replace('%', '').str.strip()
        result_df["MPI_HCR"] = pd.to_numeric(result_df["MPI_HCR"], errors='coerce')
    
    # Remove rows with missing values
    result_df = result_df.dropna(subset=["State", "District", "MPI_HCR"])
    
    print(f"\nProcessed {len(result_df)} districts")
    print(f"MPI HCR range: {result_df['MPI_HCR'].min():.2f}% - {result_df['MPI_HCR'].max():.2f}%")
    
    return result_df


def download_niti_mpi_data(method: str = "github",
                           output_dir: str = "data/niti_mpi",
                           process: bool = True) -> pd.DataFrame:
    """
    Main function to download NITI Aayog MPI district-level data.
    
    Args:
        method: Download method - "github" (recommended) or "ogd"
        output_dir: Directory to save downloaded files
        process: Whether to process and standardize the data
        
    Returns:
        DataFrame with MPI data
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    if method == "github":
        raw_path = output_dir_path / "mpi_raw_github.csv"
        processed_path = output_dir_path / "mpi_district_data.csv"
        
        df = download_mpi_from_github(str(raw_path))
        
        if process:
            # Process the data
            df_processed = process_mpi_data(df)
            df_processed.to_csv(processed_path, index=False)
            print(f"\n✓ Processed data saved to {processed_path}")
            return df_processed
        else:
            return df
            
    elif method == "ogd":
        print("\nOGD Platform download requires manual steps.")
        print("Please visit: https://www.data.gov.in/resource/stateut-wise-details-headcount-ratio-intensity-and-multi-dimensional-poverty-index-mpi")
        return download_mpi_from_ogd_platform()
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'github' or 'ogd'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NITI Aayog MPI district-level data")
    parser.add_argument("--method", choices=["github", "ogd"], default="github",
                       help="Download method (default: github)")
    parser.add_argument("--output-dir", default="data/niti_mpi",
                       help="Output directory (default: data/niti_mpi)")
    parser.add_argument("--no-process", action="store_true",
                       help="Skip data processing")
    
    args = parser.parse_args()
    
    try:
        df = download_niti_mpi_data(
            method=args.method,
            output_dir=args.output_dir,
            process=not args.no_process
        )
        
        print("\n" + "="*60)
        print("Download Summary:")
        print("="*60)
        print(f"Total districts: {len(df)}")
        print(f"\nFirst few rows:")
        print(df.head())
        print("\n✓ Download completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

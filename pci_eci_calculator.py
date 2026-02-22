"""
PCI and ECI Calculator Module

Calculates Product Complexity Index (PCI) and Economic Complexity Index (ECI)
for districts using OEC methodology.

Uses:
- OEC API to fetch PCI values
- District-level trade data (HS4 products)
- OEC formula for ECI calculation
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import requests
import time

# Try to import API token
try:
    from apikey import OEC_API_TOKEN
except ImportError:
    OEC_API_TOKEN = None
    warnings.warn("apikey.py not found. Please create it with your OEC API token.")


def load_base_dataset(path: str, min_export_usd: float = 1000) -> pd.DataFrame:
    """
    Load and prepare base trade dataset.
    
    Args:
        path: Path to base dataset CSV
        min_export_usd: Minimum export value in USD to include product
        
    Returns:
        Prepared DataFrame with filtered and cleaned data
    """
    print(f"Loading base dataset from {path}...")
    df = pd.read_csv(path)
    
    print(f"  Initial rows: {len(df)}")
    
    # Get USD column name (handle spaces and special characters)
    usd_col = None
    for col in df.columns:
        if 'USD' in col.upper() or 'US $' in col or 'US$' in col:
            usd_col = col
            break
    
    if usd_col is None:
        raise ValueError("Could not find USD export value column")
    
    # Filter by minimum export value
    initial_count = len(df)
    df = df[df[usd_col] >= min_export_usd].copy()
    print(f"  After filtering (>= ${min_export_usd:,.0f} USD): {len(df)} rows ({initial_count - len(df)} removed)")
    
    # Clean HS Code (ensure 4-digit string format)
    df['HS Code'] = df['HS Code'].astype(str).str.strip()
    # Pad with zeros if needed
    df['HS Code'] = df['HS Code'].str.zfill(4)
    # Remove invalid codes (non-numeric or wrong length)
    invalid_mask = ~df['HS Code'].str.match(r'^\d{4}$')
    if invalid_mask.sum() > 0:
        print(f"  Warning: Removing {invalid_mask.sum()} rows with invalid HS codes")
        df = df[~invalid_mask].copy()
    
    # Aggregate by district and HS4 code (sum if multiple entries)
    agg_cols = ['State', 'District', 'HS Code']
    agg_dict = {
        usd_col: 'sum',
        'Commodity Description': 'first'  # Keep first description
    }
    
    # Keep INR column if it exists
    inr_col = None
    for col in df.columns:
        if 'INR' in col.upper() and col != usd_col:
            inr_col = col
            agg_dict[col] = 'sum'
            break
    
    df_agg = df.groupby(agg_cols, as_index=False).agg(agg_dict)
    print(f"  After aggregation: {len(df_agg)} rows")
    
    # Rename USD column for easier access
    df_agg = df_agg.rename(columns={usd_col: 'Export_USD'})
    if inr_col:
        df_agg = df_agg.rename(columns={inr_col: 'Export_INR'})
    
    return df_agg


def load_oec_pci_data(path: Optional[str] = None, 
                     use_api: bool = True,
                     api_token: Optional[str] = None,
                     cache_path: str = "data/oec_pci.csv") -> pd.DataFrame:
    """
    Load Product Complexity Index (PCI) data from OEC API or local file.
    
    Args:
        path: Path to local PCI CSV file (if not using API)
        use_api: Whether to fetch from OEC API
        api_token: OEC API token (uses apikey.py if None)
        cache_path: Path to cache PCI data locally
        
    Returns:
        DataFrame with HS4 codes and PCI values
    """
    cache_file = Path(cache_path)
    
    # Check if cached file exists
    if cache_file.exists() and not use_api:
        print(f"Loading PCI data from cache: {cache_file}")
        pci_df = pd.read_csv(cache_file)
        # Clean HS4 for merge (cache may have been read as int)
        pci_df['HS4'] = pci_df['HS4'].astype(str).str.strip().str.zfill(4)
        pci_df['PCI'] = pd.to_numeric(pci_df['PCI'], errors='coerce')
        pci_df = pci_df.dropna(subset=['PCI'])
        print(f"  Loaded {len(pci_df)} products")
        return pci_df
    
    if use_api:
        # Use API token from parameter or apikey.py
        token = api_token or OEC_API_TOKEN
        
        if not token or token == "YOUR_TOKEN":
            raise ValueError(
                "OEC API token not found. Please set OEC_API_TOKEN in apikey.py"
            )
        
        print("Fetching PCI data from OEC API...")
        
        # OEC API endpoint for PCI data
        # Use the complexity_pci_a_hs12_hs4 cube which contains PCI values
        url = "https://api-v2.oec.world/tesseract/data.jsonrecords"
        
        # Use PCI complexity cube with HS4 level (latest year)
        # Note: API returns latest year by default for this cube
        all_records = []
        limit = 1000  # API limit per request
        offset = 0
        
        while True:
            params = {
                "cube": "complexity_pci_a_hs12_hs4",
                "drilldowns": "HS4 Official",
                "measures": "PCI",
                "limit": f"{limit},{offset}",
                "token": token
            }
            
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                if 'data' not in data:
                    break
                
                records = data['data']
                if not records:
                    break
                
                all_records.extend(records)
                
                # Check if there are more records
                page_info = data.get('page', {})
                total = page_info.get('total', 0)
                current_count = len(all_records)
                
                print(f"  Fetched {current_count} / {total} products...", end='\r')
                
                if current_count >= total:
                    break
                
                offset += limit
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to fetch PCI data from OEC API: {e}")
        
        print()  # New line after progress
        
        if not all_records:
            raise ValueError("No PCI data retrieved from API")
        
        # Convert to DataFrame
        pci_df = pd.DataFrame(all_records)
        
        # Extract HS4 code from 'HS4 Official ID' column
        if 'HS4 Official ID' in pci_df.columns:
            pci_df = pci_df.rename(columns={'HS4 Official ID': 'HS4'})
        elif 'HS4 Official' in pci_df.columns:
            # If only name is available, try to extract code
            # This shouldn't happen, but handle it
            pci_df['HS4'] = pci_df['HS4 Official'].astype(str)
        
        # Keep only HS4 and PCI columns
        pci_df = pci_df[['HS4', 'PCI']].copy()
        
        # Clean HS4 codes (ensure 4-digit format)
        pci_df['HS4'] = pci_df['HS4'].astype(str).str.strip().str.zfill(4)
        
        # Ensure PCI is numeric
        pci_df['PCI'] = pd.to_numeric(pci_df['PCI'], errors='coerce')
        
        # Remove rows with missing PCI
        pci_df = pci_df.dropna(subset=['PCI'])
        
        print(f"  Fetched {len(pci_df)} products from OEC API")
        
        # Cache the data
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pci_df.to_csv(cache_file, index=False)
        print(f"  Cached to {cache_file}")
        
        return pci_df
    
    elif path:
        # Load from local file
        print(f"Loading PCI data from local file: {path}")
        pci_df = pd.read_csv(path)
        
        # Ensure correct column names
        if 'HS4' not in pci_df.columns or 'PCI' not in pci_df.columns:
            # Try to find and rename columns
            for col in pci_df.columns:
                if 'hs4' in col.lower() or col.lower() == 'code':
                    pci_df = pci_df.rename(columns={col: 'HS4'})
                if 'pci' in col.lower():
                    pci_df = pci_df.rename(columns={col: 'PCI'})
        
        # Clean HS4 codes
        pci_df['HS4'] = pci_df['HS4'].astype(str).str.strip().str.zfill(4)
        pci_df['PCI'] = pd.to_numeric(pci_df['PCI'], errors='coerce')
        pci_df = pci_df.dropna(subset=['PCI'])
        
        print(f"  Loaded {len(pci_df)} products")
        return pci_df
    
    else:
        raise ValueError("Either use_api=True (with token) or provide path to local PCI file")


def add_pci_column(base_df: pd.DataFrame, pci_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PCI column to base dataset by merging with PCI data.
    
    Args:
        base_df: Base trade dataset
        pci_df: PCI DataFrame with HS4 and PCI columns
        
    Returns:
        Base dataset with PCI column added
    """
    print("\nAdding PCI column...")
    # Coerce HS codes to 4-digit string for merge
    base_df = base_df.copy()
    base_df['HS Code'] = base_df['HS Code'].astype(str).str.strip().str.zfill(4)
    pci_df = pci_df.copy()
    pci_df['HS4'] = pci_df['HS4'].astype(str).str.strip().str.zfill(4)
    # Merge PCI values
    result_df = base_df.merge(
        pci_df[['HS4', 'PCI']],
        left_on='HS Code',
        right_on='HS4',
        how='left'
    )
    
    # Drop the duplicate HS4 column from merge
    if 'HS4' in result_df.columns and 'HS4' != 'HS Code':
        result_df = result_df.drop(columns=['HS4'])
    
    # Report missing PCI values
    missing_pci = result_df['PCI'].isna().sum()
    total_products = len(result_df)
    
    if missing_pci > 0:
        print(f"  Warning: {missing_pci} products ({missing_pci/total_products*100:.1f}%) have no PCI value")
        print(f"  Products with PCI: {total_products - missing_pci}")
    else:
        print(f"  All {total_products} products have PCI values")
    
    return result_df


def calculate_rca(trade_df: pd.DataFrame,
                 district_col: str = 'District',
                 hs4_col: str = 'HS Code',
                 value_col: str = 'Export_USD') -> pd.DataFrame:
    """
    Calculate Revealed Comparative Advantage (RCA) for each district-product pair.
    
    Formula: RCA_ij = (X_ij / Σ_j X_ij) / (Σ_i X_ij / Σ_ij X_ij)
    
    Args:
        trade_df: Trade DataFrame with district, product, and export values
        district_col: Name of district column
        hs4_col: Name of HS4 code column
        value_col: Name of export value column
        
    Returns:
        DataFrame with RCA and RCA_binary columns added
    """
    print("\nCalculating Revealed Comparative Advantage (RCA)...")
    
    df = trade_df.copy()
    
    # Calculate totals
    # X_ij = exports of product j by district i
    # Σ_j X_ij = total exports of district i
    # Σ_i X_ij = total exports of product j
    # Σ_ij X_ij = total exports (all districts, all products)
    
    total_exports = df[value_col].sum()  # Σ_ij X_ij
    
    # Total exports per district
    district_totals = df.groupby(district_col)[value_col].sum()  # Σ_j X_ij
    df['district_total'] = df[district_col].map(district_totals)
    
    # Total exports per product
    product_totals = df.groupby(hs4_col)[value_col].sum()  # Σ_i X_ij
    df['product_total'] = df[hs4_col].map(product_totals)
    
    # Calculate RCA
    # RCA_ij = (X_ij / Σ_j X_ij) / (Σ_i X_ij / Σ_ij X_ij)
    df['RCA'] = (df[value_col] / df['district_total']) / (df['product_total'] / total_exports)
    
    # Binary RCA: 1 if RCA >= 1.0, else 0
    df['RCA_binary'] = (df['RCA'] >= 1.0).astype(int)
    
    # Clean up temporary columns
    df = df.drop(columns=['district_total', 'product_total'])
    
    print(f"  Calculated RCA for {len(df)} district-product pairs")
    print(f"  Districts with RCA >= 1.0: {df['RCA_binary'].sum()} pairs")
    
    return df


def calculate_eci(trade_df: pd.DataFrame,
                 pci_df: pd.DataFrame,
                 district_col: str = 'District',
                 hs4_col: str = 'HS Code',
                 min_products: int = 5,
                 standardize: bool = True) -> pd.DataFrame:
    """
    Calculate Economic Complexity Index (ECI) for each district using OEC formula.
    
    Formula: ECI_i = Σ_j (PCI_j × RCA_binary_ij) / Σ_j RCA_binary_ij
    
    Args:
        trade_df: Trade DataFrame with RCA_binary column
        pci_df: PCI DataFrame with HS4 and PCI columns
        district_col: Name of district column
        hs4_col: Name of HS4 code column
        min_products: Minimum products with RCA >= 1.0 needed for ECI calculation
        standardize: Whether to standardize ECI (mean=0, std=1)
        
    Returns:
        DataFrame with State, District, and ECI columns
    """
    print("\nCalculating Economic Complexity Index (ECI)...")
    
    # Ensure trade_df has PCI column
    if 'PCI' not in trade_df.columns:
        trade_df = add_pci_column(trade_df, pci_df)
    
    # Filter to products with PCI and RCA_binary
    df = trade_df[trade_df['PCI'].notna() & trade_df['RCA_binary'].notna()].copy()
    
    # Calculate ECI for each district
    eci_results = []
    
    for district in df[district_col].unique():
        district_data = df[df[district_col] == district].copy()
        
        # Only consider products with RCA >= 1.0
        rca_products = district_data[district_data['RCA_binary'] == 1].copy()
        
        if len(rca_products) < min_products:
            # Not enough products for reliable ECI
            eci_results.append({
                district_col: district,
                'ECI': np.nan,
                'num_products': len(rca_products)
            })
            continue
        
        # Calculate ECI: Σ_j (PCI_j × RCA_binary_ij) / Σ_j RCA_binary_ij
        # Since RCA_binary is 1 for these products, this simplifies to:
        # ECI = mean(PCI_j) for products with RCA >= 1.0
        eci = rca_products['PCI'].mean()
        
        eci_results.append({
            district_col: district,
            'ECI': eci,
            'num_products': len(rca_products)
        })
    
    eci_df = pd.DataFrame(eci_results)
    
    # Add State column if available
    if 'State' in trade_df.columns:
        state_map = trade_df.groupby(district_col)['State'].first()
        eci_df['State'] = eci_df[district_col].map(state_map)
        # Reorder columns
        eci_df = eci_df[['State', district_col, 'ECI', 'num_products']]
    
    # Standardize ECI if requested
    if standardize:
        valid_eci = eci_df['ECI'].dropna()
        if len(valid_eci) > 0:
            mean_eci = valid_eci.mean()
            std_eci = valid_eci.std()
            if std_eci > 0:
                eci_df['ECI_std'] = (eci_df['ECI'] - mean_eci) / std_eci
            else:
                eci_df['ECI_std'] = 0.0
    
    # Report statistics
    valid_eci = eci_df['ECI'].dropna()
    print(f"  Calculated ECI for {len(valid_eci)} districts")
    print(f"  Districts without ECI: {eci_df['ECI'].isna().sum()}")
    if len(valid_eci) > 0:
        print(f"  ECI range: {valid_eci.min():.3f} to {valid_eci.max():.3f}")
        print(f"  ECI mean: {valid_eci.mean():.3f}")
    
    return eci_df


def add_eci_column(base_df: pd.DataFrame, eci_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ECI column to base dataset by merging with ECI data.
    
    Args:
        base_df: Base trade dataset
        eci_df: ECI DataFrame with District and ECI columns
        
    Returns:
        Base dataset with ECI column added
    """
    print("\nAdding ECI column...")
    
    # Merge ECI values
    result_df = base_df.merge(
        eci_df[['District', 'ECI', 'ECI_std']],
        on='District',
        how='left'
    )
    
    # Report coverage
    districts_with_eci = result_df['ECI'].notna().sum()
    total_rows = len(result_df)
    
    print(f"  Rows with ECI: {districts_with_eci} ({districts_with_eci/total_rows*100:.1f}%)")
    
    return result_df


def process_and_enrich_dataset(input_path: str,
                               output_path: str,
                               pci_path: Optional[str] = None,
                               min_export_usd: float = 1000,
                               min_products_for_eci: int = 5,
                               use_oec_api: bool = True) -> pd.DataFrame:
    """
    Main function: Process base dataset and add PCI and ECI columns.
    
    Args:
        input_path: Path to base trade dataset CSV
        output_path: Path to save enriched dataset CSV
        pci_path: Optional path to local PCI file (if not using API)
        min_export_usd: Minimum export value in USD
        min_products_for_eci: Minimum products needed for ECI calculation
        use_oec_api: Whether to fetch PCI from OEC API
        
    Returns:
        Enriched DataFrame with PCI and ECI columns
    """
    print("="*60)
    print("PCI and ECI Calculation Pipeline")
    print("="*60)
    
    # Step 1: Load base dataset
    base_df = load_base_dataset(input_path, min_export_usd=min_export_usd)
    
    # Step 2: Load PCI data
    pci_df = load_oec_pci_data(
        path=pci_path,
        use_api=use_oec_api,
        cache_path="data/oec_pci.csv"
    )
    
    # Step 3: Add PCI column
    base_df = add_pci_column(base_df, pci_df)
    
    # Step 4: Calculate RCA
    base_df = calculate_rca(base_df)
    
    # Step 5: Calculate ECI
    eci_df = calculate_eci(
        base_df,
        pci_df,
        min_products=min_products_for_eci,
        standardize=True
    )
    
    # Step 6: Add ECI column
    base_df = add_eci_column(base_df, eci_df)
    
    # Step 7: Save enriched dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Output saved to: {output_file}")
    print(f"Total rows: {len(base_df)}")
    print(f"Products with PCI: {base_df['PCI'].notna().sum()}")
    print(f"Districts with ECI: {base_df['ECI'].notna().sum()}")
    
    return base_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate PCI and ECI for trade dataset")
    parser.add_argument("--input", default="dgcis_stateoforigin1771519863052/dists_2025_full.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", default="dists_2025_full.csv",
                       help="Output CSV file path")
    parser.add_argument("--min-export", type=float, default=1000,
                       help="Minimum export value in USD (default: 1000)")
    parser.add_argument("--min-products", type=int, default=5,
                       help="Minimum products for ECI calculation (default: 5)")
    parser.add_argument("--pci-file", default=None,
                       help="Local PCI file path (if not using API)")
    parser.add_argument("--no-api", action="store_true",
                       help="Don't use OEC API (use local PCI file)")
    
    args = parser.parse_args()
    
    try:
        result_df = process_and_enrich_dataset(
            input_path=args.input,
            output_path=args.output,
            pci_path=args.pci_file,
            min_export_usd=args.min_export,
            min_products_for_eci=args.min_products,
            use_oec_api=not args.no_api
        )
        
        print("\n✓ Success!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

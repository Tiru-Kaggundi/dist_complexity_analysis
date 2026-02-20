"""
Luminosity Data Merger Module

Adds VIIRS nighttime lights (luminosity) data to the enriched trade dataset.
Supports both SHRUG pre-processed data and direct raster processing.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

# Optional geopandas import for raster processing
try:
    import geopandas as gpd
    from rasterio import features
    from rasterio.mask import mask
    import rasterio
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("geopandas/rasterio not available. Raster processing disabled.")


def load_shrug_viirs(path: str, year: int = 2021, geo_keys_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load SHRUG VIIRS data from CSV or Stata file.
    
    Args:
        path: Path to SHRUG VIIRS file
        year: Year to extract (default: 2021)
        geo_keys_path: Optional path to SHRUG geographic keys file for ID-to-name mapping
        
    Returns:
        DataFrame with State, District, VIIRS_Mean columns
    """
    print(f"Loading SHRUG VIIRS data from {path}...")
    
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"SHRUG VIIRS file not found: {path}")
    
    # Try CSV first, then Stata
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.dta'):
        try:
            df = pd.read_stata(path)
        except ImportError:
            raise ImportError("pandas.stata module required for .dta files. Install: pip install pyreadstat")
    else:
        # Try CSV by default
        df = pd.read_csv(path)
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Find relevant columns
    district_id_col = None
    state_id_col = None
    district_name_col = None
    state_name_col = None
    viirs_col = None
    year_col = None
    
    # Common column name patterns
    for col in df.columns:
        col_lower = col.lower()
        if 'pc11_district_id' in col_lower:
            district_id_col = col
        elif 'pc11_state_id' in col_lower:
            state_id_col = col
        elif 'district' in col_lower and ('name' in col_lower or 'nm' in col_lower):
            district_name_col = col
        elif 'state' in col_lower and ('name' in col_lower or 'nm' in col_lower):
            state_name_col = col
        elif 'viirs' in col_lower and 'mean' in col_lower:
            viirs_col = col
        elif col_lower == 'year' or 'yr' in col_lower:
            year_col = col
    
    if not viirs_col:
        raise ValueError(f"Could not find VIIRS mean column. Available columns: {list(df.columns)}")
    
    if not district_id_col:
        raise ValueError(f"Could not find pc11_district_id column. Available columns: {list(df.columns)}")
    
    if not state_id_col:
        raise ValueError(f"Could not find pc11_state_id column. Available columns: {list(df.columns)}")
    
    # Filter by year if year column exists
    if year_col:
        df = df[df[year_col] == year].copy()
        print(f"  Filtered to year {year}: {len(df)} rows")
    
    # If we don't have district/state names, try to load from geographic keys
    if not district_name_col or not state_name_col:
        if geo_keys_path and Path(geo_keys_path).exists():
            print(f"  Loading geographic keys from {geo_keys_path}...")
            geo_df = load_geographic_keys(geo_keys_path)
            
            # Merge to get district and state names
            df = df.merge(
                geo_df[['pc11_state_id', 'pc11_district_id', 'district_name', 'state_name']],
                on=['pc11_state_id', 'pc11_district_id'],
                how='left'
            )
            district_name_col = 'district_name'
            state_name_col = 'state_name'
            print(f"  Merged geographic keys: {df[district_name_col].notna().sum()} districts have names")
        else:
            # No geographic keys available - we'll need to match by ID later
            print("  ⚠️  Warning: No district/state names found and no geographic keys provided.")
            print("     Will attempt to match using district IDs from trade dataset.")
            # Create placeholder columns
            df['district_name'] = None
            df['state_name'] = None
            district_name_col = 'district_name'
            state_name_col = 'state_name'
    
    # Build result DataFrame
    result_df = pd.DataFrame({
        'pc11_state_id': df[state_id_col],
        'pc11_district_id': df[district_id_col],
        'State': df[state_name_col] if state_name_col in df.columns else None,
        'District': df[district_name_col] if district_name_col in df.columns else None,
        'VIIRS_Mean': df[viirs_col]
    })
    
    # Clean and normalize names if available
    if result_df['District'].notna().any():
        result_df = normalize_district_names(result_df)
    else:
        print("  ⚠️  No district names available - will match by ID")
    
    print(f"  Final VIIRS data: {len(result_df)} districts")
    if result_df['VIIRS_Mean'].notna().any():
        print(f"  VIIRS range: {result_df['VIIRS_Mean'].min():.2f} - {result_df['VIIRS_Mean'].max():.2f}")
    
    return result_df


def load_geographic_keys(path: str) -> pd.DataFrame:
    """
    Load SHRUG geographic keys file to map IDs to names.
    
    Args:
        path: Path to geographic keys file
        
    Returns:
        DataFrame with pc11_state_id, pc11_district_id, district_name, state_name
    """
    path_obj = Path(path)
    
    if path_obj.suffix.lower() == '.csv':
        df = pd.read_csv(path_obj)
    elif path_obj.suffix.lower() == '.dta':
        try:
            df = pd.read_stata(path_obj)
        except ImportError:
            raise ImportError("pandas.stata module required for .dta files. Install: pip install pyreadstat")
    else:
        df = pd.read_csv(path_obj)
    
    # Find relevant columns
    district_id_col = None
    state_id_col = None
    district_name_col = None
    state_name_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'pc11_district_id' in col_lower:
            district_id_col = col
        elif 'pc11_state_id' in col_lower:
            state_id_col = col
        elif 'district' in col_lower and ('name' in col_lower or 'nm' in col_lower):
            district_name_col = col
        elif 'state' in col_lower and ('name' in col_lower or 'nm' in col_lower):
            state_name_col = col
    
    if not district_id_col or not state_id_col:
        raise ValueError(f"Could not find ID columns in geographic keys. Available: {list(df.columns)}")
    
    result = pd.DataFrame({
        'pc11_state_id': pd.to_numeric(df[state_id_col], errors='coerce'),
        'pc11_district_id': pd.to_numeric(df[district_id_col], errors='coerce'),
        'district_name': df[district_name_col] if district_name_col else None,
        'state_name': df[state_name_col] if state_name_col else None
    })
    
    result = result.dropna(subset=['pc11_state_id', 'pc11_district_id'])
    
    # If keys file had only IDs (no names), attach names from Census 2011 order
    if (result['district_name'].isna().all() or result['state_name'].isna().all()):
        result = _attach_census2011_names_to_keys(result)
    
    # Return one row per district for use in VIIRS merge
    result = result.drop_duplicates(subset=['pc11_state_id', 'pc11_district_id'])
    return result


def _attach_census2011_names_to_keys(keys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach state/district names to (pc11_state_id, pc11_district_id) using
    Census 2011 district list order (matches SHRUG/census code order).
    """
    import io
    census_path = Path("data/census2011_district_names.csv")
    census_url = "https://raw.githubusercontent.com/nishusharma1608/India-Census-2011-Analysis/master/india-districts-census-2011.csv"
    
    if census_path.exists():
        census = pd.read_csv(census_path, usecols=["State name", "District name"])
    else:
        try:
            import requests
            r = requests.get(census_url, timeout=30)
            r.raise_for_status()
            census = pd.read_csv(io.BytesIO(r.content), usecols=["State name", "District name"])
            census_path.parent.mkdir(parents=True, exist_ok=True)
            census.to_csv(census_path, index=False)
        except Exception as e:
            warnings.warn(f"Could not load Census 2011 names: {e}")
            return keys_df
    
    census = census.rename(columns={"State name": "state_name", "District name": "district_name"})
    keys_sorted = keys_df.drop_duplicates(subset=["pc11_state_id", "pc11_district_id"]).sort_values(
        ["pc11_state_id", "pc11_district_id"]
    ).reset_index(drop=True)
    n = min(len(keys_sorted), len(census))
    keys_sorted = keys_sorted.head(n).copy()
    keys_sorted["state_name"] = census["state_name"].values[:n]
    keys_sorted["district_name"] = census["district_name"].values[:n]
    keys_df = keys_df.drop(columns=["district_name", "state_name"], errors="ignore")
    keys_df = keys_df.merge(
        keys_sorted[["pc11_state_id", "pc11_district_id", "state_name", "district_name"]],
        on=["pc11_state_id", "pc11_district_id"],
        how="left"
    )
    return keys_df


# Known district name variants (trade/Census spelling -> canonical for matching)
DISTRICT_ALIASES = {
    "BUDGAM": "BADGAM",
    "BAGALKOTE": "BAGALKOT",
    "BALLARI": "BELLARY",
    "BELAGAVI": "BELGAUM",
    "CHIKKAMAGALURU": "CHIKMAGALUR",
    "CHIKKABALLAPURA": "CHICKBALLAPUR",
    "BENGALURU RURAL": "BANGALORE RURAL",
    "BENGALURU URBAN": "BANGALORE",
    "EAST SINGHBUM": "PURBI SINGHBHUM",
    "EAST SINGHBHUM": "PURBI SINGHBHUM",
    "PURBI SINGHBUM": "PURBI SINGHBHUM",
    "CHARKI DADRI": "CHARKHI DADRI",
    "NORTH  AND MIDDLE ANDAMAN": "NORTH AND MIDDLE ANDAMAN",
    "JOGULAMBA GADWAL": "GADWAL",  # Census 2011 lists as Gadwal
}
# State name variants (Census 2011 uses ORISSA, newer data uses ODISHA, etc.)
STATE_ALIASES = {
    "ODISHA": "ORISSA",
    "ORISSA": "ORISSA",
}


def _normalize_name_for_matching(name: str, use_district_aliases: bool = True) -> str:
    """Single name normalization for matching (suffixes, spaces, aliases)."""
    if pd.isna(name) or name is None:
        return name
    name = str(name).strip().upper()
    for suffix in [" DISTRICT", " DIST", " ZILLA", " ZILA"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    name = name.replace(" & ", " AND ")
    name = " ".join(name.split())
    if use_district_aliases:
        name = DISTRICT_ALIASES.get(name, name)
    return name


def _normalize_state_for_matching(name: str) -> str:
    """Normalize state name and apply state aliases."""
    n = _normalize_name_for_matching(name)
    return STATE_ALIASES.get(n, n)


def normalize_district_names(df: pd.DataFrame, district_col: str = 'District', 
                            state_col: str = 'State') -> pd.DataFrame:
    """
    Normalize district and state names for matching.
    
    Args:
        df: DataFrame with district/state columns
        district_col: Name of district column
        state_col: Name of state column
        
    Returns:
        DataFrame with normalized names
    """
    df = df.copy()
    df[district_col] = df[district_col].apply(_normalize_name_for_matching)
    if state_col in df.columns:
        df[state_col] = df[state_col].apply(_normalize_state_for_matching)
    return df


def match_districts_for_luminosity(
    trade_df: pd.DataFrame, 
    viirs_df: pd.DataFrame, 
    fuzzy_threshold: int = 85,
    geo_keys_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Match districts between trade dataset and VIIRS data.
    
    Strategy:
    1. Exact match on State + District
    2. Fuzzy match within same state for unmatched districts
    3. Average if multiple VIIRS districts match one trade district
    4. Replicate if one VIIRS district matches multiple trade districts
    
    Args:
        trade_df: Trade dataset with State, District columns
        viirs_df: VIIRS dataset with State, District, VIIRS_Mean columns
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        
    Returns:
        DataFrame with matched VIIRS values
    """
    print("\nMatching districts for luminosity data...")
    
    # Check if we have district names in VIIRS data
    has_viirs_names = viirs_df['District'].notna().any() if 'District' in viirs_df.columns else False
    has_viirs_ids = 'pc11_district_id' in viirs_df.columns and viirs_df['pc11_district_id'].notna().any()
    
    # If VIIRS has IDs but no names, try to get names from geographic keys
    if has_viirs_ids and not has_viirs_names:
        if geo_keys_path and Path(geo_keys_path).exists():
            print("  Loading geographic keys to map IDs to names...")
            geo_df = load_geographic_keys(geo_keys_path)
            viirs_df = viirs_df.merge(
                geo_df[['pc11_state_id', 'pc11_district_id', 'district_name', 'state_name']],
                on=['pc11_state_id', 'pc11_district_id'],
                how='left'
            )
            viirs_df['District'] = viirs_df['district_name']
            viirs_df['State'] = viirs_df['state_name']
            has_viirs_names = viirs_df['District'].notna().any()
            print(f"  Mapped {viirs_df['District'].notna().sum()} districts to names")
        else:
            print("\n" + "="*70)
            print("ERROR: VIIRS data has IDs but no district names!")
            print("="*70)
            print("\nTo fix this, you need to download SHRUG geographic keys:")
            print("  1. Run: python download_geographic_keys.py")
            print("  2. Download 'shrug_pc11_district_key.dta' from SHRUG")
            print("  3. Save to: data/shrug_geo/shrug_pc11_district_key.dta")
            print("  4. Run merger with: --geo-keys data/shrug_geo/shrug_pc11_district_key.dta")
            print("\n" + "="*70)
            raise ValueError(
                "VIIRS data has IDs but no district names. "
                "Please download SHRUG geographic keys. Run: python download_geographic_keys.py"
            )
    
    if not has_viirs_names:
        raise ValueError("Cannot match districts: VIIRS data has no district names or IDs")
    
    # Normalize both datasets
    trade_norm = normalize_district_names(trade_df.copy())
    viirs_norm = normalize_district_names(viirs_df.copy())
    
    # Step 1: Exact match on State + District
    print("  Step 1: Exact matching...")
    exact_match = trade_norm.merge(
        viirs_norm[['State', 'District', 'VIIRS_Mean']],
        on=['State', 'District'],
        how='left',
        suffixes=('', '_viirs')
    )
    
    matched_count = exact_match['VIIRS_Mean'].notna().sum()
    unmatched_count = len(exact_match) - matched_count
    print(f"    Matched: {matched_count} districts ({matched_count/len(exact_match)*100:.1f}%)")
    print(f"    Unmatched: {unmatched_count} districts")
    
    # Step 2: Fuzzy matching for unmatched districts (multi-scorer, two passes)
    if unmatched_count > 0:
        # Pass 1: token_sort_ratio and token_set_ratio, take best score (threshold 85)
        print(f"  Step 2a: Fuzzy matching (threshold: {fuzzy_threshold}, multi-scorer)...")
        unmatched_mask = exact_match['VIIRS_Mean'].isna()
        unmatched_trade = exact_match[unmatched_mask].copy()
        fuzzy_matched = 0

        def best_fuzzy_match(trade_district: str, candidates: list, threshold: int) -> tuple:
            """Return (best_match, best_score) using token_sort_ratio and token_set_ratio."""
            if not candidates:
                return (None, 0)
            sort_best = process.extractOne(
                trade_district, candidates, scorer=fuzz.token_sort_ratio
            )
            set_best = process.extractOne(
                trade_district, candidates, scorer=fuzz.token_set_ratio
            )
            sort_score = sort_best[1] if sort_best else 0
            set_score = set_best[1] if set_best else 0
            if sort_score >= set_score and sort_best and sort_score >= threshold:
                return (sort_best[0], sort_score)
            if set_best and set_score >= threshold:
                return (set_best[0], set_score)
            return (None, max(sort_score, set_score))

        for idx, row in unmatched_trade.iterrows():
            trade_state = row['State']
            trade_district = row['District']
            if pd.isna(trade_state) or pd.isna(trade_district):
                continue
            state_viirs = viirs_norm[viirs_norm['State'] == trade_state].copy()
            if len(state_viirs) == 0:
                continue
            candidates = state_viirs['District'].dropna().unique().tolist()
            matched_district, score = best_fuzzy_match(trade_district, candidates, fuzzy_threshold)
            if matched_district is not None:
                matched_viirs = state_viirs[state_viirs['District'] == matched_district]
                if len(matched_viirs) > 0:
                    exact_match.loc[idx, 'VIIRS_Mean'] = matched_viirs['VIIRS_Mean'].mean()
                    fuzzy_matched += 1

        print(f"    Fuzzy matched (pass 1): {fuzzy_matched} districts")

        # Pass 2: lower threshold (72) for remaining unmatched
        still_unmatched = exact_match['VIIRS_Mean'].isna()
        if still_unmatched.any():
            lower_threshold = 72
            print(f"  Step 2b: Second fuzzy pass (threshold: {lower_threshold})...")
            unmatched_trade2 = exact_match[still_unmatched].copy()
            fuzzy_matched_2 = 0
            for idx, row in unmatched_trade2.iterrows():
                trade_state = row['State']
                trade_district = row['District']
                if pd.isna(trade_state) or pd.isna(trade_district):
                    continue
                state_viirs = viirs_norm[viirs_norm['State'] == trade_state].copy()
                if len(state_viirs) == 0:
                    continue
                candidates = state_viirs['District'].dropna().unique().tolist()
                matched_district, _ = best_fuzzy_match(trade_district, candidates, lower_threshold)
                if matched_district is not None:
                    matched_viirs = state_viirs[state_viirs['District'] == matched_district]
                    if len(matched_viirs) > 0:
                        exact_match.loc[idx, 'VIIRS_Mean'] = matched_viirs['VIIRS_Mean'].mean()
                        fuzzy_matched_2 += 1
            print(f"    Fuzzy matched (pass 2): {fuzzy_matched_2} districts")
    
    # Step 3: Handle multiple matches (one VIIRS district -> multiple trade districts)
    # This is already handled by the merge (replicates value)
    
    # Step 4: Handle multiple matches (multiple VIIRS districts -> one trade district)
    # Check if we need to average multiple VIIRS values for same trade district
    print("  Step 3: Checking for multiple matches...")
    
    # Group by trade State+District and check for duplicates in VIIRS
    viirs_grouped = viirs_norm.groupby(['State', 'District'])['VIIRS_Mean'].mean().reset_index()
    
    # Re-merge with averaged values
    final_match = trade_norm.merge(
        viirs_grouped[['State', 'District', 'VIIRS_Mean']],
        on=['State', 'District'],
        how='left',
        suffixes=('', '_viirs')
    )
    
    # Use exact match values where available, otherwise use averaged
    final_match['VIIRS_Mean'] = exact_match['VIIRS_Mean'].fillna(final_match['VIIRS_Mean'])
    
    final_matched = final_match['VIIRS_Mean'].notna().sum()
    final_unmatched = len(final_match) - final_matched
    
    print(f"\n  Final matching results:")
    print(f"    Matched: {final_matched} districts ({final_matched/len(final_match)*100:.1f}%)")
    print(f"    Unmatched: {final_unmatched} districts")
    
    if final_unmatched > 0:
        unmatched_districts = final_match[final_match['VIIRS_Mean'].isna()][['State', 'District']].drop_duplicates()
        print(f"\n  Unmatched districts (sample):")
        print(unmatched_districts.head(10).to_string(index=False))
    
    return final_match[['State', 'District', 'VIIRS_Mean']]


def add_luminosity_column(
    base_df: pd.DataFrame, 
    viirs_df: pd.DataFrame, 
    fuzzy_threshold: int = 85,
    fill_missing: str = 'nan',
    geo_keys_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Add VIIRS luminosity column to base dataset.
    
    Args:
        base_df: Base trade dataset
        viirs_df: VIIRS dataset with State, District, VIIRS_Mean
        fuzzy_threshold: Minimum similarity for fuzzy matching
        fill_missing: How to handle missing values ('nan', 'state_mean', 'zero')
        
    Returns:
        Base dataset with VIIRS_Mean column added
    """
    print("\nAdding luminosity column to base dataset...")
    
    # Drop existing VIIRS_Mean if present (e.g. from a previous run) to avoid merge suffixes
    base_df = base_df.drop(columns=['VIIRS_Mean'], errors='ignore')
    
    # Get unique districts from base dataset
    base_districts = base_df[['State', 'District']].drop_duplicates().copy()
    
    # Match districts
    matched = match_districts_for_luminosity(base_districts, viirs_df, fuzzy_threshold, geo_keys_path=geo_keys_path)
    
    # Merge back to base dataset
    result_df = base_df.merge(
        matched[['State', 'District', 'VIIRS_Mean']],
        on=['State', 'District'],
        how='left'
    )
    
    # Handle missing values
    missing_count = result_df['VIIRS_Mean'].isna().sum()
    if missing_count > 0:
        print(f"\n  Handling {missing_count} missing VIIRS values...")
        
        if fill_missing == 'state_mean':
            # Fill with state average
            state_means = result_df.groupby('State')['VIIRS_Mean'].transform('mean')
            result_df['VIIRS_Mean'] = result_df['VIIRS_Mean'].fillna(state_means)
            print(f"    Filled {missing_count} missing values with state averages")
        elif fill_missing == 'zero':
            result_df['VIIRS_Mean'] = result_df['VIIRS_Mean'].fillna(0)
            print(f"    Filled {missing_count} missing values with zeros")
        else:
            print(f"    Left {missing_count} missing values as NaN")
    
    print(f"\n  Final dataset: {len(result_df)} rows")
    print(f"  VIIRS coverage: {(result_df['VIIRS_Mean'].notna().sum() / len(result_df) * 100):.1f}%")
    print(f"  VIIRS range: {result_df['VIIRS_Mean'].min():.2f} - {result_df['VIIRS_Mean'].max():.2f}")
    
    return result_df


def process_viirs_raster(
    raster_path: str, 
    shapefile_path: str, 
    year: int = 2021
) -> pd.DataFrame:
    """
    Process VIIRS raster file to extract district-level statistics.
    
    Requires geopandas and rasterio.
    
    Args:
        raster_path: Path to VIIRS GeoTIFF file
        shapefile_path: Path to district shapefile
        year: Year of data (for metadata)
        
    Returns:
        DataFrame with State, District, VIIRS_Mean columns
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("geopandas and rasterio required for raster processing")
    
    print(f"Processing VIIRS raster: {raster_path}")
    print(f"Using shapefile: {shapefile_path}")
    
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    print(f"  Loaded {len(gdf)} districts from shapefile")
    
    # Load raster
    with rasterio.open(raster_path) as src:
        print(f"  Raster bounds: {src.bounds}")
        print(f"  Raster CRS: {src.crs}")
        
        # Ensure CRS match
        if gdf.crs != src.crs:
            print(f"  Reprojecting shapefile from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Extract zonal statistics
        results = []
        for idx, row in gdf.iterrows():
            geom = [row.geometry]
            
            try:
                out_image, out_transform = mask(src, geom, crop=True)
                out_image = out_image[0]  # First band
                
                # Calculate mean (excluding NoData)
                valid_pixels = out_image[out_image > 0]
                if len(valid_pixels) > 0:
                    mean_value = np.mean(valid_pixels)
                else:
                    mean_value = np.nan
                
                # Get district name (adjust column name as needed)
                district_col = None
                state_col = None
                for col in gdf.columns:
                    if 'district' in col.lower() and 'name' in col.lower():
                        district_col = col
                    elif 'state' in col.lower() and 'name' in col.lower():
                        state_col = col
                
                results.append({
                    'State': row[state_col] if state_col else None,
                    'District': row[district_col] if district_col else None,
                    'VIIRS_Mean': mean_value
                })
            except Exception as e:
                warnings.warn(f"Error processing district {idx}: {e}")
                continue
    
    result_df = pd.DataFrame(results)
    result_df = normalize_district_names(result_df)
    
    print(f"  Extracted VIIRS data for {len(result_df)} districts")
    
    return result_df


def process_and_add_luminosity(
    input_path: str,
    output_path: str,
    viirs_path: Optional[str] = None,
    raster_path: Optional[str] = None,
    shapefile_path: Optional[str] = None,
    geo_keys_path: Optional[str] = None,
    year: int = 2021,
    fuzzy_threshold: int = 85,
    fill_missing: str = 'nan'
) -> pd.DataFrame:
    """
    Main function: Load trade dataset, add luminosity data, save.
    
    Args:
        input_path: Path to enriched trade dataset (with PCI/ECI)
        output_path: Path to save output dataset
        viirs_path: Path to SHRUG VIIRS CSV/DTA file (if using pre-processed)
        raster_path: Path to VIIRS GeoTIFF (if using raster processing)
        shapefile_path: Path to district shapefile (required if using raster)
        year: Year of VIIRS data
        fuzzy_threshold: Minimum similarity for fuzzy matching
        fill_missing: How to handle missing values
        
    Returns:
        Enriched DataFrame with VIIRS_Mean column
    """
    print("="*60)
    print("Luminosity Data Merger")
    print("="*60)
    
    # Load base dataset
    print(f"\nLoading base dataset: {input_path}")
    base_df = pd.read_csv(input_path)
    print(f"  Loaded {len(base_df)} rows")
    
    # Try to find geographic keys automatically if not provided
    if not geo_keys_path:
        common_geo_paths = [
            "data/shrug_geo/shrug_pc11_district_key.dta",
            "data/shrug_geo/shrug_pc11_district_key.csv",
            "data/shrug_geo/pc11_district_key.dta",
            "data/shrug_geo/pc11_district_key.csv",
        ]
        for path in common_geo_paths:
            if Path(path).exists():
                geo_keys_path = path
                print(f"  Found geographic keys: {geo_keys_path}")
                break
    
    # Load VIIRS data
    if viirs_path:
        viirs_df = load_shrug_viirs(viirs_path, year, geo_keys_path)
    elif raster_path and shapefile_path:
        viirs_df = process_viirs_raster(raster_path, shapefile_path, year)
    else:
        raise ValueError("Must provide either viirs_path (SHRUG) or raster_path+shapefile_path")
    
    # Check if VIIRS data has district names
    has_names = viirs_df['District'].notna().any() if 'District' in viirs_df.columns else False
    
    if not has_names and 'pc11_district_id' in viirs_df.columns:
        # Need to match by ID - create mapping from trade dataset
        print("\n⚠️  VIIRS data has IDs but no district names.")
        print("   Attempting to match using district IDs from trade dataset...")
        
        # Try to match by creating a district name mapping
        # This is a workaround - ideally we'd have geographic keys
        if geo_keys_path and Path(geo_keys_path).exists():
            print(f"   Using geographic keys: {geo_keys_path}")
        else:
            print("   ⚠️  Geographic keys not found. Matching may be limited.")
            print("   To improve matching, download SHRUG geographic keys:")
            print("   https://www.devdatalab.org/shrug_download/")
            print("   Look for 'Core Keys' → 'District Keys'")
        
        # Match using IDs if available in trade dataset
        # For now, we'll try name-based matching if possible
        if 'pc11_district_id' not in base_df.columns:
            print("   ⚠️  Trade dataset doesn't have district IDs.")
            print("   Cannot match by ID. Need geographic keys or district names in VIIRS data.")
    
    # Add luminosity column
    result_df = add_luminosity_column(
        base_df, 
        viirs_df, 
        fuzzy_threshold=fuzzy_threshold,
        fill_missing=fill_missing,
        geo_keys_path=geo_keys_path
    )
    
    # Save
    print(f"\nSaving enriched dataset: {output_path}")
    result_df.to_csv(output_path, index=False)
    print(f"  Saved {len(result_df)} rows to {output_path}")
    
    return result_df


if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge VIIRS luminosity data into trade dataset")
    parser.add_argument("input_csv", help="Input trade dataset CSV")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("viirs_path", nargs="?", help="Path to VIIRS data file")
    parser.add_argument("--raster", help="Use raster processing (requires raster_path and shapefile_path)")
    parser.add_argument("--raster-path", help="Path to VIIRS raster file")
    parser.add_argument("--shapefile-path", help="Path to district shapefile")
    parser.add_argument("--geo-keys", help="Path to SHRUG geographic keys file (for ID-to-name mapping)")
    parser.add_argument("--year", type=int, default=2021, help="Year of VIIRS data (default: 2021)")
    parser.add_argument("--fuzzy-threshold", type=int, default=85, help="Fuzzy matching threshold (default: 85)")
    parser.add_argument("--fill-missing", choices=['nan', 'state_mean', 'zero'], default='nan',
                       help="How to handle missing values (default: nan)")
    
    args = parser.parse_args()
    
    if args.raster:
        if not args.raster_path or not args.shapefile_path:
            print("Error: --raster requires --raster-path and --shapefile-path")
            sys.exit(1)
        process_and_add_luminosity(
            args.input_csv,
            args.output_csv,
            raster_path=args.raster_path,
            shapefile_path=args.shapefile_path,
            geo_keys_path=args.geo_keys,
            year=args.year,
            fuzzy_threshold=args.fuzzy_threshold,
            fill_missing=args.fill_missing
        )
    else:
        if not args.viirs_path:
            print("Error: viirs_path required (or use --raster)")
            sys.exit(1)
        process_and_add_luminosity(
            args.input_csv,
            args.output_csv,
            viirs_path=args.viirs_path,
            geo_keys_path=args.geo_keys,
            year=args.year,
            fuzzy_threshold=args.fuzzy_threshold,
            fill_missing=args.fill_missing
        )

"""
India District Data ETL Pipeline

This module provides functions to ingest, clean, harmonize, and merge three
distinct sub-national datasets for India into a single master cross-sectional
dataset at the district level.

Data Sources:
1. SHRUG (Development Data Lab) - Consumption/GDP Proxy (Census 2011 boundaries)
2. SHRUG - Remote Sensing (VIIRS) (Census 2011 boundaries)
3. NITI Aayog National MPI (2023 baseline via NFHS-5) (~2021 boundaries)
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import geopandas as gpd
from rapidfuzz import process, fuzz
import config


def load_shrug_secc(path: str) -> pd.DataFrame:
    """
    Load SHRUG SECC consumption data.
    
    Extracts rural per capita consumption proxy (secc_cons_pc_rural) along with
    geographic identifiers (pc11_state_id, pc11_district_id).
    
    Args:
        path: Path to SHRUG SECC data directory or file. Supports CSV and DTA formats.
        
    Returns:
        DataFrame containing pc11_state_id, pc11_district_id, and secc_cons_pc_rural
        
    Raises:
        FileNotFoundError: If the specified path does not exist
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    
    # Handle directory path - look for common SHRUG file patterns
    if path_obj.is_dir():
        csv_files = list(path_obj.glob("*.csv"))
        dta_files = list(path_obj.glob("*.dta"))
        
        if csv_files:
            file_path = csv_files[0]
            df = pd.read_csv(file_path)
        elif dta_files:
            file_path = dta_files[0]
            df = pd.read_stata(file_path)
        else:
            raise FileNotFoundError(f"No CSV or DTA files found in {path}")
    elif path_obj.exists():
        # Handle file path
        if path_obj.suffix.lower() == '.csv':
            df = pd.read_csv(path_obj)
        elif path_obj.suffix.lower() == '.dta':
            df = pd.read_stata(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Validate required columns
    required_cols = config.REQUIRED_COLUMNS["shrug_secc"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in SHRUG SECC data: {missing_cols}")
    
    # Extract and return relevant columns
    result_df = df[required_cols].copy()
    
    # Ensure numeric types
    result_df["pc11_state_id"] = pd.to_numeric(result_df["pc11_state_id"], errors='coerce')
    result_df["pc11_district_id"] = pd.to_numeric(result_df["pc11_district_id"], errors='coerce')
    result_df["secc_cons_pc_rural"] = pd.to_numeric(result_df["secc_cons_pc_rural"], errors='coerce')
    
    # Remove rows with missing geographic IDs
    result_df = result_df.dropna(subset=["pc11_state_id", "pc11_district_id"])
    
    return result_df


def load_shrug_viirs(path: str) -> pd.DataFrame:
    """
    Load SHRUG VIIRS nighttime lights data.
    
    Extracts mean nighttime light radiance (viirs_mean) along with
    geographic identifiers (pc11_state_id, pc11_district_id).
    
    Args:
        path: Path to SHRUG VIIRS data directory or file. Supports CSV and DTA formats.
        
    Returns:
        DataFrame containing pc11_state_id, pc11_district_id, and viirs_mean
        
    Raises:
        FileNotFoundError: If the specified path does not exist
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    
    # Handle directory path - look for common SHRUG file patterns
    if path_obj.is_dir():
        csv_files = list(path_obj.glob("*.csv"))
        dta_files = list(path_obj.glob("*.dta"))
        
        if csv_files:
            file_path = csv_files[0]
            df = pd.read_csv(file_path)
        elif dta_files:
            file_path = dta_files[0]
            df = pd.read_stata(file_path)
        else:
            raise FileNotFoundError(f"No CSV or DTA files found in {path}")
    elif path_obj.exists():
        # Handle file path
        if path_obj.suffix.lower() == '.csv':
            df = pd.read_csv(path_obj)
        elif path_obj.suffix.lower() == '.dta':
            df = pd.read_stata(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Validate required columns
    required_cols = config.REQUIRED_COLUMNS["shrug_viirs"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in SHRUG VIIRS data: {missing_cols}")
    
    # Extract and return relevant columns
    result_df = df[required_cols].copy()
    
    # Ensure numeric types
    result_df["pc11_state_id"] = pd.to_numeric(result_df["pc11_state_id"], errors='coerce')
    result_df["pc11_district_id"] = pd.to_numeric(result_df["pc11_district_id"], errors='coerce')
    result_df["viirs_mean"] = pd.to_numeric(result_df["viirs_mean"], errors='coerce')
    
    # Remove rows with missing geographic IDs
    result_df = result_df.dropna(subset=["pc11_state_id", "pc11_district_id"])
    
    return result_df


def load_shrug_geo(path: str) -> pd.DataFrame:
    """
    Load SHRUG geographic keys data.
    
    Extracts district names along with geographic identifiers
    (pc11_state_id, pc11_district_id, district_name).
    
    Args:
        path: Path to SHRUG geographic keys data directory or file.
              Supports CSV and DTA formats.
        
    Returns:
        DataFrame containing pc11_state_id, pc11_district_id, and district_name
        
    Raises:
        FileNotFoundError: If the specified path does not exist
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    
    # Handle directory path - look for common SHRUG file patterns
    if path_obj.is_dir():
        csv_files = list(path_obj.glob("*.csv"))
        dta_files = list(path_obj.glob("*.dta"))
        
        if csv_files:
            file_path = csv_files[0]
            df = pd.read_csv(file_path)
        elif dta_files:
            file_path = dta_files[0]
            df = pd.read_stata(file_path)
        else:
            raise FileNotFoundError(f"No CSV or DTA files found in {path}")
    elif path_obj.exists():
        # Handle file path
        if path_obj.suffix.lower() == '.csv':
            df = pd.read_csv(path_obj)
        elif path_obj.suffix.lower() == '.dta':
            df = pd.read_stata(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Validate required columns
    required_cols = config.REQUIRED_COLUMNS["shrug_geo"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in SHRUG geo data: {missing_cols}")
    
    # Extract and return relevant columns
    result_df = df[required_cols].copy()
    
    # Ensure numeric types for IDs
    result_df["pc11_state_id"] = pd.to_numeric(result_df["pc11_state_id"], errors='coerce')
    result_df["pc11_district_id"] = pd.to_numeric(result_df["pc11_district_id"], errors='coerce')
    
    # Ensure district_name is string
    result_df["district_name"] = result_df["district_name"].astype(str)
    
    # Remove rows with missing geographic IDs
    result_df = result_df.dropna(subset=["pc11_state_id", "pc11_district_id"])
    
    return result_df


def load_niti_mpi(path: str) -> pd.DataFrame:
    """
    Load NITI Aayog MPI data.
    
    Extracts MPI_Score and Headcount_Ratio along with district and state names.
    Handles various column name variations.
    
    Args:
        path: Path to NITI MPI data file (CSV format expected)
        
    Returns:
        DataFrame containing State, District (normalized names), MPI_Score, Headcount_Ratio
        
    Raises:
        FileNotFoundError: If the specified path does not exist
        ValueError: If required columns are missing
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Load CSV
    df = pd.read_csv(path_obj)
    
    # Normalize column names (case-insensitive matching)
    df.columns = df.columns.str.strip()
    
    # Find state and district columns (flexible naming)
    state_col = None
    district_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if state_col is None and ('state' in col_lower or 'st_name' in col_lower):
            state_col = col
        if district_col is None and ('district' in col_lower or 'dist_name' in col_lower):
            district_col = col
    
    if state_col is None:
        raise ValueError("Could not find State column in NITI MPI data")
    if district_col is None:
        raise ValueError("Could not find District column in NITI MPI data")
    
    # Validate required MPI columns
    mpi_score_col = None
    headcount_col = None
    mpi_hcr_col = None  # Some datasets have MPI HCR as a single column
    
    for col in df.columns:
        col_lower = col.lower()
        if mpi_score_col is None and ('mpi_score' in col_lower and 'hcr' not in col_lower):
            mpi_score_col = col
        if headcount_col is None and ('headcount' in col_lower or 'head_count' in col_lower):
            headcount_col = col
        if mpi_hcr_col is None and ('mpi' in col_lower and 'hcr' in col_lower):
            mpi_hcr_col = col
    
    # Handle different data formats
    # Format 1: Separate MPI_Score and Headcount_Ratio columns
    # Format 2: Single MPI_HCR column (Headcount Ratio)
    if mpi_hcr_col:
        # Use MPI_HCR as Headcount_Ratio, set MPI_Score to None
        result_df = pd.DataFrame({
            "State": df[state_col].astype(str),
            "District_2021": df[district_col].astype(str),
            "MPI_Score": None,  # Not available in this format
            "Headcount_Ratio": pd.to_numeric(df[mpi_hcr_col], errors='coerce'),
        })
        warnings.warn(f"Using MPI_HCR column as Headcount_Ratio. MPI_Score not available.")
    elif mpi_score_col and headcount_col:
        # Both columns available
        result_df = pd.DataFrame({
            "State": df[state_col].astype(str),
            "District_2021": df[district_col].astype(str),
            "MPI_Score": pd.to_numeric(df[mpi_score_col], errors='coerce'),
            "Headcount_Ratio": pd.to_numeric(df[headcount_col], errors='coerce'),
        })
    elif headcount_col:
        # Only Headcount_Ratio available
        result_df = pd.DataFrame({
            "State": df[state_col].astype(str),
            "District_2021": df[district_col].astype(str),
            "MPI_Score": None,
            "Headcount_Ratio": pd.to_numeric(df[headcount_col], errors='coerce'),
        })
        warnings.warn("Only Headcount_Ratio found. MPI_Score not available.")
    else:
        raise ValueError("Could not find MPI_Score, Headcount_Ratio, or MPI_HCR column in NITI MPI data")
    
    # Remove rows with missing essential data
    result_df = result_df.dropna(subset=["State", "District_2021"])
    
    return result_df


def merge_shrug_data(secc_df: pd.DataFrame, 
                     viirs_df: pd.DataFrame, 
                     geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform internal merge of SHRUG datasets.
    
    Merges SECC consumption data with VIIRS nighttime lights data using
    geographic identifiers, and attaches district names from geographic keys.
    
    Args:
        secc_df: DataFrame from load_shrug_secc()
        viirs_df: DataFrame from load_shrug_viirs()
        geo_df: DataFrame from load_shrug_geo()
        
    Returns:
        Merged DataFrame with all SHRUG variables and district_name
        
    Raises:
        ValueError: If merge results in empty DataFrame or significant data loss
    """
    # Merge SECC and VIIRS on geographic IDs (inner join)
    merged_df = pd.merge(
        secc_df,
        viirs_df,
        on=["pc11_state_id", "pc11_district_id"],
        how="inner",
        validate="one_to_one"
    )
    
    # Attach district names from geographic keys
    merged_df = pd.merge(
        merged_df,
        geo_df[["pc11_state_id", "pc11_district_id", "district_name"]],
        on=["pc11_state_id", "pc11_district_id"],
        how="left",
        validate="many_to_one"
    )
    
    # Validate merge completeness
    if len(merged_df) == 0:
        raise ValueError("Merge resulted in empty DataFrame - check geographic ID compatibility")
    
    # Check for significant data loss
    min_expected = min(len(secc_df), len(viirs_df))
    if len(merged_df) < min_expected * 0.8:
        warnings.warn(
            f"Merge resulted in {len(merged_df)} rows, expected at least {min_expected * 0.8:.0f} rows. "
            "Some districts may be missing from one dataset."
        )
    
    return merged_df


def normalize_district_name(name: str) -> str:
    """
    Normalize district name for fuzzy matching.
    
    Removes whitespace, converts to uppercase, and removes common spatial
    identifiers like "DISTRICT", "ZILLA", etc.
    
    Args:
        name: Raw district name string
        
    Returns:
        Normalized district name string
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    # Convert to uppercase and strip whitespace
    normalized = name.upper().strip()
    
    # Remove common spatial identifiers
    for identifier in config.SPATIAL_IDENTIFIERS:
        normalized = normalized.replace(identifier, "").strip()
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    return normalized


def fuzzy_match_districts(shrug_df: pd.DataFrame, 
                          mpi_df: pd.DataFrame, 
                          threshold: int = 85) -> pd.DataFrame:
    """
    Match districts between SHRUG (2011) and MPI (2021) using fuzzy string matching.
    
    Uses rapidfuzz to find best matches between district names. Handles cases where
    multiple 2011 districts map to a single 2021 district by aggregating values.
    
    Args:
        shrug_df: Merged SHRUG DataFrame with district_name column
        mpi_df: NITI MPI DataFrame with District_2021 column
        threshold: Minimum similarity score (0-100) for acceptable matches
        
    Returns:
        Merged DataFrame with matched districts and match quality indicators
    """
    # Normalize district names in both datasets
    shrug_df = shrug_df.copy()
    shrug_df["district_name_normalized"] = shrug_df["district_name"].apply(normalize_district_name)
    
    mpi_df = mpi_df.copy()
    mpi_df["District_2021_normalized"] = mpi_df["District_2021"].apply(normalize_district_name)
    
    # Create lookup dictionary for SHRUG districts
    shrug_lookup = {}
    for idx, row in shrug_df.iterrows():
        norm_name = row["district_name_normalized"]
        if norm_name not in shrug_lookup:
            shrug_lookup[norm_name] = []
        shrug_lookup[norm_name].append(idx)
    
    # Match each MPI district to best SHRUG match
    matches = []
    low_confidence_matches = []
    
    for mpi_idx, mpi_row in mpi_df.iterrows():
        mpi_district = mpi_row["District_2021_normalized"]
        mpi_state = mpi_row["State"]
        
        # Filter SHRUG districts by state if possible (for better matching)
        # Note: This assumes state names are consistent; if not, match across all states
        shrug_candidates = list(shrug_lookup.keys())
        
        # Find best match using rapidfuzz
        if shrug_candidates:
            best_match = process.extractOne(
                mpi_district,
                shrug_candidates,
                scorer=fuzz.ratio
            )
            
            if best_match and best_match[1] >= threshold:
                matched_shrug_name = best_match[0]
                similarity_score = best_match[1]
                
                # Get all SHRUG rows matching this normalized name
                matched_indices = shrug_lookup[matched_shrug_name]
                
                # Aggregate values if multiple 2011 districts map to one 2021 district
                matched_shrug_data = shrug_df.loc[matched_indices]
                
                # Calculate aggregated values (mean for continuous variables)
                aggregated_values = {
                    "SHRUG_Cons_PC": matched_shrug_data["secc_cons_pc_rural"].mean(),
                    "VIIRS_Mean": matched_shrug_data["viirs_mean"].mean(),
                    "match_similarity": similarity_score,
                    "num_source_districts": len(matched_indices),
                }
                
                matches.append({
                    "mpi_idx": mpi_idx,
                    "aggregated_values": aggregated_values,
                })
            else:
                if best_match:
                    low_confidence_matches.append({
                        "mpi_district": mpi_row["District_2021"],
                        "best_match": best_match[0] if best_match else None,
                        "similarity": best_match[1] if best_match else 0,
                    })
    
    # Build merged DataFrame
    matched_rows = []
    for match in matches:
        mpi_idx = match["mpi_idx"]
        mpi_row = mpi_df.loc[mpi_idx]
        agg_values = match["aggregated_values"]
        
        matched_rows.append({
            "State": mpi_row["State"],
            "District_2021": mpi_row["District_2021"],
            "SHRUG_Cons_PC": agg_values["SHRUG_Cons_PC"],
            "VIIRS_Mean": agg_values["VIIRS_Mean"],
            "MPI_Score": mpi_row["MPI_Score"],
            "Headcount_Ratio": mpi_row["Headcount_Ratio"],
            "match_similarity": agg_values["match_similarity"],
            "num_source_districts": agg_values["num_source_districts"],
        })
    
    result_df = pd.DataFrame(matched_rows)
    
    # Warn about low confidence matches
    if low_confidence_matches:
        warnings.warn(
            f"Found {len(low_confidence_matches)} districts with similarity < {threshold}. "
            "These were not matched. Review manually if needed."
        )
    
    return result_df


def spatial_crosswalk_geopandas(shrug_df: pd.DataFrame,
                                 mpi_df: pd.DataFrame,
                                 shapefile_2011: str,
                                 shapefile_2021: str,
                                 target_crs: str = "EPSG:32643",
                                 min_intersection_threshold: float = 0.01) -> pd.DataFrame:
    """
    Perform spatial crosswalk using area-weighted interpolation.
    
    Maps SHRUG data (2011 boundaries) to MPI data (2021 boundaries) using
    geographic intersections and area-weighted interpolation.
    
    Args:
        shrug_df: Merged SHRUG DataFrame with geographic IDs
        mpi_df: NITI MPI DataFrame with district names
        shapefile_2011: Path to 2011 district shapefile
        shapefile_2021: Path to 2021 district shapefile
        target_crs: Target CRS for area calculations (default: UTM Zone 43N)
        min_intersection_threshold: Minimum intersection area as fraction of 2021 district
        
    Returns:
        Merged DataFrame with spatially interpolated values
        
    Raises:
        FileNotFoundError: If shapefiles don't exist
        ValueError: If shapefiles lack required attributes
    """
    # Load shapefiles
    if not Path(shapefile_2011).exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_2011}")
    if not Path(shapefile_2021).exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_2021}")
    
    gdf_2011 = gpd.read_file(shapefile_2011)
    gdf_2021 = gpd.read_file(shapefile_2021)
    
    # Ensure CRS consistency
    if gdf_2011.crs is None:
        warnings.warn("2011 shapefile has no CRS, assuming EPSG:4326")
        gdf_2011.set_crs("EPSG:4326", inplace=True)
    
    if gdf_2021.crs is None:
        warnings.warn("2021 shapefile has no CRS, assuming EPSG:4326")
        gdf_2021.set_crs("EPSG:4326", inplace=True)
    
    # Reproject to target CRS for accurate area calculations
    gdf_2011_proj = gdf_2011.to_crs(target_crs)
    gdf_2021_proj = gdf_2021.to_crs(target_crs)
    
    # Identify district identifier columns in shapefiles
    # Common patterns: pc11_district_id, DIST_ID, district_id, etc.
    dist_id_col_2011 = None
    dist_id_col_2021 = None
    dist_name_col_2011 = None
    dist_name_col_2021 = None
    
    for col in gdf_2011.columns:
        col_lower = col.lower()
        if dist_id_col_2011 is None and ('district_id' in col_lower or 'dist_id' in col_lower or 'pc11_district_id' in col_lower):
            dist_id_col_2011 = col
        if dist_name_col_2011 is None and ('district_name' in col_lower or 'dist_name' in col_lower or 'name' in col_lower):
            dist_name_col_2011 = col
    
    for col in gdf_2021.columns:
        col_lower = col.lower()
        if dist_id_col_2021 is None and ('district_id' in col_lower or 'dist_id' in col_lower):
            dist_id_col_2021 = col
        if dist_name_col_2021 is None and ('district' in col_lower and 'name' in col_lower):
            dist_name_col_2021 = col
    
    if dist_id_col_2011 is None:
        raise ValueError("Could not find district ID column in 2011 shapefile")
    if dist_name_col_2021 is None:
        raise ValueError("Could not find district name column in 2021 shapefile")
    
    # Merge SHRUG data with 2011 shapefile
    # Try to match on district ID or name
    gdf_2011_merged = gdf_2011_proj.copy()
    
    # Attempt to merge SHRUG data with shapefile
    merge_successful = False
    if dist_id_col_2011 in gdf_2011.columns:
        # Try merging on district ID
        try:
            gdf_2011_merged = gdf_2011_proj.merge(
                shrug_df,
                left_on=dist_id_col_2011,
                right_on="pc11_district_id",
                how="inner"
            )
            merge_successful = True
        except Exception:
            pass
    
    if not merge_successful and dist_name_col_2011:
        # Try merging on district name
        try:
            gdf_2011_merged = gdf_2011_proj.merge(
                shrug_df,
                left_on=dist_name_col_2011,
                right_on="district_name",
                how="inner"
            )
            merge_successful = True
        except Exception:
            pass
    
    if not merge_successful:
        raise ValueError("Could not merge SHRUG data with 2011 shapefile. Check district ID/name columns.")
    
    # Calculate area-weighted interpolation
    # For each 2021 district, find intersecting 2011 districts and calculate weighted values
    interpolated_rows = []
    
    for idx_2021, row_2021 in gdf_2021_proj.iterrows():
        district_2021_geom = row_2021.geometry
        district_2021_name = row_2021[dist_name_col_2021] if dist_name_col_2021 else f"District_{idx_2021}"
        
        # Find intersecting 2011 districts
        intersecting = gdf_2011_merged[gdf_2011_merged.intersects(district_2021_geom)].copy()
        
        if len(intersecting) == 0:
            # No intersection - skip or flag
            continue
        
        # Calculate intersection areas
        intersecting["intersection_area"] = intersecting.geometry.intersection(district_2021_geom).area
        
        # Filter by minimum intersection threshold
        district_2021_area = district_2021_geom.area
        min_area = district_2021_area * min_intersection_threshold
        intersecting = intersecting[intersecting["intersection_area"] >= min_area]
        
        if len(intersecting) == 0:
            continue
        
        # Calculate area weights (fraction of 2011 district that overlaps with 2021 district)
        intersecting["weight"] = intersecting["intersection_area"] / intersecting.geometry.area
        
        # Calculate area-weighted averages for continuous variables
        total_weight = intersecting["weight"].sum()
        
        if total_weight > 0:
            secc_weighted = (intersecting["secc_cons_pc_rural"] * intersecting["weight"]).sum() / total_weight
            viirs_weighted = (intersecting["viirs_mean"] * intersecting["weight"]).sum() / total_weight
            
            # Find corresponding MPI row
            mpi_row = mpi_df[mpi_df["District_2021"].str.upper().str.strip() == 
                            str(district_2021_name).upper().strip()]
            
            if len(mpi_row) > 0:
                mpi_row = mpi_row.iloc[0]
                interpolated_rows.append({
                    "State": mpi_row["State"],
                    "District_2021": mpi_row["District_2021"],
                    "SHRUG_Cons_PC": secc_weighted,
                    "VIIRS_Mean": viirs_weighted,
                    "MPI_Score": mpi_row["MPI_Score"],
                    "Headcount_Ratio": mpi_row["Headcount_Ratio"],
                    "num_source_districts": len(intersecting),
                    "total_intersection_area": intersecting["intersection_area"].sum(),
                })
    
    result_df = pd.DataFrame(interpolated_rows)
    
    return result_df


def handle_missing_values(df: pd.DataFrame, 
                         drop_unmapped: bool = True,
                         imputation_method: str = "state_mean") -> pd.DataFrame:
    """
    Handle missing values in merged dataset.
    
    Args:
        df: Merged DataFrame with potential NaN values
        drop_unmapped: If True, drop rows with NaN in key columns; if False, impute
        imputation_method: Method for imputation ('state_mean' or 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    key_columns = ["SHRUG_Cons_PC", "VIIRS_Mean", "MPI_Score"]
    
    if drop_unmapped:
        # Drop rows with NaN in key columns
        initial_count = len(df)
        df = df.dropna(subset=key_columns)
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            warnings.warn(f"Dropped {dropped_count} rows with missing values in key columns")
    else:
        # Impute missing values
        if imputation_method == "state_mean":
            for col in key_columns:
                if col in df.columns:
                    state_means = df.groupby("State")[col].transform("mean")
                    df[col] = df[col].fillna(state_means)
        elif imputation_method == "drop":
            df = df.dropna(subset=key_columns)
    
    return df


def export_merged_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Export merged dataset to CSV.
    
    Args:
        df: Final merged DataFrame
        output_path: Path for output CSV file
    """
    # Select final columns in specified order
    final_columns = ["State", "District_2021", "SHRUG_Cons_PC", "VIIRS_Mean", "MPI_Score"]
    
    # Add Headcount_Ratio if available
    if "Headcount_Ratio" in df.columns:
        final_columns.append("Headcount_Ratio")
    
    # Select only columns that exist
    export_df = df[[col for col in final_columns if col in df.columns]].copy()
    
    # Export to CSV
    export_df.to_csv(output_path, index=False)
    print(f"Exported merged data to {output_path}")
    print(f"Total districts: {len(export_df)}")


def run_etl_pipeline(config_dict: Optional[Dict[str, Any]] = None,
                    use_spatial: bool = True) -> pd.DataFrame:
    """
    Main ETL pipeline function orchestrating all steps.
    
    Args:
        config_dict: Optional configuration dictionary (uses defaults if None)
        use_spatial: Whether to attempt spatial crosswalk (falls back to fuzzy matching if False or if shapefiles unavailable)
        
    Returns:
        Final merged DataFrame
        
    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If data validation fails
    """
    # Load configuration
    if config_dict is None:
        config_dict = config.get_config()
    
    config.validate_config(config_dict)
    
    paths = config_dict["paths"]
    
    print("Step 1: Loading datasets...")
    # Load all datasets
    secc_df = load_shrug_secc(paths["shrug_secc"])
    print(f"  Loaded {len(secc_df)} SHRUG SECC records")
    
    viirs_df = load_shrug_viirs(paths["shrug_viirs"])
    print(f"  Loaded {len(viirs_df)} SHRUG VIIRS records")
    
    geo_df = load_shrug_geo(paths["shrug_geo"])
    print(f"  Loaded {len(geo_df)} SHRUG geographic records")
    
    mpi_df = load_niti_mpi(paths["niti_mpi"])
    print(f"  Loaded {len(mpi_df)} NITI MPI records")
    
    print("\nStep 2: Merging SHRUG data internally...")
    shrug_merged = merge_shrug_data(secc_df, viirs_df, geo_df)
    print(f"  Merged SHRUG data: {len(shrug_merged)} districts")
    
    print("\nStep 3: Performing spatial crosswalk...")
    merged_df = None
    
    if use_spatial:
        shapefile_2011 = paths["shapefile_2011"]
        shapefile_2021 = paths["shapefile_2021"]
        
        if Path(shapefile_2011).exists() and Path(shapefile_2021).exists():
            try:
                merged_df = spatial_crosswalk_geopandas(
                    shrug_merged,
                    mpi_df,
                    shapefile_2011,
                    shapefile_2021,
                    target_crs=config_dict["target_crs"],
                    min_intersection_threshold=config_dict["min_intersection_threshold"]
                )
                print(f"  Spatial crosswalk completed: {len(merged_df)} districts matched")
            except Exception as e:
                warnings.warn(f"Spatial crosswalk failed: {e}. Falling back to fuzzy matching.")
                merged_df = None
        else:
            warnings.warn("Shapefiles not found. Falling back to fuzzy matching.")
            merged_df = None
    
    # Fall back to fuzzy matching if spatial failed or was disabled
    if merged_df is None:
        print("\nStep 3 (fallback): Using fuzzy string matching...")
        merged_df = fuzzy_match_districts(
            shrug_merged,
            mpi_df,
            threshold=config_dict["fuzzy_match_threshold"]
        )
        print(f"  Fuzzy matching completed: {len(merged_df)} districts matched")
    
    print("\nStep 4: Handling missing values...")
    merged_df = handle_missing_values(
        merged_df,
        drop_unmapped=config_dict["drop_unmapped_districts"],
        imputation_method=config_dict["imputation_method"]
    )
    print(f"  Final dataset: {len(merged_df)} districts")
    
    print("\nStep 5: Exporting results...")
    export_merged_data(merged_df, paths["output"])
    
    return merged_df


if __name__ == "__main__":
    # Example usage
    import config as cfg
    
    # Run pipeline with default configuration
    result_df = run_etl_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"\nSummary statistics:")
    print(result_df[["SHRUG_Cons_PC", "VIIRS_Mean", "MPI_Score"]].describe())

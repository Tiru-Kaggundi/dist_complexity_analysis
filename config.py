"""
Configuration module for India District Data ETL Pipeline.

This module contains default data paths and configurable parameters
for the ETL pipeline that merges SHRUG and NITI Aayog MPI datasets.
"""

from typing import Dict, Any
from pathlib import Path

# Default data paths following SHRUG 2.1 directory structure
DEFAULT_DATA_PATHS: Dict[str, str] = {
    "shrug_secc": "data/shrug_secc",
    "shrug_viirs": "data/shrug_viirs",
    "shrug_geo": "data/shrug_geo",
    "niti_mpi": "data/niti_mpi",
    "shapefile_2011": "data/shapefiles/india_districts_2011.shp",
    "shapefile_2021": "data/shapefiles/india_districts_2021.shp",
    "output": "india_district_macro_merged.csv",
}

# Configurable pipeline parameters
DEFAULT_CONFIG: Dict[str, Any] = {
    # Handling of unmapped districts (newly formed districts without matches)
    "drop_unmapped_districts": True,  # If True, drop rows with NaN; if False, impute with state means
    
    # Fuzzy matching threshold (0-100)
    "fuzzy_match_threshold": 85,  # Minimum similarity score for fuzzy matches
    
    # Imputation method for missing values (only used if drop_unmapped_districts=False)
    "imputation_method": "state_mean",  # Options: 'state_mean', 'drop'
    
    # CRS for spatial operations (UTM Zone 43N covers most of India)
    "target_crs": "EPSG:32643",  # UTM Zone 43N for accurate area calculations
    
    # Minimum intersection area threshold (as fraction of 2021 district area)
    "min_intersection_threshold": 0.01,  # 1% minimum overlap to consider valid
}

# Required columns for each dataset
REQUIRED_COLUMNS: Dict[str, list] = {
    "shrug_secc": ["pc11_state_id", "pc11_district_id", "secc_cons_pc_rural"],
    "shrug_viirs": ["pc11_state_id", "pc11_district_id", "viirs_mean"],
    "shrug_geo": ["pc11_state_id", "pc11_district_id", "district_name"],
    "niti_mpi": ["MPI_Score", "Headcount_Ratio"],  # State and District columns may vary
}

# Common spatial identifiers to remove during normalization
SPATIAL_IDENTIFIERS: list = [
    "DISTRICT",
    "ZILLA",
    "ZILA",
    "JILA",
    "JILLA",
    "DIST",
]


def get_config(custom_paths: Dict[str, str] = None, 
               custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration dictionary with optional custom overrides.
    
    Args:
        custom_paths: Dictionary of custom data paths to override defaults
        custom_params: Dictionary of custom parameters to override defaults
        
    Returns:
        Combined configuration dictionary
    """
    config = {
        "paths": DEFAULT_DATA_PATHS.copy(),
        **DEFAULT_CONFIG.copy(),
    }
    
    if custom_paths:
        config["paths"].update(custom_paths)
    
    if custom_params:
        config.update(custom_params)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    if not (0 <= config["fuzzy_match_threshold"] <= 100):
        raise ValueError("fuzzy_match_threshold must be between 0 and 100")
    
    if config["imputation_method"] not in ["state_mean", "drop"]:
        raise ValueError("imputation_method must be 'state_mean' or 'drop'")
    
    if config["min_intersection_threshold"] < 0 or config["min_intersection_threshold"] > 1:
        raise ValueError("min_intersection_threshold must be between 0 and 1")

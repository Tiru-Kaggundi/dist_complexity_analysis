# India District Data ETL Pipeline

A robust Python ETL pipeline to ingest, clean, harmonize, and merge three distinct sub-national datasets for India into a single master cross-sectional dataset at the district level.

## Overview

This pipeline merges:
1. **SHRUG (Development Data Lab) - Consumption/GDP Proxy**: Rural per capita consumption proxy (`secc_cons_pc_rural`) from SECC module (Census 2011 boundaries)
2. **SHRUG - Remote Sensing (VIIRS)**: Mean nighttime light radiance (`viirs_mean`) (Census 2011 boundaries)
3. **NITI Aayog National MPI (2023 baseline via NFHS-5)**: `MPI_Score` and `Headcount_Ratio` (~2021 administrative boundaries, 700+ districts)

The primary challenge addressed is the boundary mismatch between Census 2011 (~640 districts) and MPI 2021 (~707 districts), handled through:
- **Primary method**: Area-weighted spatial interpolation using `geopandas`
- **Fallback method**: Deterministic fuzzy string matching using `rapidfuzz`

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `pandas>=2.0.0` - Data manipulation
- `geopandas>=0.14.0` - Geospatial operations
- `numpy>=1.24.0` - Numerical computations
- `rapidfuzz>=3.0.0` - Fuzzy string matching
- `pyproj>=3.6.0` - CRS handling
- `shapely>=2.0.0` - Geospatial geometry operations

## Data Structure Requirements

### Directory Structure

Organize your data files as follows:

```
data/
├── shrug_secc/          # SHRUG SECC consumption data
│   └── *.csv or *.dta
├── shrug_viirs/         # SHRUG VIIRS nighttime lights data
│   └── *.csv or *.dta
├── shrug_geo/           # SHRUG geographic keys
│   └── *.csv or *.dta
├── niti_mpi/            # NITI Aayog MPI data
│   └── *.csv
└── shapefiles/          # Optional: district shapefiles
    ├── india_districts_2011.shp
    ├── india_districts_2011.shx
    ├── india_districts_2011.dbf
    ├── india_districts_2021.shp
    ├── india_districts_2021.shx
    └── india_districts_2021.dbf
```

### Required Columns

#### SHRUG SECC Data
- `pc11_state_id` - State ID (Census 2011)
- `pc11_district_id` - District ID (Census 2011)
- `secc_cons_pc_rural` - Rural per capita consumption proxy

#### SHRUG VIIRS Data
- `pc11_state_id` - State ID (Census 2011)
- `pc11_district_id` - District ID (Census 2011)
- `viirs_mean` - Mean nighttime light radiance

#### SHRUG Geographic Keys
- `pc11_state_id` - State ID (Census 2011)
- `pc11_district_id` - District ID (Census 2011)
- `district_name` - District name string

#### NITI Aayog MPI Data
- State column (flexible naming: "State", "ST_NAME", etc.)
- District column (flexible naming: "District", "DIST_NAME", etc.)
- `MPI_Score` or column containing "MPI" in name
- `Headcount_Ratio` or column containing "Headcount" in name

#### Shapefiles (Optional)
- 2011 shapefile: Must contain district ID or name column
- 2021 shapefile: Must contain district name column matching MPI data

## Usage

### Basic Usage

```python
from etl_pipeline import run_etl_pipeline
import config

# Run with default configuration
result_df = run_etl_pipeline()

# Access the merged dataset
print(result_df.head())
```

### Custom Configuration

```python
from etl_pipeline import run_etl_pipeline
import config

# Create custom configuration
custom_config = config.get_config(
    custom_paths={
        "shrug_secc": "path/to/secc/data",
        "shrug_viirs": "path/to/viirs/data",
        "shrug_geo": "path/to/geo/data",
        "niti_mpi": "path/to/mpi/data.csv",
        "shapefile_2011": "path/to/2011.shp",
        "shapefile_2021": "path/to/2021.shp",
        "output": "custom_output.csv",
    },
    custom_params={
        "drop_unmapped_districts": False,  # Impute instead of dropping
        "fuzzy_match_threshold": 90,  # Higher threshold for stricter matching
        "imputation_method": "state_mean",
    }
)

# Run pipeline
result_df = run_etl_pipeline(config_dict=custom_config)
```

### Using Individual Functions

```python
from etl_pipeline import (
    load_shrug_secc,
    load_shrug_viirs,
    load_shrug_geo,
    load_niti_mpi,
    merge_shrug_data,
    spatial_crosswalk_geopandas,
    fuzzy_match_districts,
)

# Load individual datasets
secc_df = load_shrug_secc("data/shrug_secc")
viirs_df = load_shrug_viirs("data/shrug_viirs")
geo_df = load_shrug_geo("data/shrug_geo")
mpi_df = load_niti_mpi("data/niti_mpi/mpi_data.csv")

# Merge SHRUG data
shrug_merged = merge_shrug_data(secc_df, viirs_df, geo_df)

# Option 1: Spatial crosswalk (if shapefiles available)
merged_df = spatial_crosswalk_geopandas(
    shrug_merged,
    mpi_df,
    "data/shapefiles/india_districts_2011.shp",
    "data/shapefiles/india_districts_2021.shp"
)

# Option 2: Fuzzy matching (fallback)
merged_df = fuzzy_match_districts(shrug_merged, mpi_df, threshold=85)
```

### Command Line Usage

```bash
python etl_pipeline.py
```

This will run the pipeline with default configuration and export results to `india_district_macro_merged.csv`.

## Data Sources and Citations

- **SHRUG (district boundaries and related variables)**  
  If you use SHRUG-based data or the district boundaries in publications, please cite:

  > Asher, S., T. Lunt, R. Matsuura, and P. Novosad (2021).  
  > *Development research at high geographic resolution: an analysis of night-lights, firms, and poverty in India using the SHRUG open data platform.*  
  > World Bank Economic Review.

  BibTeX:

  ```bibtex
  @article{almn2021,
    title={Development research at high geographic resolution: an analysis of night-lights, firms, and poverty in India using the shrug open data platform},
    author={Asher, Sam and Lunt, Tobias and Matsuura, Ryu and Novosad, Paul},
    journal={The World Bank Economic Review},
    volume={35},
    number={4},
    year={2021},
    publisher={Oxford University Press}
  }
  ```

## Configuration Options

Edit `config.py` or pass custom parameters to modify pipeline behavior:

### Key Parameters

- **`drop_unmapped_districts`** (bool, default: `True`)
  - If `True`: Drop districts that cannot be matched
  - If `False`: Impute missing values using state-level means

- **`fuzzy_match_threshold`** (int, default: `85`)
  - Minimum similarity score (0-100) for fuzzy string matching
  - Lower values allow more lenient matches but may introduce errors

- **`imputation_method`** (str, default: `"state_mean"`)
  - `"state_mean"`: Impute missing values with state-level averages
  - `"drop"`: Drop rows with missing values

- **`target_crs`** (str, default: `"EPSG:32643"`)
  - Coordinate Reference System for area calculations
  - UTM Zone 43N (EPSG:32643) is recommended for India

- **`min_intersection_threshold`** (float, default: `0.01`)
  - Minimum intersection area as fraction of 2021 district area
  - Used to filter out very small overlaps in spatial crosswalk

## Output

The pipeline generates a CSV file (`india_district_macro_merged.csv` by default) with the following columns:

- `State` - State name
- `District_2021` - District name (2021 boundaries)
- `SHRUG_Cons_PC` - Rural per capita consumption proxy
- `VIIRS_Mean` - Mean nighttime light radiance
- `MPI_Score` - Multidimensional Poverty Index score
- `Headcount_Ratio` - Headcount ratio (if available)

## How It Works

### 1. Data Ingestion
- Loads SHRUG SECC, VIIRS, and geographic keys data
- Supports both CSV and Stata (.dta) formats
- Validates required columns and data types

### 2. Internal SHRUG Merge
- Performs inner join on `pc11_state_id` and `pc11_district_id`
- Attaches district names from geographic keys
- Validates merge completeness

### 3. Spatial Crosswalk (Primary Method)
- Loads 2011 and 2021 district shapefiles
- Computes spatial intersections between boundaries
- Calculates area-weighted interpolation:
  - For each 2021 district, finds all intersecting 2011 districts
  - Calculates weights based on intersection area
  - Computes weighted averages: `Σ(value_i × area_i) / Σ(area_i)`
- Handles cases where multiple 2011 districts merge into one 2021 district

### 4. Fuzzy Matching (Fallback Method)
- Normalizes district names (uppercase, remove spatial identifiers)
- Uses `rapidfuzz` to find best matches between datasets
- Aggregates values when multiple 2011 districts map to one 2021 district
- Flags low-confidence matches (< threshold) for manual review

### 5. Missing Value Handling
- Based on configuration, either drops or imputes missing values
- Imputation uses state-level means for continuous variables

## Troubleshooting

### Common Issues

#### 1. "FileNotFoundError: Path not found"
- **Solution**: Verify data paths in `config.py` match your directory structure
- Check that files exist in the specified locations

#### 2. "Missing required columns"
- **Solution**: Verify column names match expected names (case-sensitive)
- Check data documentation for exact column names
- The pipeline attempts to handle common variations for MPI data

#### 3. "Spatial crosswalk failed"
- **Solution**: 
  - Verify shapefiles exist and are valid
  - Check that shapefiles contain district ID/name columns
  - Ensure shapefiles have valid CRS (Coordinate Reference System)
  - Pipeline will automatically fall back to fuzzy matching

#### 4. "Merge resulted in empty DataFrame"
- **Solution**: 
  - Verify geographic IDs are consistent across SHRUG datasets
  - Check for data type mismatches (string vs numeric IDs)
  - Ensure all datasets use Census 2011 district IDs

#### 5. Low match rates in fuzzy matching
- **Solution**: 
  - Lower `fuzzy_match_threshold` (but may introduce errors)
  - Review district name normalization
  - Check for spelling variations or special characters
  - Consider manual mapping for unmatched districts

### Debugging Tips

1. **Check intermediate results**: Use individual functions to inspect data at each step
2. **Validate geographic IDs**: Ensure IDs are consistent and properly formatted
3. **Review warnings**: The pipeline logs warnings for data quality issues
4. **Test with sample data**: Start with a subset of districts to verify pipeline logic

## Technical Details

### Area-Weighted Interpolation Formula

For continuous variables (means, ratios):
```
weighted_value = Σ(value_i × intersection_area_i) / Σ(intersection_area_i)
```

For absolute values:
```
weighted_value = Σ(value_i × weight_i)
where weight_i = intersection_area_i / total_2011_district_area_i
```

### CRS Handling

- Shapefiles are reprojected to UTM Zone 43N (EPSG:32643) for accurate area calculations
- Original CRS is preserved in output if needed
- If shapefiles lack CRS, EPSG:4326 (WGS84) is assumed

### String Normalization

District names are normalized by:
1. Converting to uppercase
2. Stripping whitespace
3. Removing spatial identifiers: "DISTRICT", "ZILLA", "ZILA", "JILA", "JILLA", "DIST"

## License

This pipeline is provided as-is for research purposes.

## Citation

If you use this pipeline in your research, please cite:
- SHRUG: Development Data Lab
- NITI Aayog: National Multidimensional Poverty Index Baseline Report 2023

## Contact

For issues, questions, or contributions, please refer to the project repository.

# Plan: Adding Luminosity/Light Data to Enriched Dataset

## Overview

Add VIIRS nighttime lights (luminosity) data as the 4th column to the enriched dataset (`dists_2025_full.csv`).

## Current Dataset Context

**Base Dataset:** `dists_2025_full.csv`
- **Districts:** 632 unique districts from 2025 trade data
- **States:** 36 states/UTs
- **Year:** 2025 (trade data)
- **Current Columns:** State, District, HS Code, Export values, PCI, ECI

**Target:** Add `VIIRS_Mean` column (mean nighttime light radiance)

## Luminosity Data Options

### Option 1: SHRUG VIIRS Data (Recommended - Simplest)

**Source:** Development Data Lab SHRUG 2.1 Pakora
- **URL:** https://www.devdatalab.org/shrug_download/
- **Coverage:** 2012-2021 (annual)
- **Year to Use:** 2021 (matches MPI data year)
- **Geographic Level:** District (pc11_district_id)
- **Variable:** `viirs_mean` (mean nighttime light radiance)
- **Boundaries:** Census 2011 district boundaries

**Access:**
1. Visit: https://www.devdatalab.org/shrug_download/
2. Agree to Creative Commons license
3. Navigate to: Remote Sensing → Night-time lights → VIIRS
4. Download district-level file
5. Expected file: `shrug_nl_viirs_pc11dist.dta` or `.csv`

**Advantages:**
- Pre-processed and ready to use
- Consistent methodology
- Well-documented
- Matches Census 2011 boundaries (same as MPI data)

**Disadvantages:**
- Requires manual download
- May require registration/login

### Option 2: Direct VIIRS Processing with Geopandas (More Flexible)

**Source:** Colorado School of Mines EOG / Google Earth Engine
- **URL:** https://eogdata.mines.edu/products/vnl/ or Google Earth Engine
- **Coverage:** Up to 2024 (can get 2021 specifically)
- **Format:** GeoTIFF raster files
- **Process:** Download raster → Extract zonal statistics using district shapefiles

**Advantages:**
- Can get exact year needed (2021)
- Full control over processing
- Can match exact district boundaries
- More recent data available if needed later

**Disadvantages:**
- Requires geospatial processing
- Need district shapefiles
- More complex implementation
- Larger file downloads

**Recommendation:** Start with Option 1 (SHRUG), fall back to Option 2 if SHRUG unavailable.

## Implementation Plan

### Step 1: Download VIIRS Data

#### If Using SHRUG (Option 1):
1. Download from SHRUG portal
2. Save to: `data/shrug_viirs/`
3. Expected format: CSV or Stata file with district-level VIIRS values

#### If Using Direct Processing (Option 2):
1. Download VIIRS 2021 annual composite GeoTIFF
2. Download India district shapefile (Census 2011 or current)
3. Use geopandas to extract zonal statistics

### Step 2: Process and Match Districts

**Matching Strategy:**
- **Primary Key:** Trade dataset districts (632 districts)
- **Matching Logic:**
  1. Exact match on State + District name
  2. Fuzzy match for name variations (using rapidfuzz)
  3. If multiple SHRUG districts match one trade district: **Take average**
  4. If one SHRUG district matches multiple trade districts: **Replicate value**
  5. If no match found: Set to NaN (or use state average as fallback)

**Special Cases:**
- Districts with "UNSPECIFIED" in trade data: Skip or use state average
- Newly formed districts (not in 2011): Try fuzzy match, else NaN
- Renamed districts: Use fuzzy matching with threshold

### Step 3: Add Luminosity Column

**Function:** `add_luminosity_column(base_df, viirs_df)`
- Match districts using State + District combination
- Handle multiple matches (average)
- Handle missing matches (NaN or state average)
- Add `VIIRS_Mean` column to base dataset

### Step 4: Update Enriched Dataset

**Final Dataset Columns:**
1. State
2. District
3. HS Code
4. Commodity Description
5. Export_USD
6. Export_INR
7. PCI (Product Complexity Index)
8. ECI (Economic Complexity Index)
9. MPI_HCR (Multidimensional Poverty Index Headcount Ratio) - **To be added**
10. VIIRS_Mean (Mean nighttime light radiance) - **To be added**

## Implementation Module

### `luminosity_merger.py`

**Functions:**

1. `load_shrug_viirs(path: str, year: int = 2021) -> pd.DataFrame`
   - Load SHRUG VIIRS data
   - Extract data for specified year (default: 2021)
   - Return DataFrame with State, District, VIIRS_Mean
   - Handle both CSV and Stata formats

2. `process_viirs_raster(raster_path: str, shapefile_path: str, year: int = 2021) -> pd.DataFrame`
   - Alternative: Process VIIRS raster directly
   - Extract zonal statistics (mean) for each district
   - Return DataFrame with State, District, VIIRS_Mean
   - Uses geopandas and rasterio

3. `normalize_district_names(df: pd.DataFrame, district_col: str = 'District') -> pd.DataFrame`
   - Normalize district names for matching
   - Remove common suffixes, standardize case
   - Handle special characters

4. `match_districts_for_luminosity(trade_df: pd.DataFrame, viirs_df: pd.DataFrame, fuzzy_threshold: int = 85) -> pd.DataFrame`
   - Match districts between trade dataset and VIIRS data
   - Use State + District combination
   - Fuzzy matching for name variations
   - Handle multiple matches (average)
   - Return matched DataFrame

5. `add_luminosity_column(base_df: pd.DataFrame, viirs_df: pd.DataFrame, method: str = 'fuzzy') -> pd.DataFrame`
   - Merge VIIRS data into base dataset
   - Add VIIRS_Mean column
   - Handle missing values (NaN or state average)

6. `process_and_add_luminosity(input_path: str, output_path: str, viirs_path: str, year: int = 2021, use_raster: bool = False) -> pd.DataFrame`
   - Main function: Load trade dataset, merge luminosity, save

## Matching Logic Details

### Exact Matching
```python
# Match on State + District
matched = trade_df.merge(
    viirs_df,
    on=['State', 'District'],
    how='left'
)
```

### Fuzzy Matching (for unmatched districts)
```python
# For each unmatched trade district:
# 1. Filter VIIRS districts in same state
# 2. Find best match using rapidfuzz
# 3. If similarity >= threshold, use that match
# 4. If multiple matches, average the values
```

### Multiple Match Handling
- **Case 1:** Multiple SHRUG districts → One trade district
  - **Solution:** Average the VIIRS values
  
- **Case 2:** One SHRUG district → Multiple trade districts
  - **Solution:** Replicate the same VIIRS value to all

- **Case 3:** No match found
  - **Solution:** Set to NaN (or use state average as fallback)

## Data Quality Considerations

### Temporal Alignment
- **Trade Data:** 2025
- **VIIRS Data:** 2021 (matches MPI year)
- **Gap:** 4 years (acceptable for cross-sectional analysis)

### Boundary Alignment
- **Trade Data:** 2025 administrative boundaries (632 districts)
- **VIIRS Data:** Census 2011 boundaries (~640 districts)
- **Mismatch:** Some districts may not match exactly
- **Solution:** Fuzzy matching + averaging for splits/mergers

### Missing Data
- Districts without VIIRS data:
  - Primary: Set to NaN
  - Fallback: Use state average (configurable)

## Expected Output

**Dataset:** `dists_2025_full.csv` (updated)

**New Column:**
- `VIIRS_Mean`: Mean nighttime light radiance
- Typical range: 0-100+ (higher = more luminosity)
- Coverage: ~95-98% of districts (some newly formed districts may be missing)

## Implementation Steps

1. **Download SHRUG VIIRS data** (manual)
   - Or implement direct raster processing (if preferred)

2. **Create `luminosity_merger.py` module**
   - Implement loading, matching, and merging functions

3. **Test matching logic**
   - Verify district name matching
   - Test multiple match scenarios
   - Validate fuzzy matching threshold

4. **Run merger**
   - Add VIIRS_Mean column to enriched dataset
   - Handle missing values appropriately

5. **Validate results**
   - Check coverage (% districts with data)
   - Verify reasonable values
   - Check for outliers

## Questions Resolved

1. ✅ **Data Source:** SHRUG VIIRS (2021) - matches MPI year
2. ✅ **Year:** 2021 (latest available, matches MPI)
3. ✅ **Matching:** Trade districts as key, fuzzy match, average if multiple matches
4. ✅ **Missing Districts:** Set to NaN (or state average as fallback)

## Next Steps

1. Download SHRUG VIIRS data or implement direct processing
2. Create luminosity merger module
3. Test and validate matching
4. Add VIIRS_Mean column to dataset

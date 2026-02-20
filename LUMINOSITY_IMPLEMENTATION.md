# Luminosity Data Implementation Summary

## What Was Built

A complete module (`luminosity_merger.py`) to add VIIRS nighttime lights (luminosity) data to your enriched trade dataset.

## Key Features

1. **Flexible Data Sources:**
   - SHRUG pre-processed data (CSV/Stata) - **Recommended**
   - Direct raster processing with geopandas (alternative)

2. **Smart District Matching:**
   - Exact matching on State + District
   - Fuzzy matching for name variations (using rapidfuzz)
   - Handles multiple matches (averages when multiple VIIRS districts match one trade district)
   - Replicates values when one VIIRS district matches multiple trade districts

3. **Robust Handling:**
   - Normalizes district names for better matching
   - Handles missing data (NaN, state average, or zero - configurable)
   - Reports matching statistics

## Files Created

1. **`luminosity_merger.py`** - Main merger module
2. **`download_viirs_data.py`** - Download helper with instructions
3. **`PLAN_LUMINOSITY_DATA.md`** - Detailed implementation plan

## Quick Start

### Step 1: Download SHRUG VIIRS Data

Run the helper script for instructions:
```bash
python download_viirs_data.py
```

Or manually:
1. Visit: https://www.devdatalab.org/shrug_download/
2. Navigate to: Remote Sensing → Night-time lights → VIIRS
3. Download district-level file (e.g., `shrug_nl_viirs_pc11dist.dta`)
4. Save to: `data/shrug_viirs/shrug_nl_viirs_pc11dist.dta`

### Step 2: Run the Merger

**Command Line:**
```bash
python luminosity_merger.py dists_2025_full.csv dists_2025_full.csv data/shrug_viirs/shrug_nl_viirs_pc11dist.dta
```

**Python API:**
```python
from luminosity_merger import process_and_add_luminosity

process_and_add_luminosity(
    input_path='dists_2025_full.csv',
    output_path='dists_2025_full.csv',
    viirs_path='data/shrug_viirs/shrug_nl_viirs_pc11dist.dta',
    year=2021,
    fuzzy_threshold=85,
    fill_missing='nan'  # or 'state_mean' or 'zero'
)
```

## Configuration Options

- **`year`**: Year of VIIRS data (default: 2021, matches MPI)
- **`fuzzy_threshold`**: Minimum similarity score for fuzzy matching (0-100, default: 85)
- **`fill_missing`**: How to handle unmatched districts:
  - `'nan'` - Leave as NaN (default)
  - `'state_mean'` - Fill with state average
  - `'zero'` - Fill with zero

## Matching Strategy

Your requirements:
- ✅ Trade districts are the key (632 districts)
- ✅ Closest match if no exact match
- ✅ Average if multiple VIIRS districts match one trade district
- ✅ Replicate if one VIIRS district matches multiple trade districts

## Expected Output

The enriched dataset will have a new column:
- **`VIIRS_Mean`**: Mean nighttime light radiance
  - Typical range: 0-100+ (higher = more luminosity)
  - Coverage: ~95-98% of districts (some newly formed districts may be missing)

## Alternative: Direct Raster Processing

If you prefer to process VIIRS rasters directly (requires geopandas):

```python
from luminosity_merger import process_and_add_luminosity

process_and_add_luminosity(
    input_path='dists_2025_full.csv',
    output_path='dists_2025_full.csv',
    raster_path='path/to/viirs_2021.tif',
    shapefile_path='path/to/india_districts.shp',
    year=2021
)
```

## Dependencies

Already in `requirements.txt`:
- pandas
- numpy
- rapidfuzz

Optional (for raster processing):
- geopandas
- rasterio

If using Stata files (.dta):
- pyreadstat (install: `pip install pyreadstat`)

## Next Steps

1. Download SHRUG VIIRS data
2. Run the merger to add `VIIRS_Mean` column
3. Verify matching results (check unmatched districts)
4. Proceed with MPI column addition (next task)

## Questions?

The module includes comprehensive error handling and reporting. Check the console output for:
- Matching statistics (% matched)
- List of unmatched districts
- Data quality metrics

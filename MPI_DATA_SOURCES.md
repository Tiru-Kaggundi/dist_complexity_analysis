# NITI Aayog MPI Data Sources and Download Guide

## Summary

Successfully identified and downloaded district-level Multidimensional Poverty Index (MPI) data from NITI Aayog sources.

## Data Sources Found

### 1. GitHub Repository (✅ Recommended - Already Downloaded)

**Source:** https://github.com/tam0w/poverty_data

**Dataset:** `DATESET.csv` - Pre-processed district-level MPI data extracted from NITI Aayog's 2023 MPI report

**Advantages:**
- Already processed and cleaned
- District-level granularity (638 districts)
- Includes MPI Headcount Ratio (HCR) values
- Easy to download programmatically

**Download Status:** ✅ Downloaded to `data/niti_mpi/mpi_district_data.csv`

**Data Format:**
- Columns: `State`, `District`, `MPI_HCR`
- 638 districts across 35 states/UTs
- MPI HCR range: 0.00% - 74.35%
- Based on Census 2011 district boundaries

### 2. NITI Aayog SDG Dashboard

**URL:** https://sdgindiaindex.niti.gov.in

**Access:** Interactive dashboard with MPI data visualization

**Note:** Requires manual navigation and may not have direct CSV export

### 3. Open Government Data Platform (data.gov.in)

**URL:** https://www.data.gov.in/resource/stateut-wise-details-headcount-ratio-intensity-and-multi-dimensional-poverty-index-mpi

**Data Level:** State/UT level (not district level)

**Format:** CSV download available

**Limitation:** Only state/UT aggregates, not district-level breakdowns

### 4. Official NITI Aayog Reports (PDF)

**2023 Progress Review:**
- URL: https://www.niti.gov.in/sites/default/files/2023-08/India-National-Multidimentional-Poverty-Index-2023.pdf
- Based on NFHS-5 (2019-21) data
- Contains district-level analysis in PDF format

**2021 Baseline Report:**
- URL: https://www.niti.gov.in/sites/default/files/2021-11/National_MPI_India-11242021.pdf
- Based on NFHS-4 (2015-16) data
- Contains district-level estimates

## Downloaded Data Details

**File Location:** `data/niti_mpi/mpi_district_data.csv`

**Statistics:**
- Total districts: 638
- States/UTs: 35
- MPI HCR mean: 25.37%
- MPI HCR median: 23.62%
- MPI HCR range: 0.00% - 74.35%

**Sample Data:**
```
State,District,MPI_HCR
ANDHRA PRADESH,Adilabad,27.12
UTTAR PRADESH,Agra,32.83
GUJARAT,Ahmadabad,5.85
MAHARASHTRA,Ahmadnagar,15.40
MIZORAM,Aizawl,1.76
```

## Usage

### Download Script

Use the provided `download_mpi_data.py` script:

```bash
# Download from GitHub (recommended)
python download_mpi_data.py --method github

# Output files:
# - data/niti_mpi/mpi_raw_github.csv (raw data)
# - data/niti_mpi/mpi_district_data.csv (processed data)
```

### Integration with ETL Pipeline

The `load_niti_mpi()` function in `etl_pipeline.py` has been updated to handle this data format:

```python
from etl_pipeline import load_niti_mpi

# Load the downloaded MPI data
mpi_df = load_niti_mpi("data/niti_mpi/mpi_district_data.csv")

# Returns DataFrame with columns:
# - State
# - District_2021
# - MPI_Score (None - not available in this dataset)
# - Headcount_Ratio (from MPI_HCR column)
```

## Important Notes

1. **District Boundaries:** The downloaded data uses Census 2011 district boundaries (638 districts), matching the SHRUG data baseline.

2. **MPI Score vs Headcount Ratio:** 
   - The GitHub dataset provides **Headcount Ratio (HCR)** only
   - MPI Score is not included in this dataset
   - HCR represents the percentage of population that is multidimensionally poor

3. **Data Year:** Based on NFHS-5 (2019-21) data, as reported in NITI Aayog's 2023 MPI report.

4. **Boundary Mismatch:** 
   - This dataset: 638 districts (Census 2011)
   - NITI Aayog 2023 report: ~707 districts (2021 boundaries)
   - The ETL pipeline handles this mismatch using spatial crosswalk or fuzzy matching

## Next Steps

1. ✅ MPI data downloaded and ready
2. ⏳ Download HS4 trade data (manual process as mentioned)
3. ⏳ Download OEC PCI data for HS4 products
4. ⏳ Run ECI calculation
5. ⏳ Merge all datasets and generate visualizations

## References

- NITI Aayog MPI 2023 Report: https://www.niti.gov.in/whats-new/national-multidimentional-poverty-index-2023
- GitHub Repository: https://github.com/tam0w/poverty_data
- SDG Dashboard: https://sdgindiaindex.niti.gov.in
- OGD Platform: https://www.data.gov.in

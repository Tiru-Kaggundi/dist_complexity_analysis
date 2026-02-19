# ECI Analysis and Visualization Extension

**Plan for future reference**  
*Last updated: February 2025*

## Overview

Extend the existing ETL pipeline (`etl_pipeline.py`) to:

1. Calculate Economic Complexity Index (ECI) at district level using OEC methodology
2. Merge ECI with existing merged dataset (GDP proxy, MPI, luminosity)
3. Generate correlation analysis and visualizations (scatter plots, correlation matrices, spatial maps)

## Project Structure

```
OEC_stuff/
├── etl_pipeline.py              # Existing ETL pipeline (extend)
├── config.py                     # Existing config (extend)
├── eci_calculator.py            # NEW: ECI calculation module
├── visualization.py             # NEW: Visualization and correlation analysis
├── analysis_pipeline.py         # NEW: Main analysis orchestrator
├── requirements.txt              # Update with visualization dependencies
└── data/
    ├── ... (existing data directories)
    └── eci_source/              # NEW: HS4-level international trade data
        └── [district-level export data by HS4 product codes]
```

## Implementation Plan

### 1. ECI Calculation Module (`eci_calculator.py`)

#### 1.1 HS4 Trade Data Loader

- `load_hs4_trade_data(path: str) -> pd.DataFrame`
  - Load district-level international trade data with HS4 product codes
  - Expected columns:
    - District identifier (district name or ID)
    - HS4 product code (4-digit Harmonized System code)
    - Export value (in USD or INR)
    - Optional: Year, State
  - Handle CSV/Excel formats
  - Validate HS4 codes are 4-digit strings/numeric
  - Aggregate multiple years if present (use most recent or average)

#### 1.2 OEC PCI Data Loader

- `load_oec_pci_data(path: Optional[str] = None, use_api: bool = False, api_token: Optional[str] = None) -> pd.DataFrame`
  - Load Product Complexity Index (PCI) values for HS4 products using OEC methodology
  - **Option 1: From local file** (if `path` provided):
    - Load pre-downloaded PCI data (CSV/Excel)
    - Expected columns: HS4 code, PCI value
  - **Option 2: From OEC API** (if `use_api=True`):
    - Fetch PCI rankings from OEC API: `https://api-v2.oec.world/tesseract/data.jsonrecords`
    - Use cube: `hs92_4d` or appropriate HS revision for the time period
    - Drilldown: `HS4` dimension
    - Measure: `PCI` (Product Complexity Index)
    - Cache results locally to avoid repeated API calls
  - Return DataFrame with HS4 codes and PCI values
  - Handle missing PCI values for some HS4 codes gracefully

#### 1.3 ECI Calculation Functions (OEC Method)

- `calculate_eci_using_oec_method(trade_df: pd.DataFrame, pci_df: pd.DataFrame, district_col: str, hs4_col: str, value_col: str) -> pd.DataFrame`
  - Implement OEC methodology for calculating district-level ECI:
    - **Step 1: Calculate Revealed Comparative Advantage (RCA)**
      - `RCA_ij = (X_ij / Σ_j X_ij) / (Σ_i X_ij / Σ_ij X_ij)`
      - Where X_ij = exports of HS4 product j by district i
      - Binary RCA: `RCA_binary_ij = 1 if RCA_ij >= 1.0, else 0`
    - **Step 2: Merge PCI values**
      - Join trade data with PCI DataFrame on HS4 codes
      - Handle missing PCI values (exclude products without PCI or use interpolation)
    - **Step 3: Calculate District Complexity (ECI) using OEC formula**
      - `ECI_i = Σ_j (PCI_j × RCA_binary_ij) / Σ_j RCA_binary_ij`
      - Where PCI_j is the Product Complexity Index from OEC for product j
      - Standardize ECI: `ECI_std = (ECI - mean(ECI)) / std(ECI)` for interpretability
  - Handle districts with insufficient trade data (minimum threshold of products)
  - Handle missing HS4 codes or invalid product codes
  - Return DataFrame with district identifiers and ECI values
- `validate_hs4_codes(hs4_codes: pd.Series) -> pd.Series`
  - Validate HS4 codes are 4-digit (pad with zeros if needed)
  - Remove invalid codes
  - Return cleaned series

#### 1.4 District Matching

- `match_eci_to_districts(eci_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame`
  - Match ECI values to districts in merged dataset
  - Use same fuzzy matching logic from `etl_pipeline.py` if needed
  - Handle boundary mismatches (2011 vs 2021 districts)

### 2. Configuration Extension (`config.py`)

Add to existing configuration:

```python
DEFAULT_DATA_PATHS["eci_source"] = "data/eci_source"
DEFAULT_DATA_PATHS["oec_pci"] = "data/oec_pci.csv"  # Optional: local PCI data file
DEFAULT_CONFIG["eci_calculation_params"] = {
    "min_products": 5,  # Minimum HS4 products for ECI calculation
    "rca_threshold": 1.0,  # RCA threshold for specialization (binary RCA)
    "min_export_value": 1000,  # Minimum export value (USD/INR) to include product
    "use_oec_api": False,  # Use OEC API to fetch PCI (requires API token)
    "oec_api_token": None,  # OEC API token (if using API)
    "cache_pci": True,  # Cache PCI data locally after fetching
}
```

### 3. Visualization Module (`visualization.py`)

#### 3.1 Correlation Analysis

- `calculate_correlations(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame`
  - Calculate Pearson and Spearman correlation coefficients
  - Return correlation matrix
  - Handle missing values appropriately
- `plot_correlation_matrix(corr_df: pd.DataFrame, output_path: str) -> None`
  - Generate heatmap using `seaborn.heatmap()` or `matplotlib`
  - Include correlation values and significance indicators
  - Save to file

#### 3.2 Scatter Plot Functions

- `plot_eci_vs_gdp(df: pd.DataFrame, output_path: str) -> None`
  - Scatter plot: ECI (x-axis) vs GDP per capita proxy (y-axis)
  - Add regression line and R² value
  - Color-code by state or region
- `plot_eci_vs_mpi(df: pd.DataFrame, output_path: str) -> None`
  - Scatter plot: ECI vs MPI Score
  - Inverse relationship expected (higher ECI → lower MPI)
- `plot_eci_vs_luminosity(df: pd.DataFrame, output_path: str) -> None`
  - Scatter plot: ECI vs VIIRS mean luminosity
  - Add regression line
- `plot_pairwise_scatters(df: pd.DataFrame, output_path: str) -> None`
  - Create pairwise scatter plot matrix for all variables
  - Use `seaborn.pairplot()` or `pandas.plotting.scatter_matrix()`

#### 3.3 Spatial Visualization

- `create_eci_map(df: pd.DataFrame, shapefile_path: str, output_path: str) -> None`
  - Choropleth map showing ECI distribution across districts
  - Use `geopandas` and `matplotlib` or `folium` for interactive maps
  - Color-code districts by ECI quartiles or continuous scale
- `create_correlation_map(df: pd.DataFrame, shapefile_path: str, variable: str, output_path: str) -> None`
  - Map showing spatial correlation patterns
  - Highlight districts with strong positive/negative correlations
- `create_multi_panel_maps(df: pd.DataFrame, shapefile_path: str, output_path: str) -> None`
  - Side-by-side maps: ECI, GDP, MPI, Luminosity
  - Facilitate visual comparison

#### 3.4 Statistical Summary

- `generate_summary_statistics(df: pd.DataFrame, output_path: str) -> None`
  - Descriptive statistics table
  - Correlation coefficients with p-values
  - Export to CSV/Markdown

### 4. Analysis Pipeline (`analysis_pipeline.py`)

#### 4.1 Main Orchestrator

- `run_full_analysis(config_dict: Optional[Dict] = None) -> pd.DataFrame`
  - Orchestrate complete workflow:
    1. Run existing ETL pipeline (`run_etl_pipeline()`)
    2. Load/calculate ECI data
    3. Merge ECI with existing dataset
    4. Generate all visualizations
    5. Export correlation analysis
  - Return final merged DataFrame with ECI

#### 4.2 Integration Functions

- `merge_eci_with_dataset(merged_df: pd.DataFrame, eci_df: pd.DataFrame) -> pd.DataFrame`
  - Merge ECI values into existing merged dataset
  - Handle district name matching
  - Preserve all existing columns
- `generate_analysis_report(df: pd.DataFrame, output_dir: str) -> None`
  - Generate comprehensive analysis report
  - Include all visualizations and statistics
  - Create HTML or PDF report (optional)

### 5. ETL Pipeline Extension (`etl_pipeline.py`)

#### 5.1 Optional ECI Integration

- Add optional parameter to `run_etl_pipeline()`:
  - `include_eci: bool = False`
  - If True, load and merge ECI data automatically

#### 5.2 Export Function Update

- Update `export_merged_data()` to include ECI column if present
- Add ECI to final output columns list

### 6. Dependencies Update (`requirements.txt`)

Add visualization, statistical analysis, and API access packages:

```
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0  # For statistical tests
folium>=0.14.0  # Optional: for interactive maps
plotly>=5.14.0  # Optional: for interactive visualizations
requests>=2.31.0  # For OEC API access
```

## Data Flow

```
[Existing ETL Pipeline]
    ↓
[Merged Dataset: GDP, MPI, Luminosity]
    ↓
[ECI Calculator Module]
    ↓
[ECI Values by District]
    ↓
[Merge ECI with Merged Dataset]
    ↓
[Final Dataset: GDP, MPI, Luminosity, ECI]
    ↓
[Visualization Module]
    ↓
[Correlation Analysis + Visualizations]
```

## Key Technical Considerations

### ECI Calculation Methodology (OEC Method)

**Data Requirements:**
- District-level international export data
- HS4 (4-digit Harmonized System) product codes
- Export values (USD or INR)
- OEC Product Complexity Index (PCI) values for HS4 products

**Calculation Steps (OEC Method):**
1. Load OEC PCI data (from API or local file)
2. Calculate RCA from district-level trade data
3. Calculate ECI: `ECI_i = Σ_j (PCI_j × RCA_binary_ij) / Σ_j RCA_binary_ij`
4. Standardize ECI for interpretability

**Advantages:** Uses established OEC PCI values; no need to calculate PRODY from scratch.

### Visualization Best Practices

1. **Scatter Plots**: Include regression lines, R², p-values; color-code by state
2. **Correlation Matrix**: Diverging color scale; significance indicators
3. **Spatial Maps**: ColorBrewer schemes; legend and scale; handle missing data

### Handling Missing ECI Data

- Flag missing values clearly
- Option to impute using state/region means
- Exclude from correlation analysis or handle separately

## Output Files

1. `india_district_macro_with_eci.csv` - Final merged dataset including ECI
2. `correlation_matrix.png` - Correlation heatmap
3. `eci_vs_gdp_scatter.png` - ECI vs GDP scatter plot
4. `eci_vs_mpi_scatter.png` - ECI vs MPI scatter plot
5. `eci_vs_luminosity_scatter.png` - ECI vs luminosity scatter plot
6. `pairwise_scatters.png` - Pairwise scatter matrix
7. `eci_map.png` - Spatial distribution of ECI
8. `multi_panel_maps.png` - Side-by-side comparison maps
9. `correlation_analysis_report.csv` - Statistical summary
10. `analysis_report.html` (optional) - Comprehensive HTML report

## Usage Example

```python
from analysis_pipeline import run_full_analysis
import config

# Run complete analysis
final_df = run_full_analysis()

# Or run step-by-step
from etl_pipeline import run_etl_pipeline
from eci_calculator import load_hs4_trade_data, load_oec_pci_data, calculate_eci_using_oec_method, match_eci_to_districts
from visualization import plot_correlation_matrix, create_eci_map

# Step 1: Run ETL
merged_df = run_etl_pipeline()

# Step 2: Load OEC PCI data and calculate ECI
trade_df = load_hs4_trade_data("data/eci_source/hs4_trade_data.csv")
pci_df = load_oec_pci_data(path="data/oec_pci.csv", use_api=False)  # Or use_api=True with token
eci_df = calculate_eci_using_oec_method(
    trade_df, 
    pci_df,
    district_col="district_name",
    hs4_col="hs4_code",
    value_col="export_value"
)

# Step 3: Merge ECI
final_df = match_eci_to_districts(eci_df, merged_df)

# Step 4: Visualize
plot_correlation_matrix(final_df[["ECI", "SHRUG_Cons_PC", "MPI_Score", "VIIRS_Mean"]])
create_eci_map(final_df, "data/shapefiles/india_districts_2021.shp", "eci_map.png")
```

## Implementation Todos

- [ ] Extend config.py with ECI-related paths and parameters
- [ ] Create eci_calculator.py with OEC PCI + HS4 trade ECI calculation
- [ ] Create visualization.py with correlation analysis, scatter plots, spatial mapping
- [ ] Create analysis_pipeline.py to orchestrate ETL + ECI + visualization workflow
- [ ] Update requirements.txt with visualization dependencies
- [ ] Extend etl_pipeline.py to optionally include ECI merging
- [ ] Test end-to-end workflow with sample data

## Notes

- **HS4 Trade Data**: District-level exports by HS4 code. Source: customs/export databases.
- **OEC PCI Data**: From OEC API or download from https://oec.world/en/rankings/pci/hs4/hs92
- **OEC API**: https://oec.world/en/resources/api - Free tier for basic use.
- **HS4 Codes**: 4-digit format (e.g., "0101", "8471"). Match HS revision to OEC data.
- **Missing Districts**: NaN ECI for districts with < min_products; exclude or impute.

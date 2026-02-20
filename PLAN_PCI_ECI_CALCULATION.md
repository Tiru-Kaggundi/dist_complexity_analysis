# Plan: Add PCI and ECI Columns to Base Dataset

## Overview

Add Product Complexity Index (PCI) and Economic Complexity Index (ECI) columns to the base trade dataset (`dists_2025_full.csv`) using OEC methodology.

## Current Dataset Structure

**Location:** `dgcis_stateoforigin1771519863052/dists_2025_full.csv`

**Current Columns:**
- `State` - State name
- `District` - District name  
- `HS Code` - 4-digit HS code (e.g., "0303", "0802")
- `Commodity Description` - Product description
- `January, 25 To December, 25 Value(INR)` - Annual export value in INR
- `January, 25 To December, 25 Value(US $)` - Annual export value in USD

**Dataset Stats:**
- ~79,291 rows (district-product combinations)
- Multiple products per district
- Multiple districts per state

## Target Dataset Structure

**New Location:** `dists_2025_full.csv` (main folder)

**New Columns to Add:**
1. `MPI_HCR` - Multidimensional Poverty Index Headcount Ratio (future)
2. `VIIRS_Mean` - Mean nighttime light radiance (future)
3. `PCI` - Product Complexity Index (from OEC) - **START HERE**
4. `ECI` - Economic Complexity Index (calculated) - **START HERE**

## Implementation Plan: PCI and ECI Calculation

### Step 1: Data Preparation

#### 1.1 Load Base Dataset
- Load `dists_2025_full.csv`
- Validate HS Code format (ensure 4-digit, pad if needed)
- **Filter:** Remove products with export value < $1000 USD
- Use USD values for calculations (more standardized)
- Aggregate exports by district and HS4 code (sum if multiple entries)

#### 1.2 Load OEC PCI Data
- **Method: Download from OEC API** (using API token)
  - API endpoint: `https://api-v2.oec.world/tesseract/data.jsonrecords`
  - Cube: `hs92_4d` (or appropriate HS revision)
  - Drilldown: `HS4`
  - Measure: `PCI`
  - Authentication: Use token from `apikey.py` file
  - Cache locally as `data/oec_pci.csv` to avoid repeated API calls
  - Reference: https://colab.research.google.com/drive/1u_af79O7R55SiaqBgKvPAqbwzZFe0vT3

- **API Token Storage:**
  - Store token in `apikey.py` file (gitignored)
  - Format: `OEC_API_TOKEN = "your_token_here"`
  - Never commit this file to git

- **Expected PCI Data Format:**
  ```
  HS4,PCI
  0101,0.234
  0102,-0.456
  ...
  ```

### Step 2: PCI Column Addition

#### 2.1 Merge PCI Values
- Join base dataset with PCI data on HS Code
- Handle missing PCI values:
  - Option 1: Exclude products without PCI (set PCI to NaN)
  - Option 2: Use mean PCI for missing products
  - Option 3: Interpolate based on similar products

#### 2.2 Add PCI Column
- Add `PCI` column to base dataset
- Each row gets the PCI value for its HS4 product code

### Step 3: ECI Calculation (OEC Method)

#### 3.1 Aggregate District Exports
- Group by `State` and `District`
- For each district, calculate:
  - Total exports: `Σ_j X_ij` (sum of all product exports)
  - Product-level exports: `X_ij` (exports of product j by district i)

#### 3.2 Calculate Revealed Comparative Advantage (RCA)

For each district-product pair:
```
RCA_ij = (X_ij / Σ_j X_ij) / (Σ_i X_ij / Σ_ij X_ij)
```

Where:
- `X_ij` = exports of product j by district i
- `Σ_j X_ij` = total exports of district i (across all products)
- `Σ_i X_ij` = total exports of product j (across all districts)
- `Σ_ij X_ij` = total exports (all districts, all products)

**Binary RCA:**
```
RCA_binary_ij = 1 if RCA_ij >= 1.0, else 0
```

#### 3.3 Calculate District ECI

For each district i:
```
ECI_i = Σ_j (PCI_j × RCA_binary_ij) / Σ_j RCA_binary_ij
```

Where:
- `PCI_j` = Product Complexity Index for product j (from OEC)
- `RCA_binary_ij` = Binary RCA (1 if district has comparative advantage, 0 otherwise)

**Standardization:**
```
ECI_std = (ECI - mean(ECI)) / std(ECI)
```

#### 3.4 Handle Edge Cases
- Districts with < 5 products: Set ECI to NaN or use state average
- Products without PCI: Exclude from ECI calculation
- Districts with no RCA >= 1: Set ECI to NaN

### Step 4: Add ECI Column to Dataset

#### 4.1 Create District-Level ECI DataFrame
- Calculate ECI for each district
- Result: DataFrame with `State`, `District`, `ECI`

#### 4.2 Merge ECI Back to Base Dataset
- Join ECI values to base dataset on `State` and `District`
- Each row gets the ECI value for its district

### Step 5: Output Dataset

#### 5.1 Final Dataset Structure
```
State, District, HS Code, Commodity Description, 
January, 25 To December, 25 Value(INR), 
January, 25 To December, 25 Value(US $),
PCI, ECI
```

#### 5.2 Save to Main Folder
- Save as `dists_2025_full.csv` in main folder (`/Users/tiru/Documents/Research/OEC_stuff/`)
- Keep original in `dgcis_stateoforigin1771519863052/` as backup

## Implementation Modules

### Module 1: `pci_eci_calculator.py`

**Functions:**

1. `load_base_dataset(path: str, min_export_usd: float = 1000) -> pd.DataFrame`
   - Load and validate base trade dataset
   - Filter products with export value >= min_export_usd (default: $1000)
   - Clean HS codes (ensure 4-digit format)
   - Aggregate by district and HS4 code

2. `load_oec_pci_data(path: Optional[str] = None, use_api: bool = True, api_token: Optional[str] = None) -> pd.DataFrame`
   - Load PCI data from OEC API (default) or local file
   - Use API token from `apikey.py` if not provided
   - Cache results locally to avoid repeated API calls
   - Return DataFrame with HS4 and PCI columns

3. `add_pci_column(base_df: pd.DataFrame, pci_df: pd.DataFrame) -> pd.DataFrame`
   - Merge PCI values into base dataset
   - Handle missing PCI values

4. `calculate_rca(trade_df: pd.DataFrame, district_col: str, hs4_col: str, value_col: str) -> pd.DataFrame`
   - Calculate RCA for each district-product pair
   - Return DataFrame with RCA and RCA_binary columns

5. `calculate_eci(trade_df: pd.DataFrame, pci_df: pd.DataFrame, district_col: str, hs4_col: str, value_col: str) -> pd.DataFrame`
   - Calculate ECI for each district using OEC formula
   - Return DataFrame with State, District, ECI columns

6. `add_eci_column(base_df: pd.DataFrame, eci_df: pd.DataFrame) -> pd.DataFrame`
   - Merge ECI values into base dataset

7. `process_and_enrich_dataset(input_path: str, output_path: str, pci_path: Optional[str] = None) -> pd.DataFrame`
   - Main function: orchestrate all steps
   - Load base dataset, add PCI, calculate ECI, save enriched dataset

## Data Flow

```
[Base Dataset: dists_2025_full.csv]
    ↓
[Load & Validate]
    ↓
[Load OEC PCI Data]
    ↓
[Add PCI Column] → [Base Dataset + PCI]
    ↓
[Calculate RCA] → [District-Product RCA Matrix]
    ↓
[Calculate ECI] → [District ECI Values]
    ↓
[Add ECI Column] → [Base Dataset + PCI + ECI]
    ↓
[Save Enriched Dataset]
```

## Configuration

Add to `config.py`:

```python
DEFAULT_DATA_PATHS["base_trade_data"] = "dists_2025_full.csv"
DEFAULT_DATA_PATHS["oec_pci"] = "data/oec_pci.csv"

DEFAULT_CONFIG["pci_eci_params"] = {
    "min_products_for_eci": 5,  # Minimum products needed for ECI calculation
    "min_export_value_usd": 1000,  # Minimum export value in USD to include product
    "rca_threshold": 1.0,  # RCA threshold for binary RCA
    "use_usd": True,  # Use USD values instead of INR
    "standardize_eci": True,  # Standardize ECI (mean=0, std=1)
    "handle_missing_pci": "exclude",  # Options: "exclude", "mean", "interpolate"
    "use_oec_api": True,  # Use OEC API to fetch PCI data
    "cache_pci": True,  # Cache PCI data locally after fetching
}
```

## Expected Output

**Dataset:** `dists_2025_full.csv` (main folder)

**New Columns:**
- `PCI`: Product Complexity Index (from OEC) - available for each product
- `ECI`: Economic Complexity Index (calculated) - same value for all products in a district

**Statistics:**
- PCI: Range typically -2 to +2 (OEC standardized)
- ECI: Range typically -2 to +2 (after standardization)
- Districts with ECI: ~600-700 districts (depending on data quality)

## Testing Strategy

1. **Unit Tests:**
   - Test RCA calculation with known values
   - Test ECI calculation with sample data
   - Test PCI merging logic

2. **Integration Tests:**
   - Test full pipeline with sample dataset
   - Verify PCI and ECI values are reasonable
   - Check for missing values handling

3. **Validation:**
   - Compare ECI values with known benchmarks (if available)
   - Check that districts with more diverse exports have higher ECI
   - Verify PCI values match OEC rankings

## Next Steps (Future)

1. Add MPI column (from downloaded MPI data)
2. Add VIIRS luminosity column (from SHRUG data)
3. Generate visualizations and correlation analysis

## Dependencies

- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical calculations
- `requests>=2.31.0` - OEC API access (if using API)

## API Token Setup

### Secure Token Storage

1. **File Created:** `apikey.py` (gitignored - will not be committed)
2. **Format:**
   ```python
   OEC_API_TOKEN = "YOUR_TOKEN"
   ```
3. **Usage in Code:**
   ```python
   try:
       from apikey import OEC_API_TOKEN
   except ImportError:
       OEC_API_TOKEN = None
       print("Warning: apikey.py not found. Please create it with your OEC API token.")
   ```

4. **Security:**
   - `apikey.py` is added to `.gitignore`
   - Never commit this file to git
   - Keep it local only

### Getting OEC API Token

1. Visit: https://oec.world/en/resources/api
2. Sign up for API access (free tier available)
3. Get your API token
4. Insert token in `apikey.py` file

## Notes

- **HS Code Format:** Ensure all HS codes are 4-digit strings (pad with zeros if needed)
- **PCI Data Year:** Use PCI data that matches the HS revision of trade data (likely HS92)
- **Minimum Export Value:** Products with < $1000 USD exports are filtered out before calculation
- **Missing Values:** Districts/products without sufficient data will have NaN PCI/ECI
- **Performance:** For 79K rows, calculation should complete in < 1 minute
- **API Rate Limits:** OEC API may have rate limits; caching PCI data locally avoids repeated calls

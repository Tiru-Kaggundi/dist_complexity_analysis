"""Simple test to verify MPI data format."""

import pandas as pd
from pathlib import Path

# Load the downloaded MPI data
mpi_path = Path("data/niti_mpi/mpi_district_data.csv")

if not mpi_path.exists():
    print(f"Error: {mpi_path} not found. Run download_mpi_data.py first.")
    sys.exit(1)

df = pd.read_csv(mpi_path)

print("="*60)
print("MPI Data Verification")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nData types:")
print(df.dtypes)
print(f"\nSummary statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nUnique states: {df['State'].nunique()}")
print(f"\nTotal districts: {len(df)}")
print(f"\nMPI HCR range: {df['MPI_HCR'].min():.2f}% - {df['MPI_HCR'].max():.2f}%")

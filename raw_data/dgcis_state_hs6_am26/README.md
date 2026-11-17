# DGCI&S state-of-origin exports, FY26 (AM26)

Source: DGCI&S "Export Summary Report" (state of origin), downloaded 2026-11-17.
Period: April 2025 to March 2026. Unmodified `.xls` exports, split into 7 parts
(the portal caps each download at 10,000 rows); only the file names were changed.

Layout of each file (sheet `NewData`):
- Row 1: report title ("Export Summery Report Period From Apr-2025 To Mar-2026")
- Row 2: header
- Columns: `Hs Code` (6-digit ITC-HS), `State`, `Commodity Description`,
  `March, 26 Value(INR)`, `March, 26 Value(US $)`,
  `April, 25 To March, 26 Value(INR)`, `April, 25 To March, 26 Value(US $)`

Checks on the combined data:
- 67,188 rows, no duplicate State x HS6 pairs
- 37 states/UTs (includes `UNSPECIFIED`, about US$4.3 bn)
- 5,216 distinct HS6 codes (1,217 HS4)
- FY26 total about US$441.5 bn
- Level: State x HS6 only (no district breakdown, unlike the AM25 data)

Read with `pd.read_excel(path, header=1, dtype={"Hs Code": str})`, then
`zfill(6)` the HS code, since leading zeros can be lost.

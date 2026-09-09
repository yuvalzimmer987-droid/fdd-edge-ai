import pandas as pd

FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant.csv"

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

df = pd.read_csv(FILE, nrows=5)

all_columns = list(df.columns)
print(f"Total columns in file: {len(all_columns)}\n")

found   = [s for s in SELECTED_SENSORS if s in all_columns]
missing = [s for s in SELECTED_SENSORS if s not in all_columns]

print(f"Found {len(found)} of {len(SELECTED_SENSORS)}:")
for s in found:
    print("  OK  ", s)

if missing:
    print(f"\nNot found ({len(missing)}):")
    for s in missing:
        print("  XX  ", s)
    print("\nAll column names in file:")
    for c in all_columns:
        print("   ", c)
else:
    print("\nAll sensors found. Ready to continue.")

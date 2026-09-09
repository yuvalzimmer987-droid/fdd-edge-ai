import pandas as pd

FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant.csv"

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

# Load only the selected sensors (faster than loading all 78 columns)
df = pd.read_csv(FILE, usecols=SELECTED_SENSORS)

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per sensor:")
print(df.isnull().sum())

print("\nBasic statistics (min / max / mean per sensor):")
print(df.describe())

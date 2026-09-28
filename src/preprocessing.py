import numpy as np
import pandas as pd

# Preprocessing shared by prepare_data.py (fault-free file) and
# evaluate_faults.py (fault files), so both get exactly the same treatment.

DATA_DIR = "data/LBNL_FDD_Data_Sets_Chiller_Plant"

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

ROWS_PER_DAY = 1440          # 1-minute data
RUNNING_THRESHOLD = 5.0      # kW - CHL_POW_1 above this = chiller is cooling
WARMUP_DAYS = 1              # skip the first day (simulation start-up)
STARTUP_MINUTES = 30         # skip the first minutes after each chiller start


def load_plant_file(path):
    """Load one plant CSV with the selected sensors, in a fixed column order."""
    df = pd.read_csv(path, usecols=SELECTED_SENSORS)
    # In the source files OA_TEMP and OA_TEMP_WB are swapped: the column named
    # OA_TEMP_WB is higher than OA_TEMP in 97% of rows (and never lower), but a
    # wet-bulb can never exceed the dry-bulb. Swap the names back.
    # (check_wet_bulb.py shows the evidence.)
    df = df.rename(columns={"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP"})
    # usecols keeps the file's column order - force our own, fixed order
    return df[SELECTED_SENSORS]


def day_split(n_rows):
    """Assign each whole day to train/val/test (70/15/15) at random.

    The file is one year of 1-minute data. A plain chronological split put
    Jan-Sep in train and only Nov-Dec (winter) in test, so each split saw a
    different season. With whole days drawn at random, every split covers the
    whole year, and rows inside a day stay together.
    Returns the split name ("train"/"val"/"test") of every row.
    """
    day = np.arange(n_rows) // ROWS_PER_DAY
    rng = np.random.default_rng(42)   # fixed seed -> same split every run
    split_of_day = rng.choice(["train", "val", "test"], size=day.max() + 1,
                              p=[0.70, 0.15, 0.15])
    return split_of_day[day]


def steady_state_mask(df):
    """Rows where the chiller is cooling in steady state.

    Returns (keep, running, startup, warmup) boolean arrays.

    - running: CHL_POW_1 > RUNNING_THRESHOLD. When not cooling, the chiller is
      at ~0 kW (plant off) or ~1.94 kW (standby, about half the year); those
      rows are trivial for the autoencoder and would dominate training.
    - warmup: day 0. The chiller runs on a cold January day right at the
      simulation start, a state that does not occur anywhere else.
    - startup: the first STARTUP_MINUTES after each chiller start, when the
      sensors have not settled yet (38% of the worst rows vs 10% of all rows).
    Steady-state filtering is standard in chiller FDD.
    """
    running = (df["CHL_POW_1"] > RUNNING_THRESHOLD).to_numpy()
    run_id = np.cumsum(~running)
    minutes_running = pd.Series(running.astype(int)).groupby(run_id).cumsum().to_numpy()
    day = np.arange(len(df)) // ROWS_PER_DAY

    startup = running & (minutes_running <= STARTUP_MINUTES)
    warmup = running & (day < WARMUP_DAYS)
    keep = running & ~startup & ~warmup
    return keep, running, startup, warmup

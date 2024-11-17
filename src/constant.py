import os
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MAIN_DIR / "data"
DATABASE_DIR = MAIN_DIR / "database"
LOG_DIR = MAIN_DIR / "log"
SOLVER_DIR = MAIN_DIR / "solver"
TEMP_DIR = MAIN_DIR / "temp"

LKH_PATH = SOLVER_DIR / "LKH"
CONCORDE_PATH = SOLVER_DIR / "concorde"
UBC_TSP_FEATURE_PATH = SOLVER_DIR / "TSP-feature"
MAX_WORKERS = 10

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    MAX_WORKERS = 5
    home = Path.home()
    # r_home = home / "AppData" / "Local" / "miniconda3" / "envs" / "SMAC" / "Lib" / "R"

r_home = home / "miniconda3" / "envs" / "SMAC" / "Lib" / "R"
os.environ["R_HOME"] = str(r_home)

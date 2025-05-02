import datetime as dt
import os
import random
from pathlib import Path

import numpy as np

# seed
SEED = int(os.environ.get("SEED", "0").strip())
np.random.seed(SEED)
random.seed(SEED)

# dir
MAIN_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = MAIN_DIR / "data"
DATABASE_DIR = MAIN_DIR / "database"
LOG_DIR = MAIN_DIR / "log"
SOLVER_DIR = MAIN_DIR / "solver"
TEMP_DIR = MAIN_DIR / "temp"

# environment
POLICY = os.environ.get("POLICY", "baseline").strip()
PARG = os.environ.get("PARG", "").strip()
N_TRAIN = int(os.environ.get("N_TRAIN", "25").strip())
JOB_NAME = os.environ.get("SLURM_JOB_NAME", "test")
JOB_ID = os.environ.get("SLURM_JOB_ID", dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
MAX_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 1
R_HOME = Path.home() / "miniconda3" / "envs" / "SMAC" / "lib" / "R"

# TSP
LKH_PATH = SOLVER_DIR / "LKH"
CONCORDE_PATH = SOLVER_DIR / "concorde"

IS_WINDOWS = os.name == "nt"
# if IS_WINDOWS:
# R_HOME = home / "AppData" / "Local" / "miniconda3" / "envs" / "SMAC" / "Lib" / "R"

os.environ["R_HOME"] = str(R_HOME)

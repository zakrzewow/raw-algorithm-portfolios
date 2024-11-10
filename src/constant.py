import os
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = MAIN_DIR / "log"
SOLVER_DIR = MAIN_DIR / "solver"
TEMP_DIR = MAIN_DIR / "temp"

LKH_PATH = SOLVER_DIR / "LKH"
CONCORDE_PATH = SOLVER_DIR / "concorde"

MAX_WORKERS = 8

if os.name == "nt":
    MAX_WORKERS = 5

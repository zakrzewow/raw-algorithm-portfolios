import os
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = MAIN_DIR / "temp"
SOLVER_DIR = MAIN_DIR / "solver"

LKH_PATH = SOLVER_DIR / "LKH"
CONCORDE_PATH = SOLVER_DIR / "concorde"

MAX_WORKERS = 8

if os.name == "nt":
    MAX_WORKERS = 4

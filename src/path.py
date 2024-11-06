from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = MAIN_DIR / "temp"
SOLVER_DIR = MAIN_DIR / "solver"

LKH_PATH = SOLVER_DIR / "LKH"
CONCORDE_PATH = SOLVER_DIR / "concorde"

# import os
# if os.name == "nt":
#     LKH_PATH += ".exe"

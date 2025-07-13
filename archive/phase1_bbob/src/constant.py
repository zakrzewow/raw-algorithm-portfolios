from pathlib import Path
from types import SimpleNamespace

HO = SimpleNamespace(
    # N_TRIALS=3,
    N_TRIALS=500,
    N=3,
    INSTANCE_NUMBER=10,
    SOLVER_NUMBER=90,
    RANDOM_STATE=0,
)
# RANDOM_STATE_LIST = list(range(1, 2))
RANDOM_STATE_LIST = list(range(1, 21))
# SOLVER_NUMBER_LIST = [5, 10, 15, 20, 30, 50]
SOLVER_NUMBER_LIST = [5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]


MAIN_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = MAIN_DIR / "processed"
RESULTS_DIR = MAIN_DIR / "results"
RESULTS_BASE_DIR = RESULTS_DIR / "base"
RESULTS_CONST_CUT_OFF_DIR = RESULTS_DIR / "const_cut_off"
RESULTS_PERMUTATION_DIR = RESULTS_DIR / "permutation"
RESULTS_0_10_DIR = RESULTS_DIR / "0_10"

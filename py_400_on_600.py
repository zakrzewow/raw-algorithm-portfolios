import os

import joblib

from src.constant import DATA_DIR, MAIN_DIR
from src.instance.TSP_Instance import TSP_from_index_file

if __name__ == "__main__":
    test_instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TEST_600" / "index.json"
    )
    prefix = os.environ["PREFIX"].strip()

    paths = list((MAIN_DIR / "portfolios" / "400").glob(f"{prefix}-*.pkl"))
    for path in paths:
        run_id = path.stem.split("-")[-1]
        portfolio = joblib.load(path)

        for i in range(100):
            portfolio.evaluate(
                test_instances,
                prefix=f"{run_id};test{i}",
                calculate_features=False,
                cache=False,
            )

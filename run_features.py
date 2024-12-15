import concurrent.futures
from typing import Dict

import pandas as pd

from src.constant import DATA_DIR, MAX_WORKERS
from src.database import db_connect, db_init, db_insert_instance, db_instance_exists
from src.instance import Instance
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.log import logger
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver


def _calculate_instance_features(instance: Instance) -> Dict:
    return instance.calculate_features()


if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=0,
        seed=0,
    )

    solver_class = TSP_LKH_Solver
    instance_class = TSP_Instance
    db_init(solver_class, instance_class, False)
    conn = db_connect()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)

    times = []
    futures = []
    for instance in test_instances:
        if not db_instance_exists(conn, instance):
            future = executor.submit(_calculate_instance_features, instance)
            futures.append((instance, future))
    for instance, future in futures:
        try:
            time, features = future.result()
            logger.debug(f"{instance.__hash__()} features calculated (time={time:.2f})")
            times.append(time)
            db_insert_instance(conn, instance, features)
        except Exception:
            pass
    logger.info(pd.Series(times).describe())

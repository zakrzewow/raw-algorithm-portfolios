import os

os.environ["SEED"] = "0"

from concurrent.futures import ProcessPoolExecutor

from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_train_test_from_index_file
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=5,
    )
    solver = TSP_LKH_Solver()
    executor = ProcessPoolExecutor(max_workers=5)

    futures = []
    for instance in train_instances:
        futures.append(solver.solve(instance, executor))

    for future in futures:
        future.result()

    for instance in train_instances:
        solver.solve(instance, executor, cache=True)

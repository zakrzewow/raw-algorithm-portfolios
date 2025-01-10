from concurrent.futures import ProcessPoolExecutor

from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_from_index_file

if __name__ == "__main__":
    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json"
    )
    with ProcessPoolExecutor(max_workers=9) as executor:
        for instance in instances:
            instance.calculate_features(executor)

    instances = TSP_from_index_file(filepath=DATA_DIR / "TSP" / "MY" / "index.json")
    with ProcessPoolExecutor(max_workers=9) as executor:
        for instance in instances:
            instance.calculate_features(executor)

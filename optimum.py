import json
from concurrent.futures import ProcessPoolExecutor

from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_from_index_file


def calculate_optimum(instance):
    return instance._get_optimum_with_concorde()


if __name__ == "__main__":
    dict_ = {}
    instances = TSP_from_index_file(
        filepath=DATA_DIR / "TSP" / "TRAIN_50" / "index.json"
    )

    futures = []
    with ProcessPoolExecutor(max_workers=9) as executor:
        for instance in instances:
            futures.append((instance, executor.submit(calculate_optimum, instance)))

    for future in futures:
        instance, result = future
        optimum, _ = result.result()
        key = instance._get_short_filepath()
        dict_[key] = optimum

    with open(DATA_DIR / "TSP" / "TRAIN_50" / "index.json", "w") as f:
        json.dump(dict_, f, indent=4)

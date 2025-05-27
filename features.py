import json

from src.constant import DATA_DIR
from src.instance.SAT_Instance import SAT_from_index_file

if __name__ == "__main__":
    instances = SAT_from_index_file(
        filepath=DATA_DIR / "SAT" / "index.json",
        max_cost=100.0,
        max_time=10.0,
    )
    features = {}
    train_instances = instances[:15] + instances[80:95]
    for instance in train_instances:
        print(f"Calculating features for instance {instance=}")
        result_with_time = instance.calculate_features()
        features[instance.id()] = {
            "result": result_with_time.result,
            "time": result_with_time.time,
        }
    with open(DATA_DIR / "SAT" / "features.json", "w") as f:
        json.dump(features, f, indent=4)

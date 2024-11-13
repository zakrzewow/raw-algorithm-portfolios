from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_InstanceSet

train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
    filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
    train_size=5,
    seed=0,
)

features = train_instances[0].get_features()
print(features)
print(len(features))

from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_InstanceSet

train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
    filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
    train_size=10,
    seed=0,
)

for instance in train_instances:
    print(instance.calculate_features())

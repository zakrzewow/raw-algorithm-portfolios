from src.constant import DATA_DIR
from src.instance.TSP_Instance import TSP_InstanceSet

train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
    filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
    train_size=5,
    seed=0,
)

instance = train_instances[0]
optimum, time = instance._get_optimum_with_concorde()
print(f"optimum = {optimum},  time = {time}, original optimum = {instance.optimum}")

instance2, time = instance.mutate()
print(f"instance = {instance2.filepath} optimum = {instance2.optimum},  time = {time}")

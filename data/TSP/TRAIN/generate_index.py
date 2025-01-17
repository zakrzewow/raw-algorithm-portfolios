import json

dict_ = {}

for generator in [
    "cluster_netgen",
    "compression",
    "expansion",
    "explosion",
    "grid",
]:
    for i in range(200):
        dict_[f"TSP/TRAIN/{generator}/{i:>03}.tsp"] = 0

with open("index.json", "w") as f:
    json.dump(dict_, f, indent=4)

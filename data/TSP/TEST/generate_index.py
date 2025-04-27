import json

dict_ = {}

for generator in [
    "cluster",
    "cluster_netgen",
    "compression",
    "expansion",
    "explosion",
    "grid",
    "implosion",
    "linearprojection",
    "rotation",
    "uniform_portgen",
]:
    for i in range(200):
        dict_[f"TSP/TEST/{generator}/{i:>03}.tsp"] = 0

with open("index.json", "w") as f:
    json.dump(dict_, f, indent=4)

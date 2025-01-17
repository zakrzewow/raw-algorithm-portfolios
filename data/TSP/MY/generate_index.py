import json

dict_ = {}

for generator in [
    "cluster_netgen",
    "compression",
    "expansion",
    "explosion",
    "grid",
    "cluster",
    "implosion",
    "linear_projection",
    "rotation",
    "uniform_portgen",
]:
    for i in range(5):
        dict_[f"TSP/MY/{generator}/{i:>03}.tsp"] = 0

with open("index.json", "w") as f:
    json.dump(dict_, f, indent=4)

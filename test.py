import os

from rpy2.robjects.packages import importr

# os.environ["R_HOME"] = (
#     r"C:\Users\grzegorzzakrzewski\AppData\Local\miniconda3\envs\SMAC\Lib\R"
# )
# /home2/faculty/gzakrzewski/miniconda3/envs/SMAC/lib/R


tspmeta = importr("tspmeta")
x = tspmeta.read_tsplib_instance("data/TSP/CEPS_benchmark/cluster/00.tsp")
features = tspmeta.features(x)
features = {name: features[i][0] for i, name in enumerate(features.names)}

print(features)
# print(type(features[0][0]))
# print(type(features.names[0]))

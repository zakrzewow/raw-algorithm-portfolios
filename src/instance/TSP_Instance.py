import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from src.constant import CONCORDE_PATH, DATA_DIR, IS_WINDOWS, TEMP_DIR
from src.database import DB
from src.instance.Instance import Instance
from src.instance.InstanceList import InstanceList
from src.log import logger
from src.utils import ResultWithTime, Timer


class TSP_Instance(Instance):
    FEATURES = {
        "angle_min": 0.0,
        "angle_median": 0.0,
        "angle_mean": 0.0,
        "angle_max": 0.0,
        "angle_sd": 0.0,
        "angle_span": 0.0,
        "angle_coef_of_var": 0.0,
        "centroid_centroid_x": 0.0,
        "centroid_centroid_y": 0.0,
        "centroid_dist_min": 0.0,
        "centroid_dist_median": 0.0,
        "centroid_dist_mean": 0.0,
        "centroid_dist_max": 0.0,
        "centroid_dist_sd": 0.0,
        "centroid_dist_span": 0.0,
        "centroid_dist_coef_of_var": 0.0,
        "cluster_01pct_number_of_clusters": 0.0,
        "cluster_01pct_mean_distance_to_centroid": 0.0,
        "cluster_05pct_number_of_clusters": 0.0,
        "cluster_05pct_mean_distance_to_centroid": 0.0,
        "cluster_10pct_number_of_clusters": 0.0,
        "cluster_10pct_mean_distance_to_centroid": 0.0,
        "bounding_box_10_ratio_of_cities_outside_box": 0.0,
        "bounding_box_20_ratio_of_cities_outside_box": 0.0,
        "bounding_box_30_ratio_of_cities_outside_box": 0.0,
        "chull_area": 0.0,
        "chull_points_on_hull": 0.0,
        "distance_distances_shorter_mean_distance": 0.0,
        "distance_distinct_distances": 0.0,
        "distance_mode_frequency": 0.0,
        "distance_mode_quantity": 0.0,
        "distance_mode_mean": 0.0,
        "distance_mean_tour_length": 0.0,
        "distance_sum_of_lowest_edge_values": 0.0,
        "distance_min": 0.0,
        "distance_median": 0.0,
        "distance_mean": 0.0,
        "distance_max": 0.0,
        "distance_sd": 0.0,
        "distance_span": 0.0,
        "distance_coef_of_var": 0.0,
        "modes_number": 0.0,
        "mst_depth_min": 0.0,
        "mst_depth_median": 0.0,
        "mst_depth_mean": 0.0,
        "mst_depth_max": 0.0,
        "mst_depth_sd": 0.0,
        "mst_depth_span": 0.0,
        "mst_depth_coef_of_var": 0.0,
        "mst_dists_min": 0.0,
        "mst_dists_median": 0.0,
        "mst_dists_mean": 0.0,
        "mst_dists_max": 0.0,
        "mst_dists_sd": 0.0,
        "mst_dists_span": 0.0,
        "mst_dists_coef_of_var": 0.0,
        "mst_dists_sum": 0.0,
        "nnds_min": 0.0,
        "nnds_median": 0.0,
        "nnds_mean": 0.0,
        "nnds_max": 0.0,
        "nnds_sd": 0.0,
        "nnds_span": 0.0,
        "nnds_coef_of_var": 0.0,
    }

    def __init__(
        self,
        filepath: Path,
        optimum: float,
        max_cost: float = 0,
        max_time: float = 0,
    ):
        super().__init__()
        self.filepath = filepath
        self.optimum = optimum
        self.max_cost = max_cost
        self.max_time = max_time

    def __repr__(self):
        filepath = self._get_short_filepath()
        str_ = f"TSP_Instance(filepath={filepath})"
        return str_

    @classmethod
    def from_db(cls, id_: str) -> "TSP_Instance":
        dict_ = DB().select_id(DB.SCHEMA.INSTANCES, id_)
        filepath = DATA_DIR / dict_["filepath"]
        optimum = dict_["optimum"]
        max_cost = dict_["max_cost"]
        max_time = dict_["max_time"]
        instance = cls(filepath, optimum, max_cost, max_time)
        del dict_["filepath"]
        del dict_["optimum"]
        instance.features = dict_
        return instance

    def to_dict(self) -> dict:
        return {
            "filepath": self._get_short_filepath(),
            "optimum": self.optimum,
            "max_cost": self.max_cost,
            "max_time": self.max_time,
            **self.features,
        }

    def _get_short_filepath(self) -> str:
        path_parts = self.filepath.parts
        data_dir_parts = DATA_DIR.parts
        filepath = "/".join(path_parts[len(data_dir_parts) :])
        return filepath

    @classmethod
    def _calculate_features(cls, instance: "Instance") -> ResultWithTime:
        with Timer() as timer:
            tspmeta_features = instance._calculate_tspmeta_features()
            features = {**instance.FEATURES, **tspmeta_features}
        return ResultWithTime(features, timer.elapsed_time)

    def _calculate_tspmeta_features(self) -> dict:
        try:
            from rpy2.robjects.packages import importr

            tspmeta = importr("tspmeta")
            instance = tspmeta.read_tsplib_instance(str(self.filepath))
            features = tspmeta.features(instance)
            features = {name: features[i][0] for i, name in enumerate(features.names)}
            return features
        except Exception as e:
            logger.error(f"[{self}] error calculating tspmeta features: {e}")
            return {}

    def _read_file_to_df(self) -> pd.DataFrame:
        coordinates = []

        with open(self.filepath, "r") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith(
                    (
                        "NAME",
                        "TYPE",
                        "COMMENT",
                        "DIMENSION",
                        "EDGE_WEIGHT_TYPE",
                        "NODE_COORD_SECTION",
                        "LOWER",
                        "UPPER",
                    )
                ):
                    continue

                if line == "EOF":
                    break

                parts = line.split()
                if len(parts) == 3:
                    node, x, y = parts
                    coordinates.append((int(node), float(x), float(y)))

        df = pd.DataFrame(coordinates, columns=["node", "X", "Y"]).set_index("node")
        return df

    def _get_optimum_with_concorde(self) -> ResultWithTime:
        if IS_WINDOWS:
            return ResultWithTime(0.0, 10.0)

        try:
            temp_dir = tempfile.TemporaryDirectory(dir=TEMP_DIR)
            old_cwd = os.getcwd()
            os.chdir(temp_dir.name)

            with Timer() as timer:
                result = subprocess.run(
                    [CONCORDE_PATH, "-x", self.filepath],
                    capture_output=True,
                    text=True,
                )

                os.chdir(old_cwd)
                temp_dir.cleanup()

            optimum = None
            for line in result.stdout.splitlines():
                if "Optimal Solution:" in line:
                    optimum = float(line.split()[-1])
                    break
            if optimum is None:
                raise Exception("Optimum not found")
            return ResultWithTime(optimum, timer.elapsed_time)
        except Exception as e:
            logger.error(f"[{self}] error calculating optimum with concorde: {e}")
            return ResultWithTime(0.0, 10.0)

    def plot(self, fname: str = None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        df = self._read_file_to_df()
        plt.scatter(df["X"], df["Y"], s=3)
        plt.title(f"{self}", fontsize=10)
        ax.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
        plt.grid(color="black", alpha=0.05)
        if fname:
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def TSP_from_index_file(
    filepath: Path,
    max_cost: float = 0.0,
    max_time: float = 0.0,
) -> InstanceList:
    instances = InstanceList()

    with open(filepath) as f:
        index = json.load(f)

    for k, v in index.items():
        filepath = DATA_DIR / Path(k)
        instance = TSP_Instance(filepath, v, max_cost, max_time)
        instances.append(instance)
    return instances

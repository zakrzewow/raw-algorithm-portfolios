import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.constant import (
    CONCORDE_PATH,
    DATA_DIR,
    IS_WINDOWS,
    TEMP_DIR,
    UBC_TSP_FEATURE_PATH,
)
from src.instance import Instance, InstanceSet
from src.log import logger


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
        "num_nodes": 0.0,
        "cost_matrix_avg": 0.0,
        "cost_matrix_std": 0.0,
        "cost_matrix_skew": 0.0,
        "stdTime": 0.0,
        "mst_length": 0.0,
        "mst_length_avg": 0.0,
        "mst_length_std": 0.0,
        "mst_length_skew": 0.0,
        "mst_degree_avg": 0.0,
        "mst_degree_std": 0.0,
        "mst_degree_skew": 0.0,
        "mstTime": 0.0,
        "cluster_distance_avg": 0.0,
        "cluster_distance_std": 0.0,
        "cluster_distance_skew": 0.0,
        "clusterTime": 0.0,
        "tour_const_heu_avg": 0.0,
        "tour_const_std": 0.0,
        "tour_const_skew": 0.0,
        "ls_impov_per_step_avg": 0.0,
        "ls_impov_per_step_std": 0.0,
        "ls_impov_per_step_skew": 0.0,
        "ls_steps_2lm_avg": 0.0,
        "ls_steps_2lm_std": 0.0,
        "ls_steps_2lm_skew": 0.0,
        "ls_maxdist_avg": 0.0,
        "ls_maxdist_std": 0.0,
        "ls_maxdist_skew": 0.0,
        "ls_bestsol_avg": 0.0,
        "ls_bestsol_std": 0.0,
        "ls_bestsol_skew": 0.0,
        "ls_backbone_avg": 0.0,
        "ls_backbone_std": 0.0,
        "ls_backbone_skew": 0.0,
        "lpTime": 0.0,
        "bc_improv_per_cut_avg": 0.0,
        "bc_improv_per_cut_std": 0.0,
        "bc_improv_per_cut_skew": 0.0,
        "bc_upper_lower_ratio": 0.0,
        "bc_no1s_min": 0.0,
        "bc_no1s_q25": 0.0,
        "bc_no1s_q50": 0.0,
        "bc_no1s_q75": 0.0,
        "bc_no1s_max": 0.0,
        "bc_p1s": 0.0,
        "bc_pn1s": 0.0,
        "bcTime": 0.0,
        "acc": 0.0,
        "acfTime": 0.0,
    }

    def __init__(self, filepath: Path, optimum: float):
        self.filepath = filepath
        self.optimum = optimum

    def __hash__(self):
        path_tuple = self.filepath.parts
        data_idx = path_tuple.index("data")
        return "/".join(path_tuple[data_idx:])

    def calculate_features(self) -> Tuple[float, Dict]:
        start_time = time.time()
        tspmeta_features = self._calculate_tspmeta_features()
        ubc_features = self._calculate_ubc_features()
        features = {**self.FEATURES, **tspmeta_features, **ubc_features}
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, features

    def _calculate_tspmeta_features(self) -> Dict:
        try:
            from rpy2.robjects.packages import importr

            tspmeta = importr("tspmeta")
            instance = tspmeta.read_tsplib_instance(str(self.filepath))
            features = tspmeta.features(instance)
            features = {name: features[i][0] for i, name in enumerate(features.names)}
            return features
        except Exception as e:
            logger.error(f"Error calculating tspmeta features: {e}")
            return {}

    def _calculate_ubc_features(self) -> Dict:
        if IS_WINDOWS:
            return {}

        try:
            result = subprocess.run(
                [UBC_TSP_FEATURE_PATH, "-all", self.filepath],
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip().splitlines()

            header = output[0].split(",")
            values = output[1].split(",")

            feature_dict = {header[i]: float(values[i]) for i in range(len(header))}
            return feature_dict
        except Exception as e:
            logger.error(f"Error calculating UBC features: {e}")
            return {}

    def mutate(self) -> Tuple["TSP_Instance", float]:
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

        x_min, x_max = df["X"].min(), df["X"].max()
        y_min, y_max = df["Y"].min(), df["Y"].max()

        do_shift = np.random.binomial(1, 0.9, size=df.shape[0])
        df.loc[do_shift == 1, "X"] += np.random.normal(
            0, 0.025 * (x_max - x_min), size=(do_shift == 1).sum()
        ).round(0)
        df.loc[do_shift == 1, "Y"] += np.random.normal(
            0, 0.025 * (y_max - y_min), size=(do_shift == 1).sum()
        ).round(0)
        df.loc[do_shift == 0, "X"] = np.random.uniform(
            x_min, x_max, size=(do_shift == 0).sum()
        ).round(0)
        df.loc[do_shift == 0, "Y"] = np.random.uniform(
            y_min, y_max, size=(do_shift == 0).sum()
        ).round(0)

        dir_ = DATA_DIR / "TSP" / "CEPS_generated"
        idx = len(list(dir_.glob("*.tsp")))
        out_filepath = dir_ / f"{idx}.tsp"

        with open(out_filepath, "w") as file:
            file.write(f"NAME : GENERATED_{idx}\n")
            file.write(f"TYPE : TSP\n")
            file.write(f"DIMENSION : {df.shape[0]}\n")
            file.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
            file.write(f"NODE_COORD_SECTION\n")
            for node, x, y in df.itertuples():
                file.write(f"{node} {x:.0f} {y:.0f}\n")
            file.write("EOF\n")
        instance = TSP_Instance(out_filepath, 0)
        optimum, time = instance._get_optimum_with_concorde()
        instance.optimum = optimum
        return instance, time

    def _get_optimum_with_concorde(self) -> Tuple[float, float]:
        if IS_WINDOWS:
            return 0.0, 100.0

        try:
            temp_dir = tempfile.TemporaryDirectory(dir=TEMP_DIR)
            old_cwd = os.getcwd()
            os.chdir(temp_dir.name)

            start_time = time.time()
            result = subprocess.run(
                [CONCORDE_PATH, "-x", self.filepath],
                capture_output=True,
                text=True,
            )
            end_time = time.time()

            os.chdir(old_cwd)
            temp_dir.cleanup()

            elapsed_time = end_time - start_time
            optimum = None
            for line in result.stdout.splitlines():
                if "Optimal Solution:" in line:
                    optimum = float(line.split()[-1])
                    break
            if optimum is None:
                raise Exception("Optimum not found")
            return optimum, elapsed_time
        except Exception as e:
            logger.error(f"Error calculating optimum with concorde: {e}")
            return 0.0, 100.0


class TSP_InstanceSet(InstanceSet):
    def __init__(self):
        super().__init__()

    @classmethod
    def train_test_from_index_file(
        cls,
        filepath: Path,
        train_size: int,
        seed: int = 0,
    ) -> Tuple["TSP_InstanceSet", "TSP_InstanceSet"]:
        train_instances = cls()
        test_instances = cls()

        with open(filepath) as f:
            index = json.load(f)

        instances = []
        for k, v in index.items():
            filepath = DATA_DIR / Path(k)
            instance = TSP_Instance(filepath, v)
            instances.append(instance)

        np.random.seed(seed)
        np.random.shuffle(instances)
        train_instances.extend(instances[:train_size])
        test_instances.extend(instances[train_size:])
        return train_instances, test_instances

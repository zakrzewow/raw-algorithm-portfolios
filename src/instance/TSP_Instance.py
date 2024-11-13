import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.constant import DATA_DIR, IS_WINDOWS, UBC_TSP_FEATURE_PATH
from src.instance import Instance, InstanceSet


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

    def calculate_features(self) -> Dict:
        tspmeta_features = self._calculate_tspmeta_features()
        ubc_features = self._calculate_ubc_features()
        features = {**self.FEATURES, **tspmeta_features, **ubc_features}
        return features

    def _calculate_tspmeta_features(self) -> Dict:
        from rpy2.robjects.packages import importr

        tspmeta = importr("tspmeta")
        instance = tspmeta.read_tsplib_instance(str(self.filepath))
        features = tspmeta.features(instance)
        features = {name: features[i][0] for i, name in enumerate(features.names)}
        return features

    def _calculate_ubc_features(self) -> Dict:
        if IS_WINDOWS:
            return {}

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

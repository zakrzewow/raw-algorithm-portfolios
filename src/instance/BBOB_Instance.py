import json

import cocoex
import numpy as np

from src.constant import DATA_DIR
from src.database import DB
from src.instance.Instance import Instance
from src.instance.InstanceList import InstanceList
from src.utils import ResultWithTime

with open(DATA_DIR / "BBOB" / "features.json", "r") as f:
    FEATURES = json.load(f)


class BBOB_Instance(Instance):
    FEATURES = {
        "ela_distr.skewness": 0.0,
        "ela_distr.kurtosis": 0.0,
        "ela_distr.number_of_peaks": 0.0,
        "ela_distr.costs_runtime": 0.0,
        "ela_level.mmce_lda_10": 0.0,
        "ela_level.mmce_qda_10": 0.0,
        "ela_level.lda_qda_10": 0.0,
        "ela_level.mmce_lda_25": 0.0,
        "ela_level.mmce_qda_25": 0.0,
        "ela_level.lda_qda_25": 0.0,
        "ela_level.mmce_lda_50": 0.0,
        "ela_level.mmce_qda_50": 0.0,
        "ela_level.lda_qda_50": 0.0,
        "ela_level.costs_runtime": 0.0,
        "ela_meta.lin_simple.adj_r2": 0.0,
        "ela_meta.lin_simple.intercept": 0.0,
        "ela_meta.lin_simple.coef.min": 0.0,
        "ela_meta.lin_simple.coef.max": 0.0,
        "ela_meta.lin_simple.coef.max_by_min": 0.0,
        "ela_meta.lin_w_interact.adj_r2": 0.0,
        "ela_meta.quad_simple.adj_r2": 0.0,
        "ela_meta.quad_simple.cond": 0.0,
        "ela_meta.quad_w_interact.adj_r2": 0.0,
        "ela_meta.costs_runtime": 0.0,
        "disp.ratio_mean_02": 0.0,
        "disp.ratio_mean_05": 0.0,
        "disp.ratio_mean_10": 0.0,
        "disp.ratio_mean_25": 0.0,
        "disp.ratio_median_02": 0.0,
        "disp.ratio_median_05": 0.0,
        "disp.ratio_median_10": 0.0,
        "disp.ratio_median_25": 0.0,
        "disp.diff_mean_02": 0.0,
        "disp.diff_mean_05": 0.0,
        "disp.diff_mean_10": 0.0,
        "disp.diff_mean_25": 0.0,
        "disp.diff_median_02": 0.0,
        "disp.diff_median_05": 0.0,
        "disp.diff_median_10": 0.0,
        "disp.diff_median_25": 0.0,
        "disp.costs_runtime": 0.0,
        "nbc.nn_nb.sd_ratio": 0.0,
        "nbc.nn_nb.mean_ratio": 0.0,
        "nbc.nn_nb.cor": 0.0,
        "nbc.dist_ratio.coeff_var": 0.0,
        "nbc.nb_fitness.cor": 0.0,
        "nbc.costs_runtime": 0.0,
        "pca.expl_var.cov_x": 0.0,
        "pca.expl_var.cor_x": 0.0,
        "pca.expl_var.cov_init": 0.0,
        "pca.expl_var.cor_init": 0.0,
        "pca.expl_var_PC1.cov_x": 0.0,
        "pca.expl_var_PC1.cor_x": 0.0,
        "pca.expl_var_PC1.cov_init": 0.0,
        "pca.expl_var_PC1.cor_init": 0.0,
        "pca.costs_runtime": 0.0,
        "ic.h_max": 0.0,
        "ic.eps_s": 0.0,
        "ic.eps_max": 0.0,
        "ic.eps_ratio": 0.0,
        "ic.m0": 0.0,
        "ic.costs_runtime": 0.0,
    }

    def __init__(
        self,
        function_index: int,
        dimension: int,
        instance_index: int,
        cut_off_cost: float = 0,
        cut_off_time: float = 0,
    ):
        super().__init__()
        self.function_index = function_index
        self.dimension = dimension
        self.instance_index = instance_index
        self._id = f"bbob_f{function_index:03d}_i{instance_index:02d}_d{dimension:02d}"
        self._suite_options = f"function_indices:{function_index} dimensions:{dimension} instance_indices:{instance_index}"
        self.cut_off_cost = cut_off_cost
        self.cut_off_time = cut_off_time

    def __repr__(self):
        str_ = f"BBOB_Instance(problem_id={self._id})"
        return str_

    @classmethod
    def from_db(cls, id_: str) -> "BBOB_Instance":
        dict_ = DB().select_id(DB.SCHEMA.INSTANCES, id_)
        function_index = dict_.pop("function_index")
        dimension = dict_.pop("dimension")
        instance_index = dict_.pop("instance_index")
        instance = cls(function_index, dimension, instance_index)
        instance.features = dict_
        return instance

    def to_dict(self) -> dict:
        return {
            "function_index": self.function_index,
            "dimension": self.dimension,
            "instance_index": self.instance_index,
            **self.features,
        }

    def get_problem(self) -> cocoex.Problem:
        _suite = cocoex.Suite("bbob", "", self._suite_options)
        return _suite[0]

    @classmethod
    def _calculate_features(cls, instance: "Instance", repeat=50) -> ResultWithTime:
        # import warnings

        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        # import pandas as pd
        # from pflacco.classical_ela_features import (
        #     calculate_dispersion,
        #     calculate_ela_distribution,
        #     calculate_ela_level,
        #     calculate_ela_meta,
        #     calculate_information_content,
        #     calculate_nbc,
        #     calculate_pca,
        # )
        # from pflacco.sampling import create_initial_sample

        # from src.utils import ResultWithTime, Timer

        # problem = instance.get_problem()
        # with Timer() as timer:
        #     records = []
        #     for seed in range(repeat):
        #         X = create_initial_sample(
        #             problem.dimension,
        #             n=min(1000, problem.dimension * 50),
        #             lower_bound=-5,
        #             upper_bound=5,
        #             seed=seed,
        #         )
        #         y = X.apply(lambda x: problem(x), axis=1)

        #         ela_distr = calculate_ela_distribution(X, y)
        #         ela_level = calculate_ela_level(X, y)
        #         ela_meta = calculate_ela_meta(X, y)
        #         ela_disp = calculate_dispersion(X, y)
        #         ela_nbc = calculate_nbc(X, y)
        #         ela_pca = calculate_pca(X, y)
        #         ela_ic = calculate_information_content(X, y, seed=seed)

        #         records.append(
        #             {
        #                 **ela_distr,
        #                 **ela_level,
        #                 **ela_meta,
        #                 **ela_disp,
        #                 **ela_nbc,
        #                 **ela_pca,
        #                 **ela_ic,
        #             }
        #         )

        # result = pd.DataFrame(records).median().round(6).to_dict()
        # result = {k.replace(".", "_"): v for k, v in result.items()}
        # time = round(timer.elapsed_time / repeat, 6)
        # return ResultWithTime(result, time)
        dict_ = FEATURES.get(instance.id(), {})
        result = dict_.get("result", {})
        time = dict_.get("time", 0.0)
        return ResultWithTime(result, time)

    def plot(self, fname: str = None):
        import matplotlib.pyplot as plt

        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)
        problem = self.get_problem()

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = [X[i, j], Y[i, j]]
                Z[i, j] = problem(point)

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.colorbar(surf, ax=ax, shrink=0.5)
        plt.title(f"{self}", fontsize=10)
        plt.grid(color="black", alpha=0.05)
        if fname:
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def BBOB_test() -> InstanceList:
    function_index_list = list(range(1, 25))
    dimension_list = [2, 3, 5, 10, 20]
    instance_index_list = [1, 2]
    instance_list = InstanceList()

    for function_index in function_index_list:
        for dimension in dimension_list:
            for instance_index in instance_index_list:
                instance = BBOB_Instance(
                    function_index=function_index,
                    dimension=dimension,
                    instance_index=instance_index,
                )
                instance.cut_off_time = 100.0
                instance.cut_off_cost = 10 * instance.cut_off_time
                instance_list.append(instance)
    return instance_list


def BBOB_train(
    n: int = None,
    seed: int = 0,
) -> InstanceList:
    rng = np.random.default_rng(seed=seed)
    function_index_list = list(range(1, 25))
    dimension_list = [2, 3, 5, 10, 20]
    instance_index_list = [3, 4, 5]
    n_per_dimension = n // len(dimension_list)
    instance_list = InstanceList()

    for dimension in dimension_list:
        sampled_pairs = rng.choice(
            len(function_index_list) * len(instance_index_list),
            size=n_per_dimension,
            replace=False,
        )

        for idx in sampled_pairs:
            function_idx = function_index_list[idx // len(instance_index_list)]
            instance_idx = instance_index_list[idx % len(instance_index_list)]

            bbob_instance = BBOB_Instance(
                function_index=function_idx,
                dimension=dimension,
                instance_index=instance_idx,
            )
            instance_list.append(bbob_instance)
    return instance_list


def set_08_cut_off_time(instance_list: InstanceList):
    dimension_to_cut_off_time = {
        2: 0.50,
        3: 1.05,
        5: 2.25,
        10: 6.65,
        20: 20.0,
    }

    for instance in instance_list:
        instance.cut_off_time = dimension_to_cut_off_time.get(instance.dimension, 20.0)
        instance.cut_off_cost = 10 * instance.cut_off_time
    return instance_list

import cocoex
import numpy as np

from src.instance.Instance import Instance
from src.utils import ResultWithTime


class BBOB_Instance(Instance):
    FEATURES = {}

    def __init__(
        self,
        function_index: int,
        dimension: int,
        instance_index: int,
        cut_off_cost: float = 0,
        cut_off_time: float = 0,
    ):
        super().__init__()
        self._suite_options = f"function_indices:{function_index} dimensions:{dimension} instance_indices:{instance_index}"
        self._suite = cocoex.Suite("bbob", "", self._suite_options)
        self.cut_off_cost = cut_off_cost
        self.cut_off_time = cut_off_time

    def __repr__(self):
        str_ = f"BBOB_Instance(id={self._suite_options})"
        return str_

    @classmethod
    def from_db(cls, id_: str) -> "BBOB_Instance":
        raise NotImplementedError()

    def to_dict(self) -> dict:
        return {
            **self.features,
        }

    def get_problem(self) -> cocoex.Problem:
        return self._suite[0]

    @classmethod
    def _calculate_features(cls, instance: "Instance") -> ResultWithTime:
        raise NotImplementedError()
        # return ResultWithTime(result, time)

    def plot(self, fname: str = None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

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


# def TSP_from_index_file(
#     filepath: Path,
#     cut_off_cost: float = 0.0,
#     cut_off_time: float = 0.0,
#     n: int = None,
#     seed: int = 0,
# ) -> InstanceList:
#     instances = InstanceList()

#     with open(filepath) as f:
#         index = json.load(f)

#     for k, v in index.items():
#         filepath = DATA_DIR / Path(k)
#         instance = BBOB_Instance(filepath, v, cut_off_cost, cut_off_time)
#         instances.append(instance)

#     if n is not None:
#         rng = np.random.default_rng(seed=seed)
#         tsp_generators = set([instance.tsp_generator for instance in instances])
#         n_generators = len(tsp_generators)

#         if n % n_generators != 0:
#             raise ValueError(f"{n=} must be divisible by the number of {n_generators=}")

#         generators_dict = {}
#         for instance in instances:
#             generator = instance.tsp_generator
#             if generator not in generators_dict:
#                 generators_dict[generator] = []
#             generators_dict[generator].append(instance)

#         samples_per_generator = n // n_generators

#         new_instances = InstanceList()
#         for generator, generator_instances in generators_dict.items():
#             if len(generator_instances) < samples_per_generator:
#                 raise ValueError(
#                     f"Not enough instances for generator {generator}. Needed {samples_per_generator}, but only have {len(generator_instances)}"
#                 )
#             selected_instances = rng.choice(
#                 generator_instances,
#                 size=samples_per_generator,
#                 replace=False,
#             )
#             new_instances.extend(selected_instances)

#         instances = new_instances

#     return instances


# def set_n22_cut_off_time(
#     instances: InstanceList,
#     reference_cut_off_time: float = 10.0,
# ):
#     for instance in instances:
#         instance.cut_off_time = round(
#             reference_cut_off_time * ((instance.n_cities / 600) ** 2.2), 2
#         )
#         instance.cut_off_cost = 10 * instance.cut_off_time
#     return instances

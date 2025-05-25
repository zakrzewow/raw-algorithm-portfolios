from src.constant import DATA_DIR
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

solver = SAT_Riss_Solver()
instances = SAT_from_index_file(
    filepath=DATA_DIR / "SAT" / "index.json",
    max_cost=10.0,
    max_time=10.0,
)

for instance in instances:
    solver.solve(instance, prefix="test")

from src.constant import DATA_DIR
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

instances = SAT_from_index_file(
    filepath=DATA_DIR / "SAT" / "index_simple.json",
    max_cost=100.0,
    max_time=10.0,
)

for i in range(3):
    solver = SAT_Riss_Solver()

    for instance in instances:
        solver.solve(instance, prefix=f"test-{i}", cache=False)

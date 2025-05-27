from src.constant import DATA_DIR
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

instances = SAT_from_index_file(
    filepath=DATA_DIR / "SAT" / "index_u150.json",
    max_cost=100.0,
    max_time=10.0,
)

for i in range(3):
    solver = SAT_Riss_Solver()

    for instance in instances:
        try:
            solver.solve(instance, prefix=f"test-{i}", cache=False)
        except Exception as e:
            print(f"Error solving instance {instance}: {e}")

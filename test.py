from src.aac.SurrogateEstimator import Estimator1
from src.constant import DATA_DIR
from src.database import DB
from src.database.queries import get_model_training_data
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

instances = SAT_from_index_file(
    filepath=DATA_DIR / "SAT" / "index.json",
    max_cost=100.0,
    max_time=10.0,
)[:5]

for i in range(3):
    solver = SAT_Riss_Solver()

    for instance in instances:
        try:
            instance.calculate_features()
            solver.solve(instance, prefix=f"test-{i}", cache=False)
        except Exception as e:
            print(f"Error solving instance {instance}: {e}")

db = DB()

X, y = get_model_training_data(db)
print(X, y)
estimator = Estimator1(max_cost=100.0, estimator_pct=0.5)
estimator.fit(X, y)
estimator.log()

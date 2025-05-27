from src.aac.AAC import AAC
from src.constant import DATA_DIR
from src.instance.SAT_Instance import SAT_from_index_file
from src.solver.Portfolio import Portfolio
from src.solver.SAT_Riss_Solver import SAT_Riss_Solver

if __name__ == "__main__":
    instances = SAT_from_index_file(
        filepath=DATA_DIR / "SAT" / "index.json",
        max_cost=100.0,
        max_time=10.0,
    )

    portfolio = Portfolio.from_solver_class(SAT_Riss_Solver, size=2)

    aac = AAC(
        portfolio=portfolio,
        instance_list=instances,
        prefix="config",
        max_iter=75,
        calculate_features=False,
        estimator=None,
    )
    aac.configure()

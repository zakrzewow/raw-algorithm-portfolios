import sqlite3

import pandas as pd
from ConfigSpace import Configuration

from src.constant import DATA_DIR, MAIN_DIR
from src.database import db_init
from src.instance.TSP_Instance import TSP_Instance, TSP_InstanceSet
from src.portfolio import Portfolio
from src.solver.TSP_LKH_Solver import TSP_LKH_Solver

if __name__ == "__main__":
    train_instances, test_instances = TSP_InstanceSet.train_test_from_index_file(
        filepath=DATA_DIR / "TSP" / "CEPS_benchmark" / "index.json",
        train_size=0,
        seed=0,
    )

    for db in ["CEPS1.db", "CEPS2.db"]:
        conn = sqlite3.connect(MAIN_DIR / db)
        portfolio = pd.read_sql_query(
            """
        WITH portfolio AS (
            SELECT
                DISTINCT solver_id 
            FROM results WHERE comment LIKE 'test%'
        )
        SELECT solvers.* 
        FROM solvers
        JOIN portfolio ON solvers.id = portfolio.solver_id
        """,
            conn,
        )
        conn.close()

        portfolio = [
            Configuration(
                TSP_LKH_Solver.CONFIGURATION_SPACE, values=config.drop(["id"]).to_dict()
            )
            for _, config in portfolio.iterrows()
        ]
        portfolio = [TSP_LKH_Solver(config) for config in portfolio]
        portfolio = Portfolio.from_solver_list(portfolio)

        solver_class = TSP_LKH_Solver
        instance_class = TSP_Instance
        db_init(solver_class, instance_class, False)

        portfolio.evaluate(test_instances, comment="test1", cache=False)
        portfolio.evaluate(test_instances, comment="test2", cache=False)
        portfolio.evaluate(test_instances, comment="test3", cache=False)
        portfolio.evaluate(test_instances, comment="test4", cache=False)
        portfolio.evaluate(test_instances, comment="test5", cache=False)

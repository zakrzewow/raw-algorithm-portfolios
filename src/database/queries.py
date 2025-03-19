import pandas as pd

from src.database.db import DB


def get_model_training_data(db: DB) -> pd.DataFrame:
    query = f"""
    select 
        {db.SCHEMA.EVALUATIONS}.cost,
        {db.SCHEMA.SOLVERS}.*,
        {db.SCHEMA.INSTANCES}.*
    from {db.SCHEMA.EVALUATIONS}
    join {db.SCHEMA.INSTANCES} on {db.SCHEMA.EVALUATIONS}.instance_id = {db.SCHEMA.INSTANCES}.id
    join {db.SCHEMA.SOLVERS} on {db.SCHEMA.EVALUATIONS}.solver_id = {db.SCHEMA.SOLVERS}.id
    """
    df = db.query2df(query)
    df = df.drop(columns=["id", "filepath", "optimum"])
    df = df.dropna()
    y = df["cost"].to_numpy()
    X = df.drop(columns="cost").to_numpy()
    return X, y

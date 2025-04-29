import numpy as np
import pandas as pd

from src.database.db import DB


def get_model_training_data(db: DB) -> pd.DataFrame:
    query = f"""
    select 
        {db.SCHEMA.RESULTS}.cost,
        {db.SCHEMA.RESULTS}.cut_off_time,
        {db.SCHEMA.SOLVERS}.*,
        {db.SCHEMA.INSTANCES}.*
    from {db.SCHEMA.RESULTS}
    join {db.SCHEMA.INSTANCES} on {db.SCHEMA.RESULTS}.instance_id = {db.SCHEMA.INSTANCES}.id
    join {db.SCHEMA.SOLVERS} on {db.SCHEMA.RESULTS}.solver_id = {db.SCHEMA.SOLVERS}.id
    where {db.SCHEMA.RESULTS}.cached = 0 and {db.SCHEMA.RESULTS}.surrogate = 0
    """
    df = db.query2df(query)
    df = df.drop(columns=["id", "filepath", "optimum"])
    df = df.dropna()
    y = df["cost"].to_numpy()
    cut_off = df["cut_off_time"].to_numpy()
    y = np.where(y >= cut_off, cut_off, y)
    X = df.drop(columns=["cost", "cut_off_time"]).to_numpy()
    return X, y, cut_off


def get_solvers_count(db: DB) -> int:
    if not db._table_exists(DB.SCHEMA.SOLVERS):
        return 0
    return db.query2df(f"select count(*) from {DB.SCHEMA.SOLVERS}").iloc[0, 0]

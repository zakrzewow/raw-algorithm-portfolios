import datetime as dt
import sqlite3
from typing import Dict, Tuple, Type

import numpy as np

from src.constant import DATABASE_DIR
from src.instance import Instance
from src.solver import Solver

DB_PATH = DATABASE_DIR / f"{dt.datetime.now():%Y_%m_%d_%H_%M_%S}.db"


def db_connect():
    return sqlite3.connect(DB_PATH)


def _db_create_solvers_table(conn, solver_class: Type[Solver]):
    query = ["id TEXT PRIMARY KEY"]
    for k, v in solver_class.CONFIGURATION_SPACE.items():
        type_ = None
        if type(v.default_value) == int:
            type_ = "INTEGER"
        elif type(v.default_value) == float:
            type_ = "REAL"
        elif type(v.default_value) == str:
            type_ = "TEXT"
        query.append(f"{k} {type_}")
    query = ",\n".join(query)
    query = f"CREATE TABLE solvers ({query})"

    with conn:
        conn.execute(query)


def _db_create_instances_table(
    conn,
    instance_class: Type[Instance],
    calculate_instance_features: bool,
):
    query = ["id TEXT PRIMARY KEY"]
    if calculate_instance_features:
        for k, v in instance_class.FEATURES.items():
            type_ = None
            if type(v) == int:
                type_ = "INTEGER"
            elif type(v) == float:
                type_ = "REAL"
            elif type(v) == str:
                type_ = "TEXT"
            query.append(f"{k} {type_}")
    query = ",\n".join(query)
    query = f"CREATE TABLE instances ({query})"
    with conn:
        conn.execute(query)


def _db_create_results_table(conn):
    query = (
        "CREATE TABLE results ("
        "instance_id TEXT,"
        "solver_id TEXT,"
        "cost REAL,"
        "time REAL,"
        "comment TEXT,"
        "created_at DATETIME DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    with conn:
        conn.execute(query)


def db_init(
    solver_class: Type[Solver],
    instance_class: Type[Instance],
    calculate_instance_features: bool = False,
):
    conn = db_connect()
    _db_create_solvers_table(conn, solver_class)
    _db_create_instances_table(conn, instance_class, calculate_instance_features)
    _db_create_results_table(conn)
    conn.close()


def db_solver_exists(conn, solver: Solver) -> bool:
    id_ = str(hash(solver))
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM solvers WHERE id = ?", (id_,))
    return cursor.fetchone() is not None


def db_insert_solver(conn, solver: Solver):
    if db_solver_exists(conn, solver):
        return

    id_ = str(hash(solver))

    values = [id_] + list(solver.config.values())
    for i in range(len(values)):
        if isinstance(values[i], np.integer):
            values[i] = int(values[i])
        elif isinstance(values[i], np.floating):
            values[i] = float(values[i])
    query = f"INSERT INTO solvers VALUES ({', '.join(['?'] * len(values))})"

    with conn:
        conn.execute(query, values)


def db_instance_exists(conn, instance: Instance) -> bool:
    id_ = instance.__hash__()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM instances WHERE id = ?", (id_,))
    return cursor.fetchone() is not None


def db_insert_instance(conn, instance: Instance, features: Dict = {}):
    if db_instance_exists(conn, instance):
        return

    id_ = instance.__hash__()

    columns = ", ".join(["id"] + list(features.keys()))
    values = [id_] + list(features.values())
    question_marks = ", ".join(["?"] * (len(features) + 1))

    query = f"INSERT INTO instances ({columns}) VALUES ({question_marks})"
    with conn:
        conn.execute(query, values)


def db_insert_result(
    conn,
    instance: Instance,
    solver: Solver,
    cost: float,
    time: float,
    comment: str,
):
    instance_id = instance.__hash__()
    solver_id = str(hash(solver))

    query = "INSERT INTO results (instance_id, solver_id, cost, time, comment) VALUES (?, ?, ?, ?, ?)"
    with conn:
        conn.execute(query, (instance_id, solver_id, cost, time, comment))


def db_fetch_result(conn, instance: Instance, solver: Solver) -> Tuple[float, float]:
    instance_id = instance.__hash__()
    solver_id = str(hash(solver))

    query = "SELECT MIN(cost), MIN(time) FROM results WHERE instance_id = ? AND solver_id = ?"
    cursor = conn.cursor()
    cursor.execute(query, (instance_id, solver_id))
    result = cursor.fetchone()
    if result[0] is not None and result[1] is not None:
        return result[0], result[1]
    return None

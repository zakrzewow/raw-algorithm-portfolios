import sqlite3
from enum import Enum

import pandas as pd

from src.constant import DATABASE_DIR, JOB_ID, JOB_NAME, N_TRAIN, PARG, POLICY

DB_PATH = DATABASE_DIR / f"{JOB_NAME}-{POLICY}-{PARG}-{N_TRAIN}-{JOB_ID}.db"


class DB:
    class SCHEMA(Enum):
        SOLVERS = "solvers"
        INSTANCES = "instances"
        RESULTS = "results"
        EVALUATIONS = "evaluations"

        def __str__(self):
            return self.value

    def __init__(self, db_path: str = DB_PATH):
        self._conn = sqlite3.connect(db_path, isolation_level=None)

    def __del__(self):
        self._conn.close()

    def insert(self, table: "DB.SCHEMA", id_: str = None, data: dict = {}):
        if not self._table_exists(table):
            self._create_table_from_data(table, id_, data)
        if self._row_exists(table, id_):
            return
        query = self._get_insert_query(table, id_, data)
        values = self._get_insert_values(id_, data)
        self._conn.execute(query, values)

    def _table_exists(self, table: "DB.SCHEMA") -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name=?;
            """,
            (table.value,),
        )
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def _create_table_from_data(self, table: "DB.SCHEMA", id_: None, data: dict = {}):
        query = []
        if id_ is not None:
            query.append("id TEXT PRIMARY KEY")

        for k, v in data.items():
            if isinstance(v, int):
                type_ = "INTEGER"
            elif isinstance(v, float):
                type_ = "REAL"
            elif isinstance(v, str):
                type_ = "TEXT"
            q = f"{k} {type_}"
            query.append(q)
        query = ", ".join(query)
        query = f"CREATE TABLE {table} ({query})"
        self._conn.execute(query)

    def _row_exists(self, table: "DB.SCHEMA", id_: str = None) -> bool:
        if id_ is None:
            return False
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT 1 FROM {table} WHERE id = ?", (id_,))
        exists = cursor.fetchone() is not None
        cursor.close()
        return exists

    def _get_insert_query(
        self,
        table: "DB.SCHEMA",
        id_: str = None,
        data: dict = {},
    ) -> str:
        columns = []
        if id_ is not None:
            columns.append("id")
        columns.extend(data.keys())
        n = len(columns)
        cols_str = ", ".join(columns)
        question_marks = ", ".join(["?"] * n)
        query = f"INSERT INTO {table} ({cols_str}) VALUES ({question_marks})"
        return query

    def _get_insert_values(self, id_: str = None, data: dict = {}) -> list:
        values = []
        if id_ is not None:
            values.append(id_)
        values.extend(data.values())
        return values

    def select_id(self, table: "DB.SCHEMA", id_: str) -> dict:
        if not self._table_exists(table):
            return {}
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE id = ?", (id_,))
        cols = [description[0] for description in cursor.description]
        values = cursor.fetchone()
        cursor.close()
        if values is None:
            return {}
        return dict(zip(cols, values))

    def query2df(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self._conn)

    def get_instances(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.SCHEMA.INSTANCES}"
        return self.query2df(query)

    def get_solvers(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.SCHEMA.SOLVERS}"
        return self.query2df(query)

    def get_results(self) -> pd.DataFrame:
        query = f"SELECT * FROM {self.SCHEMA.RESULTS}"
        return self.query2df(query)

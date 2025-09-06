# This file is for Database connections,Schema Autodetections, Safe SQL Executions and To Trace Chinook Database Fallbacks

import os
import re
import requests
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
import sqlparse
from typing import Dict, Any, List

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)

def ensure_chinook_file(path: str = ".data/chinook.sqlite"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    try:
        r = requests.get(CHINOOK_URL, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception:
        # create tiny fallback if download fails
        from sqlite3 import connect
        conn = connect(path)
        cur = conn.cursor()
        cur.executescript("""
DROP TABLE IF EXISTS Customers;
DROP TABLE IF EXISTS Invoices;
CREATE TABLE Customers(CustomerId INTEGER PRIMARY KEY, FirstName TEXT, LastName TEXT, Country TEXT);
CREATE TABLE Invoices(InvoiceId INTEGER PRIMARY KEY, CustomerId INTEGER, InvoiceDate TEXT, BillingCountry TEXT, Total REAL);
INSERT INTO Customers VALUES (1,'Jane','Doe','USA'),(2,'Arun','K','India'),(3,'Maria','Santos','Brazil');
INSERT INTO Invoices VALUES (1,1,'2024-11-10','USA',25.0),(2,1,'2025-01-12','USA',18.0),(3,2,'2025-02-05','India',30.0);
""")
        conn.commit()
        conn.close()
        return path

def create_engine_from_uri(uri: str) -> Engine:
    # If user inputs a local sqlite path (not a sqlalchemy uri), convert to uri
    if uri.strip() == "" or uri.strip().lower() == "chinook":
        dbfile = ensure_chinook_file()
        uri = f"sqlite:///{dbfile}"
    if uri.startswith("sqlite:///") and uri.count("/") >= 3:
        # ok
        pass
    return create_engine(uri, future=True)

def get_inspector(engine: Engine):
    return inspect(engine)

def get_schema(engine: Engine) -> Dict[str, Any]:
    insp = get_inspector(engine)
    tables = insp.get_table_names()
    schema = {}
    for t in tables:
        cols = insp.get_columns(t)
        fks = insp.get_foreign_keys(t)
        pk = insp.get_pk_constraint(t)
        schema[t] = {
            "columns": cols,  # list of dicts with name & type
            "foreign_keys": fks,  # list of fk dicts
            "primary_key": pk.get("constrained_columns", [])
        }
    return schema

# ---- safe SQL execution utilities ----
DESCTRUCTIVE = ["drop", "delete", "update", "insert", "alter", "truncate", "attach", "detach", "vacuum", "reindex", "pragma"]

def is_safe_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    # parse using sqlparse to ensure single statement
    try:
        parsed = sqlparse.parse(sql)
        if len(parsed) != 1:
            return False
        token0 = parsed[0].token_first(skip_cm=True)
        if token0 is None:
            return False
        # ensure first non-whitespace token is SELECT
        if token0.normalized.upper() != "SELECT":
            return False
    except Exception:
        return False
    lower = sql.lower()
    # ban destructive words anywhere
    for bw in DESCTRUCTIVE:
        if re.search(r"\b" + re.escape(bw) + r"\b", lower):
            return False
    # ban semicolon chaining
    if ";" in lower.strip()[1:]:
        return False
    return True

def _maybe_add_limit(sql: str, engine: Engine, limit: int = 500) -> str:
    if "limit" in sql.lower():
        return sql
    dialect = engine.dialect.name
    if dialect in ("sqlite", "postgresql", "mysql", "mariadb"):
        return sql.strip() + f" LIMIT {limit}"
    return sql

def run_sql(engine: Engine, sql: str, limit: int = 500) -> pd.DataFrame:
    if not is_safe_sql(sql):
        raise ValueError("Only single SELECT queries allowed. Query rejected as unsafe.")
    sql_to_run = _maybe_add_limit(sql, engine, limit=limit)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql_to_run), conn)
    return df

def list_tables(engine: Engine) -> List[str]:
    insp = get_inspector(engine)
    return insp.get_table_names()


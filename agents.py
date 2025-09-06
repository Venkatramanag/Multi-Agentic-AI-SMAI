# Here we define 4 different type of agents like schema agent, query planner agent, nl2sql agent(heuristic + optional LLM), sql executor agent, visualization agent and insight generator agent
# Each agent has its own class and methods to perform specific tasks related to database querying and analysis
# The agents can be used together to create a pipeline for natural language querying of a database, from understanding the schema to generating SQL queries, executing them, visualizing results, and summarizing insights.

import os
import re
import time
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import date
from dateutil.relativedelta import relativedelta

# Optional OpenAI usage
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

from db import get_schema, run_sql

# ---------- SchemaFormatter ----------
class SchemaFormatter:
    def __init__(self, schema: Dict[str, Dict]):
        self.schema = schema

    def formatted(self) -> str:
        lines = []
        for t, meta in self.schema.items():
            cols = meta["columns"]
            lines.append(f"Table {t}:")
            for c in cols:
                lines.append(f" - {c['name']} ({c.get('type')})")
            if meta.get("foreign_keys"):
                for fk in meta["foreign_keys"]:
                    lines.append(f" FK -> {fk.get('referred_table')}({fk.get('referred_columns')})")
            lines.append("")
        return "\n".join(lines)

# ---------- SchemaAgent ----------
class SchemaAgent:
    def __init__(self, engine):
        self.engine = engine
        self.schema = get_schema(engine)

    def build_schema_text(self) -> str:
        return SchemaFormatter(self.schema).formatted()

# ---------- QueryPlannerAgent ----------
class QueryPlannerAgent:
    def suggest(self, schema: Dict[str, Dict]) -> List[str]:
        tables = set(schema.keys())
        ideas = []
        if "Invoices" in tables:
            ideas += [
                "total sales by BillingCountry this year",
                "monthly sales trend",
                "top 10 customers by total spend",
                "invoice count by year",
            ]
        if "Customers" in tables:
            ideas += [
                "customer count by Country",
                "top countries by number of customers",
            ]
        if not ideas:
            ideas = ["row count by table", "show first 10 rows from <table>"]
        return ideas

# ---------- NL2SQLAgent ----------
class NL2SQLAgent:
    AGG_KEYWORDS = ["sum", "total", "avg", "average", "count", "max", "min", "median"]
    TIME_PHRASES = ["last year", "this year", "last month", "last 30 days", "last 90 days", "last 7 days"]

    def __init__(self, schema: Dict[str, Dict], use_llm: bool = False, openai_model: str = "gpt-3.5-turbo"):
        self.schema = schema
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self.openai_model = openai_model

        # Build quick lookups
        self.col_to_table = {}
        for t, meta in self.schema.items():
            for c in meta["columns"]:
                self.col_to_table[c["name"].lower()] = t

    def _detect_time_clause(self, text: str, date_col_candidates: List[str]) -> str:
        low = text.lower()
        today = date.today()
        if "last year" in low:
            start = date(today.year - 1, 1, 1)
            end = date(today.year - 1, 12, 31)
            return f"{date_col_candidates[0]} BETWEEN '{start}' AND '{end}'"
        if "this year" in low:
            start = date(today.year, 1, 1)
            return f"{date_col_candidates[0]} >= '{start}'"
        if "last month" in low:
            start = (today.replace(day=1) - relativedelta(months=1)).replace(day=1)
            end = start + relativedelta(months=1, days=-1)
            return f"{date_col_candidates[0]} BETWEEN '{start}' AND '{end}'"
        m = re.search(r"last\s+(\d{1,3})\s+days", low)
        if m:
            n = int(m.group(1))
            return f"{date_col_candidates[0]} >= date('now','-{n} days')"
        return ""

    def _find_date_columns(self) -> List[str]:
        res = []
        for t, meta in self.schema.items():
            for c in meta["columns"]:
                if "date" in (c['name'] or "").lower():
                    res.append(f"{t}.{c['name']}")
        return res

    def _choose_numeric_column(self, table: str) -> Optional[str]:
        meta = self.schema.get(table, {})
        if not meta:
            return None
        for c in meta["columns"]:
            n = c['name'].lower()
            if any(k in n for k in ("total", "amount", "price", "revenue", "sales")):
                return f"{table}.{c['name']}"
        for c in meta["columns"]:
            if str(c.get('type')).lower() in ("integer", "int", "real", "numeric", "float", "double"):
                return f"{table}.{c['name']}"
        return None

    def _tables_from_text(self, text: str) -> List[str]:
        found = set()
        for t in self.schema.keys():
            if t.lower() in text.lower():
                found.add(t)
        # Detect via column names
        for col, t in self.col_to_table.items():
            if re.search(r"\b" + re.escape(col) + r"\b", text.lower()):
                found.add(t)
        return list(found)

    def _build_join_clause(self, tables: List[str]) -> Tuple[str, List[str]]:
        if len(tables) == 1:
            return f"FROM {tables[0]}", tables
        joins = []
        used = set([tables[0]])
        base = tables[0]
        for t in tables[1:]:
            joined = False
            # Check if t has fk to any used table
            for other in used:
                fk_list = self.schema[t].get("foreign_keys", []) or []
                for fk in fk_list:
                    if fk.get("referred_table") == other:
                        left_cols = fk.get("constrained_columns", [])
                        right_cols = fk.get("referred_columns", [])
                        if left_cols and right_cols:
                            joins.append(f"JOIN {t} ON {t}.{left_cols[0]} = {other}.{right_cols[0]}")
                            joined = True
                            break
                    if joined:
                        break
                # Reverse: other has fk to t?
                fk_list2 = self.schema[other].get("foreign_keys", []) or []
                for fk2 in fk_list2:
                    if fk2.get("referred_table") == t:
                        left_cols = fk2.get("constrained_columns", [])
                        right_cols = fk2.get("referred_columns", [])
                        if left_cols and right_cols:
                            joins.append(f"JOIN {t} ON {other}.{left_cols[0]} = {t}.{right_cols[0]}")
                            joined = True
                            break
                    if joined:
                        break
            if not joined:
                joins.append(f", {t}")
            used.add(t)
        clause = f"FROM {base} " + " ".join(joins)
        return clause, list(used)

    def to_sql(self, text: str) -> Tuple[str, str]:
        text = text.strip()
        low = text.lower()

        # If LLM mode and available -> delegate
        if self.use_llm:
            try:
                prompt = (
                    "You are a SQL generator for an unknown SQL database. "
                    "Given the schema below, produce a single valid SELECT statement for SQLite/Postgres/MySQL only. "
                    "Return the SQL only (no explanation). Use table names and column names exactly as in schema.\n\n"
                    f"SCHEMA:\n{SchemaFormatter(self.schema).formatted()}\n\n"
                    f"USER QUERY: {text}\nSQL:"
                )
                resp = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                sql = resp.choices[0].message.content.strip().strip("`")
                return sql, "Generated by LLM"
            except Exception:
                # Fallback to heuristic
                pass

        # Heuristic rules:
        tables = self._tables_from_text(low)
        # 1) Simple "show first N rows from X"
        m = re.search(r"(show|list)\s+(first\s+|top\s+)?(\d+)\s+(rows|records)\s+(from\s+)?(\w+)", low)
        if m:
            n = int(m.group(3))
            t = m.group(6)
            if t.capitalize() in self.schema:
                return f"SELECT * FROM {t.capitalize()} LIMIT {n}", f"Previewing first {n} rows from {t.capitalize()}"

        # 2) Row counts / counts by table
        if "row count by table" in low or ("row counts" in low and "table" in low):
            parts = []
            for t in self.schema.keys():
                parts.append(f"SELECT '{t}' AS table_name, COUNT(*) AS row_count FROM {t}")
            return " UNION ALL ".join(parts), "Row counts per table"

        # 3) Simple group-by / aggregations for sales-like tables
        cand_table = None
        for name in ["Invoices", "Sales", "Orders", "Transactions"]:
            if name in self.schema:
                cand_table = name
                break
        if not cand_table and tables:
            cand_table = tables[0]

        if cand_table:
            num_col = self._choose_numeric_column(cand_table)
            mby = re.search(r"by\s+([\w\.]+)", low)
            if "monthly" in low or "per month" in low:
                date_cols = self._find_date_columns()
                date_col = date_cols[0] if date_cols else None
                if date_col:
                    sql = f"SELECT strftime('%Y-%m', {date_col}) AS month, SUM({num_col}) AS total FROM {cand_table} WHERE 1=1 GROUP BY month ORDER BY month"
                    return sql, "Monthly sales/aggregate by detected date column"
            if any(k in low for k in ("total", "sum", "revenue", "sales", "amount")) and num_col:
                if mby:
                    grp = mby.group(1)
                    if "." not in grp:
                        grp_col = None
                        for t, meta in self.schema.items():
                            for c in meta["columns"]:
                                if c["name"].lower() == grp:
                                    grp_col = f"{t}.{c['name']}"
                                    break
                            if grp_col:
                                break
                        if grp_col:
                            sql = f"SELECT {grp_col} AS groupkey, SUM({num_col}) AS total FROM {cand_table} GROUP BY {grp_col} ORDER BY total DESC"
                            return sql, f"Sum({num_col}) grouped by {grp_col}"
                        else:
                            sql = f"SELECT {grp} AS groupkey, SUM({num_col}) AS total FROM {cand_table} GROUP BY {grp} ORDER BY total DESC"
                            return sql, "Grouped aggregate"
                sql = f"SELECT SUM({num_col}) AS total FROM {cand_table}"
                return sql, f"Total sum of {num_col}"

        # 4) Customer counts by country
        if "customer" in low and "country" in low and "Customers" in self.schema:
            return "SELECT Country AS Country, COUNT(*) AS CustomerCount FROM Customers GROUP BY Country ORDER BY CustomerCount DESC", "Customer count by country"

        # 5) Invoice count by year
        if "invoice count" in low or ("number of invoices" in low):
            if "Invoices" in self.schema:
                return "SELECT strftime('%Y', InvoiceDate) AS Year, COUNT(*) AS InvoiceCount FROM Invoices GROUP BY Year ORDER BY Year", "Invoice count by year"

        # Fallback: list tables
        return "SELECT name FROM sqlite_master WHERE type='table';", "Fallback: listing tables (no heuristic matched)"

# ---------- SQLExecutorAgent ----------
class SQLExecutorAgent:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, sql: str):
        start = time.time()
        df = run_sql(self.engine, sql)
        dur = time.time() - start
        return df, dur

# ---------- VisualizationAgent ----------
import plotly.express as px

class VisualizationAgent:
    def choose_and_plot(self, df: pd.DataFrame):
        if df is None or df.empty:
            return "table", None
        cols = df.columns.tolist()
        if len(cols) == 1:
            return "table", None
        x = cols[0]
        y = cols[-1]
        if re.search(r"(date|year|month)", str(x).lower()) or pd.to_datetime(df[x], errors='coerce').notna().all():
            try:
                df[x] = pd.to_datetime(df[x], errors='coerce')
                fig = px.line(df, x=x, y=y, markers=True, title=f"{y} over {x}")
                return "line", fig
            except Exception:
                pass
        if len(cols) == 2 and pd.api.types.is_numeric_dtype(df[cols[1]]):
            fig = px.bar(df, x=cols[0], y=cols[1], title=f"{cols[1]} by {cols[0]}")
            return "bar", fig
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            fig = px.bar(df, x=cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cols[0]}")
            return "bar", fig
        return "table", None

# ---------- InsightAgent ----------
class InsightAgent:
    def summarize(self, df: pd.DataFrame, sql: str, user_q: str) -> str:
        if df is None or df.empty:
            return "No results to summarize."
        cols = df.columns.tolist()
        if len(cols) == 2 and pd.api.types.is_numeric_dtype(df[cols[1]]):
            sorted_df = df.sort_values(cols[1], ascending=False)
            topk = sorted_df.head(3)
            lines = [f"Top {len(topk)} by {cols[1]}:"]
            for _, r in topk.iterrows():
                lines.append(f"- {r[cols[0]]}: {r[cols[1]]}")
            return "\n".join(lines)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            c = num_cols[0]
            s = df[c].describe()
            return f"{c} — count: {int(s['count'])}, mean: {s['mean']:.2f}, min: {s['min']:.2f}, max: {s['max']:.2f}"
        return f"Returned {len(df)} rows and {len(cols)} columns."
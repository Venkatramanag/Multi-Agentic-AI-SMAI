# agents1.py
import re
import os
import pandas as pd
import google.generativeai as genai
from sqlalchemy import text, inspect
from db import get_schema, is_safe_sql

# Gemini config: use env var if set; keep your provided key as fallback.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBheljzKOEL1K37-DcQ63f6KRKpLbQE6gI")
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------
# SchemaAgent (unchanged)
# -------------------------
class SchemaAgent:
    def __init__(self, engine, schema=None):
        self.engine = engine
        if schema:
            self.schema = schema
        else:
            self.schema = self._introspect()

    def _introspect(self):
        insp = inspect(self.engine)
        schema = {}
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            fks = insp.get_foreign_keys(table)
            pk = insp.get_pk_constraint(table)
            schema[table] = {
                "columns": cols,
                "foreign_keys": fks,
                "primary_key": pk.get("constrained_columns", []),
            }
        return schema

# -------------------------
# QueryPlannerAgent (unchanged)
# -------------------------
class QueryPlannerAgent:
    def suggest(self, schema):
        suggestions = []
        for table, meta in schema.items():
            colnames = [c["name"] for c in meta["columns"]]
            suggestions.append(f"Show all rows from {table}")
            if len(colnames) >= 2:
                suggestions.append(f"Show {colnames[0]} and {colnames[1]} from {table}")
        return suggestions

# -------------------------
# NL2SQLAgent: heuristics first, then Gemini (updated)
# -------------------------
class NL2SQLAgent:
    def __init__(self, schema: dict, use_llm: bool = True, gemini_api_key: str = None):
        self.schema = schema
        self.use_llm = use_llm
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        if self.use_llm and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

        # Column->table mapping for heuristics
        self.col_to_table = {}
        for t, meta in self.schema.items():
            for c in meta["columns"]:
                self.col_to_table[c["name"].lower()] = t

    # Heuristic rules (kept intact)
    def _heuristic_generate_sql(self, text: str):
        low = text.lower().strip()

        m = re.search(r"(show|list)\s+(first\s+|top\s+)?(\d+)\s+(rows|records)\s+(from\s+)?(\w+)", low)
        if m:
            n = int(m.group(3))
            t = m.group(6)
            for table in self.schema.keys():
                if table.lower() == t.lower() or table.lower().startswith(t.lower()):
                    return f"SELECT * FROM {table} LIMIT {n}", f"Heuristic: first {n} rows from {table}"

        if re.search(r"show all rows from (\w+)", low):
            m2 = re.search(r"show all rows from (\w+)", low)
            t = m2.group(1)
            for table in self.schema.keys():
                if table.lower() == t.lower() or table.lower().startswith(t.lower()):
                    return f"SELECT * FROM {table} LIMIT 500", f"Heuristic: select * from {table}"

        m3 = re.search(r"tracks longer than (\d+)", low)
        if m3:
            minutes = int(m3.group(1))
            for cand in ["tracks", "track", "Tracks", "Track"]:
                if cand in self.schema:
                    return f"SELECT * FROM {cand} WHERE milliseconds > {minutes*60000} LIMIT 500", f"Heuristic: tracks longer than {minutes} minutes"

        if "customer" in low and "country" in low:
            if "Customers" in self.schema:
                return "SELECT Country AS Country, COUNT(*) AS CustomerCount FROM Customers GROUP BY Country ORDER BY CustomerCount DESC LIMIT 500", "Heuristic: customers by country"

        if "invoice count" in low or ("number of invoices" in low):
            if "Invoices" in self.schema:
                return "SELECT strftime('%Y', InvoiceDate) AS Year, COUNT(*) AS InvoiceCount FROM Invoices GROUP BY Year ORDER BY Year LIMIT 500", "Heuristic: invoice count by year"

        return None, "No heuristic match"

    # Gemini / LLM call (robust, handles any query, no single-SELECT restriction)
    def _gemini_generate_sql_once(self, user_text: str):
        schema_text_lines = []
        for table, meta in self.schema.items():
            cols = [f"{c.get('name')}" + (f" ({c.get('type')})" if c.get("type") else "") for c in meta["columns"]]
            schema_text_lines.append(f"{table}: {', '.join(cols)}")
        schema_text = "\n".join(schema_text_lines)

        prompt = f"""You are a SQL generator for SQLite/MySQL/Postgres.
Given the schema below, produce a valid SQL statement for the user's request.
Schema:
{schema_text}

User request:
{user_text}

Return only SQL, no explanation, no backticks. Use table and column names exactly as presented."""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            sql = None
            if resp and getattr(resp, "text", None):
                sql = resp.text.strip()
                sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
            return sql
        except Exception:
            return None

    # Validate SQL (less strict now; allows complex SELECTs)
    def _validate_sql(self, sql: str) -> bool:
        if not sql or not isinstance(sql, str):
            return False
        try:
            # Only block dangerous commands (DROP, DELETE, UPDATE, etc.)
            forbidden = ["drop", "delete", "update", "alter", "insert", "truncate"]
            if any(f in sql.lower() for f in forbidden):
                return False
            return True
        except Exception:
            return False

    # Main NL->SQL interface
    def to_sql(self, user_query: str):
        sql, rationale = self._heuristic_generate_sql(user_query)
        if sql and self._validate_sql(sql):
            return sql, rationale, "heuristic"

        if not self.use_llm:
            return None, f"Heuristics failed and LLM disabled. {rationale}", "none"

        for attempt in range(2):
            sql_generated = self._gemini_generate_sql_once(user_query)
            if sql_generated and self._validate_sql(sql_generated):
                return sql_generated, "Generated by Gemini (attempt %d)" % (attempt + 1), "gemini"

        return None, "Gemini failed to generate a valid SQL statement. Try rephrasing.", "none"

# -------------------------
# SQLExecutorAgent (unchanged)
# -------------------------
class SQLExecutorAgent:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, sql: str):
        if not sql:
            raise ValueError("No SQL was provided.")
        if not self._validate_safe_sql(sql):
            raise ValueError("Query rejected as unsafe.")
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df, 0.0

    def _validate_safe_sql(self, sql: str):
        # Block only destructive statements
        forbidden = ["drop", "delete", "update", "alter", "insert", "truncate"]
        if any(f in sql.lower() for f in forbidden):
            return False
        return True

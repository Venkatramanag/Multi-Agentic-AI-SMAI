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
# SchemaAgent (unchanged logic; can accept schema or introspect)
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
# NL2SQLAgent: heuristics first, then Gemini (robust)
# - Always tries heuristics first (freeform/simple).
# - If heuristics fails, always call Gemini (complex fallback).
# - Validate results; retry Gemini once if initial output invalid.
# - Returns (sql_or_None, rationale_text, agent_used)
# -------------------------
class NL2SQLAgent:
    def __init__(self, schema: dict, use_llm: bool = True, gemini_api_key: str = None):
        self.schema = schema
        self.use_llm = use_llm
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        if self.use_llm and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

        # Build column->table map for heuristics (keeps your original heuristic style)
        self.col_to_table = {}
        for t, meta in self.schema.items():
            for c in meta["columns"]:
                self.col_to_table[c["name"].lower()] = t

    # Heuristic rules (kept minimal and non-destructive)
    def _heuristic_generate_sql(self, text: str):
        low = text.lower().strip()

        # Simple: show first N / show all rows
        m = re.search(r"(show|list)\s+(first\s+|top\s+)?(\d+)\s+(rows|records)\s+(from\s+)?(\w+)", low)
        if m:
            n = int(m.group(3))
            t = m.group(6)
            # try match table ignoring case
            for table in self.schema.keys():
                if table.lower() == t.lower() or table.lower().startswith(t.lower()):
                    return f"SELECT * FROM {table} LIMIT {n}", f"Heuristic: first {n} rows from {table}"

        if re.search(r"show all rows from (\w+)", low):
            m2 = re.search(r"show all rows from (\w+)", low)
            t = m2.group(1)
            for table in self.schema.keys():
                if table.lower() == t.lower() or table.lower().startswith(t.lower()):
                    return f"SELECT * FROM {table} LIMIT 500", f"Heuristic: select * from {table}"

        # specific heuristic examples you had: tracks longer than X minutes
        m3 = re.search(r"tracks longer than (\d+)", low)
        if m3:
            minutes = int(m3.group(1))
            # trust user's Chinook schema: Track table maybe 'Track' or 'tracks'
            for cand in ["tracks", "track", "Tracks", "Track"]:
                if cand in self.schema:
                    return f"SELECT * FROM {cand} WHERE milliseconds > {minutes*60000} LIMIT 500", f"Heuristic: tracks longer than {minutes} minutes"

        # customer by country, if Customers present
        if "customer" in low and "country" in low:
            if "Customers" in self.schema:
                return "SELECT Country AS Country, COUNT(*) AS CustomerCount FROM Customers GROUP BY Country ORDER BY CustomerCount DESC LIMIT 500", "Heuristic: customers by country"

        # invoice count by year
        if "invoice count" in low or ("number of invoices" in low):
            if "Invoices" in self.schema:
                return "SELECT strftime('%Y', InvoiceDate) AS Year, COUNT(*) AS InvoiceCount FROM Invoices GROUP BY Year ORDER BY Year LIMIT 500", "Heuristic: invoice count by year"

        # fallback not matched
        return None, "No heuristic match"

    # Use Gemini / LLM to generate SQL (robust handling)
    def _gemini_generate_sql_once(self, user_text: str):
        # Prepare schema string with columns + types to help the model
        schema_text_lines = []
        for table, meta in self.schema.items():
            cols = [f"{c.get('name')}" + (f" ({c.get('type')})" if c.get("type") else "") for c in meta["columns"]]
            schema_text_lines.append(f"{table}: {', '.join(cols)}")
        schema_text = "\n".join(schema_text_lines)

        prompt = f"""You are a SQL generator for SQLite/MySQL/Postgres. Given the schema below, produce a single valid SELECT statement only.
Schema:
{schema_text}

User request:
{user_text}

Return only the SQL SELECT statement, no explanation, no backticks. Use table and column names exactly as presented."""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            # gemini client returns content in resp.text (string)
            sql = None
            if resp and getattr(resp, "text", None):
                sql = resp.text.strip()
                # remove surrounding backticks / code fences if present
                sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
            return sql
        except Exception as e:
            # return None on any error
            return None

    def _validate_sql(self, sql: str) -> bool:
        if not sql or not isinstance(sql, str):
            return False
        # basic check: starts with SELECT and passes db.is_safe_sql
        if not sql.strip().lower().startswith("select"):
            return False
        try:
            return is_safe_sql(sql)
        except Exception:
            return False

    def to_sql(self, user_query: str):
        """
        Returns tuple: (sql_or_none, rationale_text, agent_used)
        agent_used is 'heuristic' or 'gemini' or 'none'
        """
        # 1. Heuristics-first
        sql, rationale = self._heuristic_generate_sql(user_query)
        if sql:
            # ensure SQL is safe according to db rules before returning
            if self._validate_sql(sql):
                return sql, rationale, "heuristic"
            else:
                # heuristic produced something unsafe — reject it and proceed to LLM
                # but keep rationale for logging
                heuristic_rationale = rationale
        else:
            heuristic_rationale = rationale

        # 2. If heuristics didn't match or produced unsafe SQL → use Gemini (if allowed)
        if not self.use_llm:
            # LLM not allowed: return explicit failure
            return None, f"Heuristics failed and LLM disabled. {heuristic_rationale}", "none"

        # try Gemini up to 2 attempts (retry once)
        for attempt in range(2):
            sql_generated = self._gemini_generate_sql_once(user_query)
            if sql_generated and self._validate_sql(sql_generated):
                return sql_generated, "Generated by Gemini (attempt %d)" % (attempt + 1), "gemini"
            # if returned something non-select-like, try again (retry)
        # 3. If still invalid
        return None, "Gemini failed to generate a valid single SELECT statement. Try rephrasing.", "none"


# -------------------------
# SQLExecutorAgent (unchanged but safe-checked)
# -------------------------
class SQLExecutorAgent:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, sql: str):
        if not sql:
            raise ValueError("No SQL was provided.")
        if not sql.lower().startswith("select"):
            raise ValueError("Only SELECT queries allowed. Query rejected as unsafe.")
        # rely on db.is_safe_sql already called earlier; still safe to check here
        if not is_safe_sql(sql):
            raise ValueError("Query rejected as unsafe by is_safe_sql.")

        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df, 0.0

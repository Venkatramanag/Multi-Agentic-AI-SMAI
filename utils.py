
import os
import sqlite3
from datetime import datetime
import pandas as pd


# NOTE: Keycloak integration removed temporarily.
# This module now exposes a small local auth stub so the app can continue
# to run while you troubleshoot Keycloak. Replace or extend this with
# the Keycloak logic later when ready.


def verify_user(username: str, password: str):
	"""
	Lightweight local auth stub.
	Returns a dict with keys: token, userinfo, roles — or None on failure.
	This does not perform real authentication. It's intended as a stop-gap
	while Keycloak is being fixed.
	"""
	if not username:
		return None

	# Basic role heuristics: adjust as needed
	uname = username.lower()
	if uname in ("admin", "administrator"):
		roles = ["admin"]
	elif "analyst" in uname:
		roles = ["analyst"]
	elif "manager" in uname:
		roles = ["manager"]
	else:
		roles = ["viewer"]

	userinfo = {
		"preferred_username": username,
		"realm_access": {"roles": roles}
	}

	return {"token": None, "userinfo": userinfo, "roles": roles}



def get_role(userinfo: dict) -> str:
	"""
	Extract primary role from Keycloak userinfo.
	Priority: admin > analyst > manager > viewer
	"""
	roles = userinfo.get("realm_access", {}).get("roles", [])
	if "admin" in roles:
		return "admin"
	elif "analyst" in roles:
		return "analyst"
	elif "manager" in roles:
		return "manager"
	return "viewer"



# ==============================
# 📜 Audit Logger
# ==============================
AUDIT_DB = ".data/audit.sqlite"
os.makedirs(".data", exist_ok=True)

def init_audit():
	"""Create audit DB if not exists."""
	conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
	cur = conn.cursor()
	cur.execute("""
		CREATE TABLE IF NOT EXISTS query_audit (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts TEXT,
			user TEXT,
			role TEXT,
			user_query TEXT,
			sql TEXT,
			status TEXT,
			rows INTEGER,
			duration REAL
		)
	""")
	conn.commit()
	conn.close()

def log_query(user: str, role: str, user_q: str, sql: str,
			  status: str, rows: int, duration: float):
	"""Insert query log into audit DB."""
	conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
	cur = conn.cursor()
	ts = datetime.utcnow().isoformat()
	cur.execute("""
		INSERT INTO query_audit (ts, user, role, user_query, sql, status, rows, duration)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	""", (ts, user, role, user_q, sql, status, rows, duration))
	conn.commit()
	conn.close()

def fetch_audit(limit: int = 200) -> pd.DataFrame:
	"""Fetch recent audit logs."""
	conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
	df = pd.read_sql_query(
		f"SELECT * FROM query_audit ORDER BY id DESC LIMIT {limit}", conn
	)
	conn.close()
	return df

# ==============================
# 🗂 Schema Formatter
# ==============================
class SchemaFormatter:
	"""Format DB schema for LLM prompt context."""

	def __init__(self, schema: dict):
		self.schema = schema

	def formatted(self) -> str:
		parts = []
		for table, meta in self.schema.items():
			parts.append(f"Table {table}:")
			for column in meta["columns"]:
				parts.append(f" - {column['name']}: {column.get('type')}")
			if meta.get("foreign_keys"):
				for fk in meta["foreign_keys"]:
					parts.append(
						f" FK: {fk.get('constrained_columns')} "
						f"-> {fk.get('referred_table')}({fk.get('referred_columns')})"
					)
			parts.append("")
		return "\n".join(parts)


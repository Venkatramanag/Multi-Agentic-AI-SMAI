# utils1.py (updated minimally)
import os
import sqlite3
import pandas as pd
from datetime import datetime
from keycloak import KeycloakOpenID
from keycloak.exceptions import KeycloakAuthenticationError

# ======================================================
# 📌 Audit DB Setup
# ======================================================
AUDIT_DB = ".data/audit.sqlite"
os.makedirs(".data", exist_ok=True)


# ======================================================
# 🔐 Keycloak Config (adjust to your setup)
# ======================================================
KEYCLOAK_SERVER_URL = "http://localhost:8080"
REALM_NAME = "master"         # change if not using master realm
CLIENT_ID = "MAI-Project"        # your Keycloak client
CLIENT_SECRET = None          # no secret needed for public client

keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_SERVER_URL,
    client_id=CLIENT_ID,
    realm_name=REALM_NAME,
    client_secret_key=CLIENT_SECRET
)


# ======================================================
# 🔐 Authentication Helpers
# ======================================================
def verify_user(username: str, password: str):
    """Verify user with Keycloak and return minimal userinfo dict."""
    try:
        token = keycloak_openid.token(username, password)
        if token.get("access_token"):
            return {
                "userinfo": {"preferred_username": username},
                "roles": []   # no roles for now
            }
        return None
    except KeycloakAuthenticationError:
        return None
    except Exception as e:
        print(f"Keycloak error: {e}")
        return None


def get_role(userinfo: dict) -> str:
    """Always return admin role (so all features are enabled)."""
    return "admin"


# ======================================================
# 📝 Audit Logging (unchanged DB schema)
# ======================================================
def init_audit():
    """Initialize the audit DB and table."""
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


def log_query(user: str, role: str, user_q: str, sql: str, status: str, rows: int, duration: float):
    """Insert a query log record into the audit DB."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO query_audit (ts, user, role, user_query, sql, status, rows, duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, user, role, user_q, sql, status, rows, duration)
    )
    conn.commit()
    conn.close()


def fetch_audit(limit: int = 200):
    """Fetch recent audit logs as a pandas DataFrame."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    df = pd.read_sql_query(f"SELECT * FROM query_audit ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df


def clear_audit():
    """Clear audit table (destructive — for demo/admin only)."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("DELETE FROM query_audit")
    conn.commit()
    conn.close()


def audit_to_csv_bytes(limit: int = 10000) -> bytes:
    """Return audit log CSV bytes for download."""
    df = fetch_audit(limit=limit)
    return df.to_csv(index=False).encode("utf-8")


# ======================================================
# 🏗 Schema Formatter (keeps your original, unchanged semantics)
# ======================================================
class SchemaFormatter:
    """Format DB schema for prompt injection into LLMs."""
    def __init__(self, schema: dict):
        self.schema = schema

    def formatted(self) -> str:
        parts = []
        for table, meta in self.schema.items():
            parts.append(f"📂 Table `{table}`:")
            # Columns
            for column in meta.get("columns", []):
                parts.append(f" - {column['name']} ({column.get('type')})")
            # Primary Key
            if meta.get("primary_key"):
                parts.append(f" 🔑 Primary Key: {meta['primary_key']}")
            # Foreign Keys
            if meta.get("foreign_keys"):
                for fk in meta["foreign_keys"]:
                    parts.append(
                        f" 🔗 FK {fk.get('constrained_columns')} → {fk.get('referred_table')}({fk.get('referred_columns')})"
                    )
            parts.append("")
        return "\n".join(parts)

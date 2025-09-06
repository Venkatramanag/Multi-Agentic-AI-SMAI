import streamlit as st
from db import create_engine_from_uri
from agents import SchemaAgent, QueryPlannerAgent, NL2SQLAgent, SQLExecutorAgent
from utils import verify_user, get_role, init_audit, log_query, fetch_audit, SchemaFormatter

# ==============================
# ⚙️ Streamlit Config
# ==============================
st.set_page_config(page_title="Multi-Agent Data Query System", layout="wide")
st.title("🚀 Intelligent Multi-Agent Data Query System")
st.write("Interact with databases using **natural language**. Secured with **Keycloak Authentication**.")

# Init audit DB
init_audit()

# ==============================
# 🔐 Authentication
# ==============================
if "user" not in st.session_state:
	st.sidebar.subheader("🔐 Login")
	username = st.sidebar.text_input("Username")
	password = st.sidebar.text_input("Password", type="password")

	if st.sidebar.button("Login"):
		auth_data = verify_user(username, password)
		if auth_data:
			st.session_state["user"] = auth_data["userinfo"]["preferred_username"]
			st.session_state["role"] = get_role(auth_data["userinfo"])
			st.session_state["roles"] = auth_data["roles"]
			# refresh the app so rest of UI sees the logged-in state
			if hasattr(st, "experimental_rerun"):
				st.experimental_rerun()
			else:
				# some Streamlit versions don't expose experimental_rerun; ask user to refresh
				st.success("✅ Login successful — please refresh the page to continue.")
				st.stop()
		else:
			st.error("❌ Invalid credentials")
			st.stop()

	# if we reach here the user is not logged in; stop further execution
	st.stop()

st.sidebar.success(f"Logged in as {st.session_state.get('user')} ({st.session_state.get('role')})")

# ==============================
# 📂 Database Connection
# ==============================
st.subheader("📂 Database Connection")
db_choice = st.text_input("Enter DB name or path", "chinook") # e.g., "chinook", "northwind.db"

if st.button("Connect to DB"):
	try:
		engine = create_engine_from_uri(db_choice)
		schema_agent = SchemaAgent(engine)
		st.session_state["engine"] = engine
		st.session_state["schema"] = schema_agent.schema
		st.session_state["schema_text"] = SchemaFormatter(schema_agent.schema).formatted()
		st.success("✅ Connected successfully!")
	except Exception as e:
		st.error(f"❌ Connection failed: {e}")

# ==============================
# 💡 Suggested Queries
# ==============================
if "schema" in st.session_state:
	st.subheader("💡 Suggested Queries")
	planner = QueryPlannerAgent()
	suggestions = planner.suggest(st.session_state["schema"])
	for q in suggestions:
		st.markdown(f"- {q}")

# ==============================
# 📝 NL-to-SQL Query (Analyst role)
# ==============================
if "schema" in st.session_state and st.session_state.get("role") in ["analyst", "admin"]:
	st.subheader("📝 Ask a Query")
	question = st.text_input("Type your natural language query")

	if st.button("Generate & Run SQL"):
		nl2sql_agent = NL2SQLAgent(st.session_state["schema"], use_llm=False)
		sql, rationale = nl2sql_agent.to_sql(question)

		st.code(sql, language="sql")
		st.caption(f"🤖 Reasoning: {rationale}")

		executor = SQLExecutorAgent(st.session_state["engine"])
		try:
			df, duration = executor.execute(sql)
			st.success(f"✅ Query executed in {duration:.3f} sec")
			st.dataframe(df)

			# Log query in audit DB
			log_query(
				user=st.session_state["user"],
				role=st.session_state["role"],
				user_q=question,
				sql=sql,
				status="success",
				rows=len(df),
				duration=duration
			)
			st.session_state["last_df"] = df
		except Exception as e:
			st.error(f"❌ Query failed: {e}")
			log_query(
				user=st.session_state["user"],
				role=st.session_state["role"],
				user_q=question,
				sql=sql,
				status="failed",
				rows=0,
				duration=0.0
			)

# ==============================
# 📊 Dashboard (Manager role)
# ==============================
if "schema" in st.session_state and st.session_state.get("role") in ["manager", "admin"]:
	st.subheader("📊 Dashboard View")
	if "last_df" in st.session_state:
		st.bar_chart(st.session_state["last_df"].select_dtypes(include=["int64", "float64"]))
	else:
		st.info("No data available yet. Run a query first.")

# ==============================
# 📜 Audit Log (Admin only)
# ==============================
if st.session_state.get("role") == "admin":
	st.subheader("📜 Query Audit Log")
	audit_df = fetch_audit(limit=100)
	st.dataframe(audit_df)


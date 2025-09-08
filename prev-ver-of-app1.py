# app1.py
import streamlit as st
import pandas as pd
import plotly.express as px
from db import create_engine_from_uri, get_schema, run_sql, is_safe_sql
from agents1 import SchemaAgent, QueryPlannerAgent, NL2SQLAgent, SQLExecutorAgent
from utils1 import (
    verify_user,
    init_audit,
    log_query,
    fetch_audit,
    clear_audit,
    audit_to_csv_bytes,
    SchemaFormatter,
)

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Multi-Agent Data Query System", layout="wide")

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .big-title {font-size: 32px; font-weight: 700; color: #1F77B4; text-align:center; animation: fadeIn 1s;}
    .muted {color: #6b6b6b;}
    .sidebar .stButton button {background-color:#1F77B4;color:white;border-radius:8px; transition: 0.3s;}
    .sidebar .stButton button:hover {background-color:#155a8a;}
    .st-expanderHeader {font-weight:600; color:#1F77B4;}
    @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🚀 Intelligent Multi-Agent Data Query System</div>', unsafe_allow_html=True)
st.write("Interact with databases using **natural language** — heuristics first, LLM fallback. Secure and auditable.")

# Initialize audit DB
init_audit()

# -------------------------
# Sidebar: Login, DB, Audit
# -------------------------
with st.sidebar:
    st.header("🔐 Login")
    if "user" not in st.session_state:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            auth = verify_user(username, password)
            if auth:
                st.session_state["user"] = auth["userinfo"]["preferred_username"]
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.stop()
    else:
        st.success(f"✅ Logged in as {st.session_state['user']}")

    st.markdown("---")
    st.header("📂 Database Connection")
    db_choice = st.text_input("DB name/path", "chinook")
    if st.button("Connect"):
        try:
            engine = create_engine_from_uri(db_choice)
            schema = get_schema(engine)
            schema_agent = SchemaAgent(engine, schema=schema)
            st.session_state["engine"] = engine
            st.session_state["schema"] = schema
            st.session_state["schema_text"] = SchemaFormatter(schema).formatted()
            st.success("✅ Connected successfully!")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    st.markdown("---")
    st.header("📜 Audit")
    if st.button("Clear history"):
        clear_audit()
        st.success("✅ Audit cleared")
    csv_bytes = audit_to_csv_bytes()
    st.download_button("Download audit (CSV)", data=csv_bytes, file_name="audit_log.csv", mime="text/csv")

# -------------------------
# Main layout: Left (controls) / Right (results)
# -------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("💡 Suggested Queries")
    if "schema" in st.session_state:
        planner = QueryPlannerAgent()
        for q in planner.suggest(st.session_state["schema"]):
            st.markdown(f"- {q}")

    st.subheader("📝 Ask a Query")
    question = st.text_area("Type your natural language query", height=120)

    gemini_key = st.text_input("Gemini API key (optional)", type="password")
    run_clicked = st.button("Generate & Run SQL")

    # Agent lifecycle expander
    lifecycle_exp = st.expander("🛠️ Agent Lifecycle (live)")
    lifecycle_placeholder = lifecycle_exp.empty()

with right_col:
    st.subheader("🔎 Schema (auto-detected)")
    if "schema_text" in st.session_state:
        st.code(st.session_state["schema_text"], language="markdown")
    else:
        st.info("Connect a database to see schema.")

    st.markdown("---")
    st.subheader("🧾 SQL & Results")
    sql_display = st.empty()
    result_display = st.empty()
    chart_display = st.empty()

# -------------------------
# Main action: Generate & Run
# -------------------------
if run_clicked:
    if "engine" not in st.session_state or "schema" not in st.session_state:
        st.error("Please connect a database first.")
    elif not question.strip():
        st.warning("Please type a natural language query.")
    else:
        engine = st.session_state["engine"]
        schema = st.session_state["schema"]

        # Agent setup
        nl2sql = NL2SQLAgent(schema=schema, use_llm=True, gemini_api_key=gemini_key or None)
        executor = SQLExecutorAgent(engine)

        # Lifecycle steps display
        steps = []
        def emit(step):
            steps.append(step)
            lifecycle_placeholder.write("\n".join(steps))

        emit("➡️ Received query: " + question)
        emit("🧭 Heuristic Agent: attempting to parse...")

        sql, rationale, agent_used = nl2sql.to_sql(question)

        if sql is None:
            emit(f"❌ {agent_used.upper()} failed: {rationale}")
            st.error("❌ No valid SQL could be generated. " + rationale)
            log_query(user=st.session_state["user"], role="basic", user_q=question,
                      sql="N/A", status="failed", rows=0, duration=0.0)
            sql_display.empty()
            result_display.empty()
            chart_display.empty()
        else:
            emit(f"✅ {agent_used.capitalize()} returned SQL.")
            sql_display.code(sql, language="sql")

            emit("▶️ Executing SQL...")
            try:
                df, duration = executor.execute(sql)
                emit(f"✅ Execution successful ({len(df)} rows, {duration:.3f}s)")
                result_display.dataframe(df, use_container_width=True)

                # Auto-chart for numeric columns
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                if not df.empty and num_cols:
                    non_num_cols = [c for c in df.columns if c not in num_cols]
                    if non_num_cols:
                        fig = px.bar(df, x=non_num_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {non_num_cols[0]}")
                    else:
                        fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
                    chart_display.plotly_chart(fig, use_container_width=True)
                else:
                    chart_display.info("No numeric columns to visualize.")

                # Persist audit & session
                log_query(user=st.session_state["user"], role="basic", user_q=question,
                          sql=sql, status="success", rows=len(df), duration=duration)
                st.session_state["last_df"] = df

            except Exception as e:
                emit(f"❌ Execution error: {e}")
                st.error(f"Execution failed: {e}")
                log_query(user=st.session_state["user"], role="basic", user_q=question,
                          sql=sql, status="failed", rows=0, duration=0.0)

# -------------------------
# Audit log (bottom)
# -------------------------
st.markdown("---")
st.subheader("📜 Audit Log (recent)")
audit_df = fetch_audit(limit=100)
st.dataframe(audit_df, use_container_width=True)

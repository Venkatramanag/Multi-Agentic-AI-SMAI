# app1.py
import streamlit as st
import pandas as pd
import plotly.express as px
from db import create_engine_from_uri, get_schema, run_sql, is_safe_sql
from agents1 import SchemaAgent, QueryPlannerAgent, NL2SQLAgent, SQLExecutorAgent
from utils1 import verify_user, init_audit, log_query, fetch_audit, clear_audit, audit_to_csv_bytes, SchemaFormatter

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Multi-Agent Data Query System", layout="wide")
st.markdown("<style> .big-title {font-size:30px;font-weight:700;} .muted {color: #6b6b6b;} </style>", unsafe_allow_html=True)
st.markdown('<div class="big-title">🚀 Intelligent Multi-Agent Data Query System</div>', unsafe_allow_html=True)
st.write("Interact with databases using natural language — heuristics first, LLM fallback. Secure and auditable.")

# Init audit
init_audit()

# -------------------------
# Sidebar: login, DB, audit controls
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
                st.error("Invalid credentials")
                st.stop()
        st.stop()
    else:
        st.success(f"Logged in as {st.session_state['user']}")

    st.markdown("---")
    st.header("📂 Database")
    db_choice = st.text_input("DB name/path", "chinook")
    if st.button("Connect"):
        try:
            engine = create_engine_from_uri(db_choice)
            schema = get_schema(engine)
            schema_agent = SchemaAgent(engine, schema=schema)
            st.session_state["engine"] = engine
            st.session_state["schema"] = schema
            st.session_state["schema_text"] = SchemaFormatter(schema).formatted()
            st.success("Connected to DB")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.markdown("---")
    st.header("📜 Audit")
    if st.button("Clear history"):
        clear_audit()
        st.success("Audit cleared")
    csv_bytes = audit_to_csv_bytes()
    st.download_button("Download audit (CSV)", data=csv_bytes, file_name="audit_log.csv", mime="text/csv")

# -------------------------
# Main layout: left (controls) / right (results)
# -------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("💡 Suggested queries")
    if "schema" in st.session_state:
        planner = QueryPlannerAgent()
        for q in planner.suggest(st.session_state["schema"]):
            st.markdown(f"- {q}")

    st.subheader("📝 Ask a query")
    question = st.text_area("Type your natural language query", height=120)

    gemini_key = st.text_input("Gemini API key (optional, leave blank to use server env)", type="password")

    run_clicked = st.button("Generate & Run SQL")

    # Agent lifecycle expander
    status_exp = st.expander("🛠️ Agent lifecycle (live)")
    status_placeholder = status_exp.empty()

with right:
    st.subheader("🔎 Schema (auto-detected)")
    if "schema_text" in st.session_state:
        st.code(st.session_state["schema_text"], language="markdown")
    else:
        st.info("Connect a database to see schema.")

    st.markdown("---")
    st.subheader("🧾 SQL & Results")
    sql_area = st.empty()
    result_area = st.empty()
    chart_area = st.empty()

# -------------------------
# Main action: run query
# -------------------------
if run_clicked:
    if "engine" not in st.session_state or "schema" not in st.session_state:
        st.error("Please connect to a database first.")
    elif not question or not question.strip():
        st.warning("Please type a natural language query.")
    else:
        # Prepare agent and show lifecycle
        engine = st.session_state["engine"]
        schema = st.session_state["schema"]
        # Create NL2SQLAgent with LLM allowed
        nl2sql = NL2SQLAgent(schema=schema, use_llm=True, gemini_api_key=gemini_key or None)
        executor = SQLExecutorAgent(engine)

        # Show progress steps
        steps = []
        def emit(step):
            steps.append(step)
            status_placeholder.write("\n".join(steps))

        emit("➡️ Received query: " + question)

        # Heuristics attempt
        emit("🧭 Heuristic Agent: attempting to parse...")
        sql, rationale, agent_used = nl2sql.to_sql(question)

        if sql is None:
            # failure path — show clear message, no graph, do not execute
            emit(f"❌ {agent_used.upper()} failed: {rationale}")
            st.error("❌ No valid SQL could be generated. " + rationale)
            # log failed attempt
            log_query(user=st.session_state.get("user", "unknown"),
                      role="basic",
                      user_q=question,
                      sql="N/A",
                      status="failed",
                      rows=0,
                      duration=0.0)
            sql_area.empty()
            result_area.empty()
            chart_area.empty()
        else:
            emit(f"✅ {agent_used.capitalize()} returned SQL.")
            # display SQL
            sql_area.code(sql, language="sql")
            # safety check before running
            if not is_safe_sql(sql):
                emit("⚠️ Safety check failed for SQL. Execution blocked.")
                st.error("SQL rejected by safety rules (only single SELECT allowed).")
                log_query(user=st.session_state.get("user", "unknown"),
                          role="basic",
                          user_q=question,
                          sql=sql,
                          status="failed",
                          rows=0,
                          duration=0.0)
            else:
                # execute
                emit("▶️ Executing SQL...")
                try:
                    df, duration = executor.execute(sql)
                    emit(f"✅ Execution successful ({len(df)} rows, {duration:.3f}s)")
                    # show results
                    result_area.dataframe(df, use_container_width=True)
                    # auto-visualize if numeric columns exist
                    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    if not df.empty and num_cols:
                        # simple default: bar chart of first numeric against first non-numeric (if exists)
                        x_col = None
                        y_col = None
                        # choose x: first non-numeric or index
                        non_num = [c for c in df.columns if c not in num_cols]
                        if non_num:
                            x_col = non_num[0]
                            y_col = num_cols[0]
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            chart_area.plotly_chart(fig, use_container_width=True)
                        else:
                            # only numeric columns -> show histogram of first numeric
                            fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
                            chart_area.plotly_chart(fig, use_container_width=True)
                    else:
                        chart_area.info("No numeric columns to auto-visualize.")
                    # persist audit
                    log_query(user=st.session_state.get("user", "unknown"),
                              role="basic",
                              user_q=question,
                              sql=sql,
                              status="success",
                              rows=len(df),
                              duration=duration)
                    # keep last df in session for dashboard
                    st.session_state["last_df"] = df
                except Exception as e:
                    emit(f"❌ Execution error: {e}")
                    st.error(f"Execution failed: {e}")
                    log_query(user=st.session_state.get("user", "unknown"),
                              role="basic",
                              user_q=question,
                              sql=sql,
                              status="failed",
                              rows=0,
                              duration=0.0)

# -------------------------
# Audit log display (bottom)
# -------------------------
st.markdown("---")
st.subheader("📜 Audit Log (recent)")
audit_df = fetch_audit(limit=100)
st.dataframe(audit_df, use_container_width=True)

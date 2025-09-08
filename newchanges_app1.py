import os
import sqlite3
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from agents1 import (
    SchemaAgent,
    QueryPlannerAgent,
    NL2SQLAgent,
    SQLExecutorAgent
)

from db import create_engine_from_uri, get_schema, run_sql, is_safe_sql
from utils1 import (
    verify_user,
    init_audit,
    log_query,
    fetch_audit,
    clear_audit,
    audit_to_csv_bytes,
    SchemaFormatter,
)

# ------------------------
# Page Config & Styling
# ------------------------
st.set_page_config(
    page_title="Multi-Agent SQL Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* General font */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px;
        padding: 0.5em 1em;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background-color: #0d6efd !important;
        color: white !important;
    }

    /* Tabs */
    .stTabs [role="tab"] {
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #0d6efd;
        color: white;
    }

    /* KPI metrics */
    .stMetric {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("⚙️ Settings")

db_path = st.sidebar.text_input("📂 Database Path", "chinook")
if st.sidebar.button("🔗 Connect DB"):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        st.session_state["db_path"] = db_path
        st.session_state["schema"] = QueryPlannerAgent().inspect_schema(conn)

        init_audit(conn)  # Initialize audit DB

        st.sidebar.success("✅ Connected")
    else:
        st.sidebar.error("❌ Database not found")

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Key")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    st.sidebar.success("✅ Key Set")

st.sidebar.markdown("---")
st.sidebar.info("📜 Use the tabs to navigate through features")

# ------------------------
# Tabs Layout
# ------------------------
tabs = st.tabs([
    "💡 Suggested Queries",
    "📝 Query Builder",
    "📈 Results & Charts",
    "📜 Audit Log"
])

# ------------------------
# Tab 1 - Suggested Queries
# ------------------------
with tabs[0]:
    st.subheader("💡 Suggested Queries")
    if "schema" in st.session_state:
        planner = QueryPlannerAgent()
        for q in planner.suggest(st.session_state["schema"]):
            st.markdown(f"🔹 {q}")

# ------------------------
# Tab 2 - Query Builder
# ------------------------
with tabs[1]:
    st.subheader("📝 Natural Language Query")
    question = st.text_area("Type your query here...", height=120, placeholder="E.g., Show all invoices above $50")

    if st.button("🚀 Generate & Run SQL"):
        if "db_path" not in st.session_state:
            st.error("⚠️ Connect to a database first")
        elif not api_key:
            st.error("⚠️ Provide your Gemini API Key")
        else:
            # Step 1: Generate SQL
            generator = NL2SQLAgent(api_key=api_key)
            schema = st.session_state["schema"]
            sql = generator.generate(question, schema)

            # Step 2: Execute SQL
            executor = SQLExecutorAgent()
            conn = sqlite3.connect(st.session_state["db_path"])
            start = time.time()
            df = executor.execute(conn, sql)
            duration = time.time() - start

            # Step 3: Audit
            log_query(question, sql)   # <-- using utils1

            # Save to session for visualization
            st.session_state["last_df"] = df
            st.session_state["last_duration"] = duration
            st.session_state["last_agent"] = "QueryGeneratorAgent"

            st.success(f"✅ Query executed in {duration:.3f}s")

# ------------------------
# Tab 3 - Results & Charts
# ------------------------
with tabs[2]:
    st.subheader("📈 Query Results")

    if "last_df" in st.session_state:
        df = st.session_state["last_df"]

        # KPI cards
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("📊 Rows Returned", len(df))
        kpi2.metric("⚡ Execution Time", f"{st.session_state.get('last_duration', 0):.3f}s")
        kpi3.metric("🤖 Agent Used", st.session_state.get("last_agent", "N/A"))
        st.markdown("---")

        # Show dataframe
        st.dataframe(df, use_container_width=True)

        # Automatic visualizations
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if num_cols and cat_cols:
            st.subheader("📊 Auto-Generated Visualizations")
            try:
                # Chart 1: Bar
                fig1 = px.bar(df, x=cat_cols[0], y=num_cols[0])
                st.plotly_chart(fig1, use_container_width=True)

                # Chart 2: Line
                if len(num_cols) > 1:
                    fig2 = px.line(df, x=cat_cols[0], y=num_cols[1])
                    st.plotly_chart(fig2, use_container_width=True)

                # Chart 3: Pie
                fig3 = px.pie(df, names=cat_cols[0], values=num_cols[0])
                st.plotly_chart(fig3, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Could not auto-generate charts: {e}")
        else:
            st.info("ℹ️ Not enough categorical/numeric columns for charts")

# ------------------------
# Tab 4 - Audit Log
# ------------------------
with tabs[3]:
    st.subheader("📜 Audit Log")

    audit_df = fetch_audit(limit=100)   # <-- using utils1
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True)

        # Download Button
        st.download_button(
            "⬇️ Download Audit Log (CSV)",
            data=audit_to_csv_bytes(audit_df),
            file_name="audit_log.csv",
            mime="text/csv"
        )

        # Clear Button
        if st.button("🧹 Clear Audit Log"):
            clear_audit()
            st.success("✅ Audit cleared!")
    else:
        st.info("ℹ️ No audit logs yet.")

import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- CONFIG ---
st.set_page_config(page_title="Semantic Cache Analytics", layout="wide", page_icon="❇️")
DB_NAME = "traffic_logs.db"


# --- DATA FETCHING ---
def load_data():
    try:
        conn = sqlite3.connect(DB_NAME)
        # We load everything into a DataFrame
        df = pd.read_sql_query(
            "SELECT * FROM request_logs ORDER BY timestamp DESC", conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()


# --- SIDEBAR ---

st.sidebar.title("Semantic Cache Analytics")
if st.sidebar.button("Refresh"):
    st.rerun()

df = load_data()

if not df.empty:
    # --- METRICS ROW ---
    total_reqs = len(df)
    cache_hits = len(df[df["response_source"] == "cache"])
    cache_misses = len(df[df["response_source"] == "llm"])
    hit_rate = (cache_hits / total_reqs) * 100 if total_reqs > 0 else 0

    # Calculate saved time (Assumes LLM avg is ~1.5s vs Cache ~0.02s)
    # Ideally, we sum the actual latency delta, but this is a good estimation
    saved_time = cache_hits * 1.5

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", total_reqs)
    col2.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    col3.metric("Time Saved", f"{saved_time:.1f}s")
    col4.metric("Avg Latency", f"{df['latency'].mean():.3f}s")

    # --- CHARTS ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Latency Distribution (Hit vs Miss)")
        fig_lat = px.box(
            df,
            x="response_source",
            y="latency",
            color="response_source",
            title="Latency Impact",
            points="all",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with c2:
        st.subheader("Traffic Volume")
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        traffic_chart = (
            df.set_index("timestamp").resample("1T").count()["id"]
        )  # 1 Minute buckets
        st.line_chart(traffic_chart)

    # --- RAW DATA TABLE ---
    st.subheader("Request Logs")
    # Show specific columns
    display_df = df[
        ["timestamp", "response_source", "latency", "prompt", "tokens_saved"]
    ]

    # Color code the source
    def highlight_source(val):
        color = "green" if val == "cache" else "red"
        return f"color: {color}; font-weight: bold"

    st.dataframe(
        display_df.style.map(highlight_source, subset=["response_source"]),
        use_container_width=True,
    )

else:
    st.warning("No traffic logs found. Run some curl requests to generate data!")
    st.code('curl -X POST "http://localhost:8000/chat" -d \'{"prompt": "Hello"}\'')

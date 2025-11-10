import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os


st.set_page_config(
    page_title="Meteor ML Analysis",
    layout="wide",
)

st.title("🛰️ Meteor ML Analysis Dashboard")
st.markdown("View clustering results and anomaly detection outputs from your trained models.")

# -------------------------------------------------
# Load dataset if present
# -------------------------------------------------
DATA_PATH = "meteor_data.csv"

if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
    st.success(f"Loaded dataset: **{len(df_raw)} rows**")
else:
    df_raw = pd.DataFrame()
    st.info("Place `meteor_data.csv` in the same folder as app.py")

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Dataset Preview",
    "Clustering Results",
    "Anomaly Detection"
])

# -------------------------------------------------
# TAB 1 — Dataset Preview
# -------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")

    if df_raw.empty:
        st.warning("No dataset found.")
    else:
        st.write(df_raw.head(10))

        st.markdown("### Basic Statistics")
        st.write(df_raw.describe())

# -------------------------------------------------
# TAB 2 — Clustering
# -------------------------------------------------
with tab2:
    st.subheader("KMeans / DBSCAN / Agglomerative Clustering")

    CLUSTER_PATH = "artifacts/clusters.parquet"

    if not os.path.exists(CLUSTER_PATH):
        st.warning("Clustering results not found. Expected: artifacts/clusters.parquet")
    else:
        df_clusters = pd.read_parquet(CLUSTER_PATH)

        st.success("Clustering results loaded.")
        st.write("### Sample of Clustered Data")
        st.dataframe(df_clusters.head())

        st.markdown("### KMeans Cluster Map")

        # Map visualization (requires reclat / reclong)
        fig = px.scatter_mapbox(
            df_clusters,
            lat="reclat",
            lon="reclong",
            color="cluster_kmeans",
            hover_name="id",
            zoom=1,
            height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 3 — Anomaly Detection
# -------------------------------------------------
with tab3:
    st.subheader("Anomaly Detection (Isolation Forest + Autoencoder)")

    ANOM_PATH = "artifacts/anomaly_scores.parquet"

    if not os.path.exists(ANOM_PATH):
        st.warning("Anomaly results missing. Expected: artifacts/anomaly_scores.parquet")
    else:
        df_anom = pd.read_parquet(ANOM_PATH)

        st.success("Anomaly scores loaded succesfully.")
        st.write("### Top Anomalies (Highest Reconstruction Error)")
        st.dataframe(
            df_anom.sort_values("ae_recon_error", ascending=False).head(20)
        )

        st.markdown("### Anomaly Scatter Plot")

        fig = px.scatter(
            df_anom,
            x="iso_anomaly_score",
            y="ae_recon_error",
            hover_data=["id", "year", "mass"],
            title="Isolation Forest vs Autoencoder Error"
        )
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
from functions import *

st.set_page_config(layout="wide")
def start_server():
    st.title("OSRS GE Report Dashboard")
    
    #-----------GENERAL REPORT OPTIONS----------------
    # General report options
    st.sidebar.header("General Report Options")
    report_type = st.sidebar.selectbox(
        "Select Report Type",
        options=["5-minute", "Hourly", "Daily"]
    )
    # Load report in advance so it's available in both tabs if needed
    report_df = pd.DataFrame()
    if st.sidebar.button("Generate General Report"):
        with st.spinner("Generating report..."):
            report_df = load_report(report_type=report_type)

    st.sidebar.markdown("---")

    #-----------SINGLE ITEM REPORT OPTIONS----------------
    st.sidebar.header("Single-Item Report Options")
    # Single Item Report Options
    report_item = st.sidebar.text_input(label="Enter Item Name")
    if st.sidebar.button("Generate Single-Item Report"):
        with st.spinner("Generating report..."):
            report_df = load_report(report_type=report_type)

    #-----------OTHER OPTIONS----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Update Data Cache"):
        with st.spinner("Caching time-series data..."):
            time_series_cache()

    
    #-----------TABS----------------
    tab1, tab2 = st.tabs(["📊 General Report", "📈 Single-Item Report"])

    with tab1:
        if not report_df.empty:
            st.subheader(f"Results for {report_type}")
            st.dataframe(report_df.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Select a report type and click Generate Report.")

    with tab2:
        if not report_df.empty:
            st.subheader(f"Graphs for {report_type}")

            # Example graph: ROI distribution
            st.bar_chart(report_df.set_index("name")["ROI"].head(10))

            # Example line chart: Prices over time (if you have a time series for 1 item)
            selected_item = st.selectbox("Select an item", report_df["name"].unique())
            item_df = report_df[report_df["name"] == selected_item]
            st.line_chart(item_df[["avgHighPrice", "avgLowPrice"]])
        else:
            st.info("Generate a report first to view graphs.")

if __name__ == "__main__":
    start_server()

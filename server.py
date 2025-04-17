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

    #-----------SHOCK REPORT OPTIONS----------------
    st.sidebar.header("Specific Report Options")
    # Single Item Report Options
    specific_report_type = st.sidebar.selectbox(
        "Select Report Type",
        options=["Shock-Report"]
    )
    specific_report_df = pd.DataFrame()
    if st.sidebar.button("Generate Specific Report"):
        with st.spinner("Generating report..."):
            specific_report_df = load_specific_report(report_type=specific_report_type)

    #-----------OTHER OPTIONS----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Update Data Cache"):
        with st.spinner("Caching time-series data..."):
            time_series_cache()

    
    #-----------TABS----------------
    tab1, tab2 = st.tabs(["📊 General Report", "📊 Specific Report"])

    with tab1:
        if not report_df.empty:
            st.subheader(f"Results for {report_type}")
            st.dataframe(report_df.reset_index(drop=True), hide_index=True)
        else:
            st.info("Select a report type and click Generate Report.")

    with tab2:
        if not specific_report_df.empty:
            st.subheader(f"Results for {specific_report_type}")
            st.dataframe(specific_report_df.reset_index(drop=True), hide_index=True)
        else:
            st.info("Select a report type and click Generate Report.")

if __name__ == "__main__":
    start_server()

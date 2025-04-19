import streamlit as st
import pandas as pd
from functions import *
import matplotlib.pyplot as plt
import config

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
    profit_weight = st.sidebar.number_input("Profit Weight", 
                                            value=config.general_profit_weight,
                                            max_value=1.0,
                                            min_value=0.0,
                                            step=0.01)
    sold_weight = st.sidebar.number_input("Percent_Sold Weight",
                                          value=config.general_sold_weight, 
                                            max_value=1.0,
                                            min_value=0.0,
                                            step=0.01)
    roi_weight = st.sidebar.number_input("ROI Weight", 
                                            value=config.general_roi_weight,
                                            max_value=1.0,
                                            min_value=0.0,
                                            step=0.01)
    vol_weight = st.sidebar.number_input("Volume Weight", 
                                            value=config.general_vol_weight,
                                            max_value=1.0,
                                            min_value=0.0,
                                            step=0.01)
    report_weights = [profit_weight,sold_weight,roi_weight,vol_weight]
    # Load report in advance so it's available in both tabs if needed
    report_df = pd.DataFrame()
    if st.sidebar.button("Generate General Report"):
        with st.spinner("Generating report..."):
            report_df = load_report(report_type=report_type,
                                    weights=report_weights)

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
    tab1, tab2, tab3 = st.tabs(["📊 General Report", "📊 Specific Report", "📈 Single-Item Report"])

    with tab1:
        if not report_df.empty:
            st.subheader(f"Results for {report_type}")
            # Create a row with 5 columns: 1 for the subheader, 4 for the number inputs

            st.dataframe(report_df.reset_index(drop=True), hide_index=True)
        else:
            st.info("Select a report type and click Generate Report.")

    with tab2:
        if not specific_report_df.empty:
            st.subheader(f"Results for {specific_report_type}")
            st.dataframe(specific_report_df.reset_index(drop=True), hide_index=True)
        else:
            st.info("Select a report type and click Generate Report.")
    with tab3:
        st.subheader("Sample Matplotlib Plot")
        
        # Example plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, y)
        ax.set_title("Sine Wave")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")

        st.pyplot(fig)
if __name__ == "__main__":
    start_server()

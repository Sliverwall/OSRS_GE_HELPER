import streamlit as st
import pandas as pd
from functions import *
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
    # Shock Report Options
    specific_report_type = st.sidebar.selectbox(
        "Select Report Type",
        options=["Shock-Report", "High-alch Report"]
    )
    specific_report_df = pd.DataFrame()

    if st.sidebar.button("Generate Specific Report"):
        with st.spinner("Generating report..."):
            specific_report_df = load_specific_report(report_type=specific_report_type)

    #-----------Single Item REPORT OPTIONS----------------
    st.sidebar.header("Single-Item Report Options")
    # Shock Report Options
    item_report_name = st.sidebar.text_input("Item Name") # Get name of item to run report on
    item_report_type = st.sidebar.selectbox(
        "Select Report Type",
        options=["5m", '1h', '24h']
    )
    item_report_df = pd.DataFrame()

    if st.sidebar.button("Generate Single-Item Report"):
        with st.spinner("Generating report..."):
            item_report_df = load_single_item_report(item_name=item_report_name,
                                                     report_type=item_report_type)
            with st.spinner("Modeling using selected time-series..."):
                # Model time-series for item then return a prediction df and a figure to plot
                single_item_pred_df, single_item_fig= model_single_item_report(item_report_df=item_report_df, item_report_type=item_report_type)



    #-----------OTHER OPTIONS----------------
    st.sidebar.markdown("---")
    if st.sidebar.button("Update Data Cache"):
        with st.spinner("Caching time-series data..."):
            time_series_cache()

    st.sidebar.markdown("----")
    st.sidebar.header("Profit Calc")
    buy_value = st.sidebar.text_input(label="Buy Value", value = "0")
    sell_value = st.sidebar.text_input(label="Sell Value", value = "0")
    vol = st.sidebar.text_input(label="Volume", value="1")
    if st.sidebar.button("Get Profit"):
        # Check if k is used
        if "k" in buy_value:
            buy_value = buy_value.replace('k', '')
        
        if "k" in sell_value:
            sell_value = sell_value.replace('k', '')

        buy_value = float(buy_value)
        sell_value = float(sell_value)
        vol = int(vol)
        margin, profit = get_profit(buy_value=buy_value, sell_value=sell_value, vol=vol)

        # Format display values in terms of k if they are above 1k

        if margin >= 1000:
            margin = f"{round(margin/1000,0)}k"
        if profit >= 1000:
            profit = f"{round(profit/1000,0)}k"

        st.sidebar.text_input(label="Margin Value", value=str(margin))
        st.sidebar.text_input(label="Profit Value", value=str(profit))
    
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
        st.subheader("Single Item Report")
        if not item_report_df.empty:
            st.subheader(f"Results for {item_report_name} at time step {item_report_type}")
            if single_item_fig:
                st.pyplot(single_item_fig)
                st.subheader("Prediticion Table")
                st.dataframe(single_item_pred_df.reset_index(drop=True), hide_index=True)
            st.subheader("Historical table")
            st.dataframe(item_report_df.reset_index(drop=True), hide_index=True)
if __name__ == "__main__":
    start_server()

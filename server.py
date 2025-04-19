import streamlit as st
import pandas as pd
from functions import *
import matplotlib.pyplot as plt
import config
import modules
import models.RuneLSTM
import models.RuneTrainer
import modules.rune_plots

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
        options=["Shock-Report"]
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
                # Choose input features
                input_features = ['avgHighPrice', 'avgLowPrice', 'percent_sold','highPriceVolume','lowPriceVolume','margin','ROI']
                # Choose target features
                target_indices = [0, 1]

                # Init model
                model = models.RuneLSTM.RuneLSTM(num_inputs=len(input_features),
                                                 hidden_size=32,
                                                 num_layers=2,
                                                 output_size=len(input_features))
                # Init Trainer
                trainer = models.RuneTrainer.RuneTrainer(model=model,
                                                         input_features=input_features,
                                                         num_epochs=100,
                                                         batch_size=32,
                                                         seq_length=10)
                # Create data loader using generated item report df
                data_loader, X, y = trainer.create_data_loader(data=item_report_df)

                # Use trainer to fit the model
                trainer.fit(data_loader=data_loader)

                ### Save model later
                # Use autogression to predict next set of points
                pred_y = trainer.autoregressive_pred(input_seq=X[-1], n_steps=10)
                
                # Get specific columns for plotting
                pred_selected = pred_y[:, target_indices]
                y_selected = y[:,target_indices]
                # Create figure using predicted values
                single_item_fig = modules.rune_plots.plot_price_prediction(y=y_selected, preds=pred_selected)


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
        st.subheader("Single Item Report")
        if not item_report_df.empty:
            st.subheader(f"Results for {item_report_name} at time step {item_report_type}")
            if single_item_fig:
                st.pyplot(single_item_fig)
            st.dataframe(item_report_df.reset_index(drop=True), hide_index=True)
if __name__ == "__main__":
    start_server()

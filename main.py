import streamlit as st
import pandas as pd
from utils import utils
def app():
    # Set up the dashboard title and sidebar options.
    st.title("OSRS GE Report Dashboard")
    st.sidebar.header("Report Options")
    
    # Option selector for choosing the type of report
    report_type = st.sidebar.selectbox(
        "Select Report Type",
        options=["5-minute", "Hourly", "Daily"]
    )
    
    # A button to trigger the report generation
    if st.sidebar.button("Generate Report"):
        with st.spinner("Generating report..."):
            report_df = utils.load_report(report_type)
        
        # Displaying the report if data is available
        if not report_df.empty:
            st.subheader(f"Results for {report_type}")
            st.table(report_df)  # You can also use st.dataframe(report_df) for interactive tables
        else:
            st.warning("No data available. Please check your query or database.")

    if st.sidebar.button("Query OSRS API"):
        with st.spinner("Querying results..."):
            report_df = pd.DataFrame()
        
        if not report_df.empty:
            st.dataframe(report_df)  # You can also use st.dataframe(report_df) for interactive tables
        else:
            st.warning("No data available. Please check your query or database.")

    st.write("This is a basic dashboard shell. Customize and expand it with your own scripts and queries.")

if __name__ == "__main__":
    app()

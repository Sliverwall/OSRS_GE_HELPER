import pandas as pd
import config
from modules import utils
import time
import numpy as np

def load_report(report_type: str):
    """
    Generate a report based on the selected report_type.
    """
    
    # Use switch statement to toggle between different reports
    match report_type:
        case ("5-minute"):
            data = f"{config.DATA_DIR}time_series_5m.csv"
        case ("Hourly"):
            data = f"{config.DATA_DIR}time_series_1h.csv"
        case ("Daily"):
            data = f"{config.DATA_DIR}time_series_24h.csv"

    # Construct df for report      
    time_series_df = pd.read_csv(data)

    # Get item_mappings
    item_mapping_df = pd.read_csv(f"{config.DATA_DIR}item_mapping.csv")

    # Only need the id, name and limit columns
    needed_item_cols = ["id", "name", "limit"]
    item_mapping_df = item_mapping_df[needed_item_cols]

    # Merge time_series df and item_mapping to get the report df
    report_df = pd.merge(left=time_series_df, right=item_mapping_df, on='id', how='left')

    # Construct a signal metrics
    report_df['signal'] = (report_df['margin'] * report_df['limit'] * (report_df['total_volume']/2) * report_df['percent_sold'])
    report_df['signal'] = report_df['signal'].apply(np.log10)

    report_df = report_df[["name", "formatted_timestamp", "avgHighPrice", "avgLowPrice", "total_volume", "percent_sold", "limit", "margin", "signal", "ROI"]]

    report_df = report_df.sort_values(by="margin", ascending=False)

    return report_df

def time_series_cache():
    '''
    cache routine to grab timeseries data from OSRS wiki API
    '''
    time_steps = ['5m', '1h', '24h']

    for time_step in time_steps:
        outputFile = f"{config.DATA_DIR}time_series_{time_step}.csv"
        url = f'https://prices.runescape.wiki/api/v1/osrs/{time_step}'

        r = utils.get_API_request(url=url, headers=config.HEADERS)

        dailyCSV = utils.extract_timeseries_request(r=r)

        dailyCSV.to_csv(outputFile, index=False)

        # Wait a moment between each api call
        time.sleep(1)

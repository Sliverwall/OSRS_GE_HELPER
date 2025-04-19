import pandas as pd
import config
from modules import utils
import time
import numpy as np

def load_report(report_type: str, weights:list=[
                                            0.94, # "profit" 0
                                            0.03,  # "sold" 1
                                            0.01,    # "roi" 2
                                            0.02,    # total_volume
                                            ], alpha:float = 1.02):
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

    # Use buy limit to get potential purchase from using 1 buy slot. Limit b Can it in k units
    report_df['potential_profit'] = (report_df['margin'] * np.minimum(report_df['limit'], report_df['minVol'])) / 1000

    # Use dip score to see how relatively low the current buy price is.
    report_df = get_dip_score(df=report_df, alpha=alpha)

    # Construct a signal metrics
    # Min-max normalize both features to range [0, 1]
    report_df['normalized_profit'] = (report_df['potential_profit'] - report_df['potential_profit'].min()) / (report_df['potential_profit'].max() - report_df['potential_profit'].min())
    report_df['normalized_sold'] = (report_df['percent_sold'] - report_df['percent_sold'].min()) / (report_df['percent_sold'].max() - report_df['percent_sold'].min())
    report_df['normalized_ROI'] = (report_df['ROI'] - report_df['ROI'].min()) / (report_df['ROI'].max() - report_df['ROI'].min())
    report_df['normalized_vol'] = (report_df['minVol'] - report_df['minVol'].min()) / (report_df['minVol'].max() - report_df['minVol'].min())

    # Now compute the signal metric
    report_df['signal'] = (weights[0] * report_df['normalized_profit']) + (weights[1] * report_df['normalized_sold']) + (weights[2] * report_df['normalized_ROI']) + (weights[3] * report_df['normalized_vol'])


    report_df = report_df[["name", "avgHighPrice", "avgLowPrice", "total_volume", "percent_sold", "limit", "margin", "signal", "ROI", 'dip_score', "potential_profit"]]

    # Filter by set conditions
    MIN_POTENTIAL_PROFIT = 100
    MIN_SIGNAL = 0.5
    MIN_ROI = 3

    # Create a boolean column based on the conditions
    report_df['good'] = (report_df['potential_profit'] >= MIN_POTENTIAL_PROFIT) & (report_df['signal'] >= MIN_SIGNAL) & (report_df['ROI'] >= MIN_ROI)

    # Apply sort by signal
    report_df = report_df.sort_values(by=["good", "signal"], ascending=[False, False])

    return report_df

def load_specific_report(report_type: str):
    match report_type:
        case ("Shock-Report"):
            return load_dip_report()


def get_dip_score(df: pd.DataFrame, alpha: float = 1.02, merge_id: str = 'id') -> pd.DataFrame:
    '''
    This individually pairs item ids to a dip score
    '''
    # Get needed tables
    time_reference_step = '1h'
    time_reference_csv = f"{config.DATA_DIR}time_series_{time_reference_step}.csv"
    latest_csv = f"{config.DATA_DIR}time_series_latest.csv"
    item_mapping_csv = f"{config.DATA_DIR}item_mapping.csv"

    # Load in each dataframe
    time_reference_csv = pd.read_csv(time_reference_csv)
    latest_df = pd.read_csv(latest_csv)
    item_mapping_df = pd.read_csv(item_mapping_csv)

    # Only need the id, name and limit columns
    needed_item_cols = ["id"]
    item_mapping_df = item_mapping_df[needed_item_cols]

    # Begin building the report df using 5m data as the base
    # Merge time_series df and item_mapping to get the report df
    dip_df = pd.merge(left=time_reference_csv, right=item_mapping_df, on='id', how='left')

    dip_df = pd.merge(left=dip_df, right=latest_df, on='id', how='left')

    # Consturct a dip signal'
    dip_df['dip_score'] = dip_df['avgLowPrice']/(dip_df['low'] * alpha)

    dip_df = dip_df[['id', 'dip_score']]

    df = pd.merge(left=df, right=dip_df, on=merge_id, how='left')

    return df

def load_dip_report() -> pd.DataFrame:
    '''
    The dip report attempts to detect price shocks. These shocks represent latest lows that do not reflect the general price of the item.
    Will aim to buy at the latest low and sell at the 5min daily low avg
    '''
    # Get needed tables
    daily_csv = f"{config.DATA_DIR}time_series_24h.csv"
    hourly_csv = f"{config.DATA_DIR}time_series_1h.csv"
    latest_csv = f"{config.DATA_DIR}time_series_latest.csv"
    item_mapping_csv = f"{config.DATA_DIR}item_mapping.csv"

    # Load in each dataframe
    daily_df = pd.read_csv(daily_csv)
    hourly_df = pd.read_csv(hourly_csv)
    latest_df = pd.read_csv(latest_csv)
    item_mapping_df = pd.read_csv(item_mapping_csv)

    # Only need the id, name and limit columns
    needed_item_cols = ["id", "name", "limit"]
    item_mapping_df = item_mapping_df[needed_item_cols]

    # Extract needed components from the hourly and daily dfs
    hourly_df['hourlyLowPrice'] = hourly_df['avgLowPrice']
    hourly_df['hourlyHighPrice'] =  hourly_df['avgHighPrice']
    hourly_cols = ['id', 'hourlyLowPrice', 'hourlyHighPrice']
    hourly_df = hourly_df[hourly_cols]

    # Begin building the report df using 5m data as the base
    # Merge time_series df and item_mapping to get the report df
    report_df = pd.merge(left=daily_df, right=item_mapping_df, on='id', how='left')

    report_df = pd.merge(left=report_df, right=latest_df, on='id', how='left')

    # Merge the hourly to get data on recent spikes
    report_df = pd.merge(left=report_df, right=hourly_df, on='id', how='left')

    # Since will be selling at avglow, need a min tax 
    TAX_LIMIT = 5000000
    TAX_RATE = 0.01 # 1% tax for items above 100
    MIN_TAX = 1
    # Tax rounds down to nearest int.
    report_df['low_tax'] = round((report_df["avgLowPrice"] * TAX_RATE).clip(upper=TAX_LIMIT), 0)

    report_df['low_tax'] = report_df['low_tax'].where(report_df['low_tax'] >= MIN_TAX, 0)

    # Calculate the profit_per_unit. Assume will buy at latest low and sell at avg low
    report_df['profit_per_unit'] = report_df['avgLowPrice'] - report_df['low'] - report_df['low_tax']

    # Calc an ROI for the dip specifically
    report_df['ROI'] = (report_df['profit_per_unit'] / report_df['low']) * 100

    # Use buy limit to get potential purchase from using 1 buy slot. Limit b Can it in k units
    report_df['potential_profit'] = (report_df['profit_per_unit'] * np.minimum(report_df['limit'], report_df['minVol'])) / 1000

    # Consturct a dip signal'
    alpha = 1.02
    report_df['good'] = report_df['hourlyLowPrice'] > (report_df['low'] * alpha)

    # Clean up 
    ideal_cols = ['name', 'avgHighPrice', 'avgLowPrice', 'low', 'limit', 'total_volume', 'potential_profit', 'ROI', 'good']

    report_df = report_df[ideal_cols]

    # Apply sort by signal
    report_df = report_df.sort_values(by=["good", "potential_profit"], ascending=[False, False])

    return report_df

def time_series_cache():
    '''
    cache routine to grab timeseries data from OSRS wiki API
    '''

    # Update time specific time-step series
    time_steps = ['5m', '1h', '24h']

    for time_step in time_steps:
        output_file = f"{config.DATA_DIR}time_series_{time_step}.csv"
        url = f'https://prices.runescape.wiki/api/v1/osrs/{time_step}'

        r = utils.get_API_request(url=url, headers=config.HEADERS)

        time_series_csv = utils.extract_timeseries_request(r=r)

        time_series_csv.to_csv(output_file, index=False)

        # Wait a moment between each api call
        time.sleep(1)

    # Update latest
    output_file = f"{config.DATA_DIR}time_series_latest.csv"
    url = 'https://prices.runescape.wiki/api/v1/osrs/latest'

    r = utils.get_API_request(url=url, headers=config.HEADERS)

    latest_csv = utils.extract_latest_timeseries_request(r=r)

    latest_csv.to_csv(output_file, index=False)


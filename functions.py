import pandas as pd
import config
from modules import utils
import time
import numpy as np
import models.RuneLSTM
import models.RuneTrainer
import modules.rune_plots
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
        case ("High-alch Report"):
              return load_high_alch_report()


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

    # Add an additional 10% of the high value if it is a bond
    report_df.loc[report_df['id'] == 13190, 'low_tax'] += report_df[report_df['id'] == 13190]["avgHighPrice"] * 0.1

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


def load_high_alch_report() -> pd.DataFrame:
    '''
    The high-alch report will take the instant buy price for each item, and nature rune then calculate profit per high-alch.
    '''
    # Get needed tables
    hourly_csv = f"{config.DATA_DIR}time_series_1h.csv"
    item_mapping_csv = f"{config.DATA_DIR}item_mapping.csv"

    # Load in each dataframe
    hourly_df = pd.read_csv(hourly_csv)
    item_mapping_df = pd.read_csv(item_mapping_csv)

    # Only need the id, name and limit columns
    needed_item_cols = ["id", "name", "limit", "lowalch", "highalch"]
    item_mapping_df = item_mapping_df[needed_item_cols]

    # Merge time_series df and item_mapping to get the report df
    report_df = pd.merge(left=hourly_df, right=item_mapping_df, on='id', how='left')

    # Get the high price for a nature rune (id=561)
    nature_rune_high_price = report_df[report_df['id'] == 561]["avgHighPrice"].iloc[0]

    # Calc a cost per alch
    report_df['cost_per_alch'] = (report_df['avgHighPrice'] + nature_rune_high_price)
    # Calc a profit per high-alch
    report_df['profit_per_high_alch'] = report_df['highalch'] - report_df['cost_per_alch']

    # Calc an ROI for each alch
    report_df['ROI'] = (report_df['profit_per_high_alch'] / report_df['cost_per_alch']) * 100

    # Use buy limit to get potential purchase from using 1 buy slot. Limit b Can it in k units
    report_df['potential_profit'] = (report_df['profit_per_high_alch'] * np.minimum(report_df['limit'], report_df['minVol'])) / 1000

    # Calc the profit per hour, using 1200 as the number of alchs per hour
    alch_per_hour = 1200
    report_df['profit_per_hour'] = (report_df['profit_per_high_alch'] * alch_per_hour)/1000
    # Clean up 
    ideal_cols = ['name', 'avgHighPrice', 'avgLowPrice', 'highalch','profit_per_high_alch', 'cost_per_alch', 'limit', 'total_volume', 'potential_profit', 'profit_per_hour', 'ROI']

    report_df = report_df[ideal_cols]

    # Apply sort by signal
    report_df = report_df.sort_values(by=["potential_profit"], ascending=[False])

    return report_df
def load_single_item_report(item_name:str, report_type:str) -> pd.DataFrame:
    '''
    Report to generarte single item time-series data
    '''
    # Load in item mappings to link item name to specific id
    item_mapping_df = pd.read_csv(f"{config.DATA_DIR}item_mapping.csv")

    # Get the item's mapping data before querying API
    item_data = item_mapping_df[item_mapping_df['name'] == item_name].iloc[0]
    itemId = item_data['id']

    # Query time series data for choosen item
    # Response query example: https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id=4151
    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={report_type}&id={itemId}"

    time_series_response = utils.get_API_request(url=url, headers=config.HEADERS)

    # Use extraction function to create an organzied pandas frame for the time-series response data
    time_series_df = utils.extract_single_item_data(r=time_series_response)

    # Add name column
    time_series_df['name'] = item_name

    # Organize columns
    col_names = ['id', 'name', 'timestamp', 'formatted_timestamp', 'avgHighPrice', 'avgLowPrice', 'highPriceVolume','lowPriceVolume','percent_sold','minVol','margin','ROI']
    time_series_df = time_series_df[col_names]
    # Return cleaned df
    return time_series_df


def model_single_item_report(item_report_df:pd.DataFrame, item_report_type:str):
                # Choose input features
                input_features = ['avgHighPrice', 'avgLowPrice', 'percent_sold']
                # Choose target features
                target_indices = [0,1]

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
                train_loader, val_loader, X, y = trainer.create_data_loader(data=item_report_df)

                # Use trainer to fit the model
                trainer.fit(train_loader=train_loader,
                            val_loader=val_loader)

                ### Save model later
                # Use autogression to predict next set of points
                n_steps = 10
                pred_y = trainer.autoregressive_pred(input_seq=X[-1], n_steps=n_steps)
                
                # Get specific columns for plotting
                pred_selected = pred_y[:, target_indices]
                y_selected = y[:,target_indices]

                # Create figure using predicted values
                single_item_fig = modules.rune_plots.plot_price_prediction(y=y_selected, preds=pred_selected)
                
                # Add time stamps to prediction output
                latest_stamp = item_report_df['timestamp'].max()
                # endpoints = [['5m/', 300, 2], ['1h/', 3600, 30], ['24h/', 86400, 365]]
                interval = {
                    "5m":300,
                    "1h":3600,
                    "24h":86400
                }
                # Init value array to track preds
                new_pred_y = []
                for i, pred in enumerate(pred_y):
                    # Init value to track new time-stamps
                    new_time = (interval[item_report_type]*(1+i)) + (latest_stamp)
                    # Append to existing pred array: ['avgHighPrice', 'avgLowPrice', 'percent_sold','highPriceVolume','lowPriceVolume']
                    new_pred_y.append([pred[0], pred[1], pred[2], new_time])

                new_pred_y = np.array(new_pred_y)
                # Create pred df for display
                prediction_features = ['avgHighPrice', 'avgLowPrice', "percent_sold", "timestamp"]
                single_item_pred_df = pd.DataFrame(new_pred_y, columns=prediction_features)
                # Format time
                single_item_pred_df['formatted_timestamp'] = pd.to_datetime(single_item_pred_df['timestamp'], unit='s')

                single_item_pred_df['formatted_timestamp']= single_item_pred_df['formatted_timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

                # Add the delta between the current low price and predicted high price
                single_item_pred_df['current_low'] = item_report_df[item_report_df['timestamp'] == item_report_df['timestamp'].max()]['avgLowPrice'].iloc[0]
                # Since will be selling at avglow, need a min tax 
                TAX_LIMIT = 5000000
                TAX_RATE = 0.01 # 1% tax for items above 100
                MIN_TAX = 1
                # Tax rounds down to nearest int.
                single_item_pred_df['tax'] = round((single_item_pred_df["avgHighPrice"] * TAX_RATE).clip(upper=TAX_LIMIT), 0)

                single_item_pred_df['tax'] = single_item_pred_df['tax'].where(single_item_pred_df['tax'] >= MIN_TAX, 0)

                single_item_pred_df['future-margin'] = (single_item_pred_df['avgHighPrice'] - single_item_pred_df['tax']) - single_item_pred_df['current_low']

                # single_item_pred_df = single_item_pred_df['avgHighPrice', 'avgLowPrice', 'future-margin', 'tax', 'formatted_timestamp']
                return single_item_pred_df, single_item_fig
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


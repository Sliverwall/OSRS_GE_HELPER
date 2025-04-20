import requests
import pandas as pd
import numpy as np

def get_API_request(url: str, headers: dict) -> requests.Response:
    '''
    Uses python's  built in request handler to get an HTTP request.
    For the wiki API, make sure to include a header that lets them know why and who is using the API.
    '''
    # Get get request for json 
    r = requests.get(url, headers=headers)
    # return quest output
    return r

def extract_timeseries_request(r: requests.Response) -> pd.DataFrame:
    '''
    Timeseries extraction takes an OSRS wiki API for timeseries data then returns a pandas dataframe
    The API key should be as follows: 'https://prices.runescape.wiki/api/v1/osrs/{time}'
    {time} can be 5m/1h/24h
    '''
    # Convert request into json format
    json_data = r.json()

    # Extract data
    time_stamp = json_data["timestamp"]
    response_data = json_data["data"]

    # Turn dict components to lists
    key_list = list(response_data.keys())
    value_list = list(response_data.values())

    # Make a list of lists where each entry is a example
    initial_cols = ['id','avgHighPrice','highPriceVolume','avgLowPrice','lowPriceVolume','timestamp']
    total_values = []
    total_keys_and_values = []
    for value in value_list:
        sub_values = list(value.values())
        sub_values.append(time_stamp)
        total_values.append(sub_values)
    for key1, value1 in zip(key_list, total_values):
        total_keys_and_values.append([key1] + value1)

    # Create a df from the extracted data
    df = pd.DataFrame(total_keys_and_values, columns=initial_cols)

    # Convert the Unix timestamp to datetime. By default, the Unix epoch is in UTC.
    df['formatted_timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df['formatted_timestamp']= df['formatted_timestamp'].dt.tz_localize('UTC')

    # ADD FEATURES
    # Add tax column
    TAX_LIMIT = 5000000
    TAX_RATE = 0.01 # 1% tax for items above 100
    MIN_TAX = 1
    # Tax rounds down to nearest int.
    df['tax'] = round((df["avgHighPrice"] * TAX_RATE).clip(upper=TAX_LIMIT), 0)

    df['tax'] = df['tax'].where(df['tax'] >= MIN_TAX, 0)

    # Add an additional 300k if it is a bond
    df.loc[df['id'] == 13190, 'tax'] += 300000

    # Add margin column. margin = (sell - tax) - buy
    df['margin'] = round((df["avgHighPrice"] - df["tax"]) - df["avgLowPrice"],0)

    #  Calculate Return on Investment as a percent. High margin item may be very expensive, so holds up more captial. The Margin/sell_price -> ROI
    df['ROI'] = round((df['margin']/df['avgLowPrice']) * 100, 3)
    
    # Calculate total Volume
    df['total_volume'] = df['highPriceVolume'] + df['lowPriceVolume']

    # Calculate percent sold: How much more it is listed at sell price then bought price
    df['percent_sold'] = (df['lowPriceVolume'] / df['total_volume']) * 100

    # Get relationship between the volume and spread. Use the buy volume
    df['margin-volume'] = df['lowPriceVolume'] * df['margin']

    # map out the bottle neck volume
    df['minVol'] = np.minimum(df['highPriceVolume'], df['lowPriceVolume'])

    # Reorganize the field positions
    ideal_cols = ['id','timestamp', 'formatted_timestamp', 'avgHighPrice','highPriceVolume','avgLowPrice','lowPriceVolume', 'total_volume', 'percent_sold', 'tax', 'margin', 'ROI', 'margin-volume', 'minVol']

    df = df[ideal_cols]
    return df

def extract_latest_timeseries_request(r: requests.Response) -> pd.DataFrame:
    '''
    Timeseries extraction takes an OSRS wiki API for the latest timeseries data then returns a pandas dataframe
    The API key should be as follows: 'https://prices.runescape.wiki/api/v1/osrs/latest'
    '''
    # Convert request into json format
    json_data = r.json()

    # Extract data
    response_data = json_data["data"]

    # Turn dict components to lists
    key_list = list(response_data.keys())
    value_list = list(response_data.values())

    # Make a list of lists where each entry is a example
    initial_cols = ['id','high','highTime','low','lowTime']
    total_values = []
    total_keys_and_values = []
    for value in value_list:
        sub_values = list(value.values())
        total_values.append(sub_values)
    for key1, value1 in zip(key_list, total_values):
        total_keys_and_values.append([key1] + value1)

    # Create a df from the extracted data
    df = pd.DataFrame(total_keys_and_values, columns=initial_cols)

    # Convert the Unix timestamp to datetime. By default, the Unix epoch is in UTC.
    df['highTime'] = pd.to_datetime(df['highTime'], unit='s').dt.tz_localize('UTC')
    df['lowTime'] = pd.to_datetime(df['lowTime'], unit='s').dt.tz_localize('UTC')

    return df


def extract_item_mapping(r: requests.Response) -> pd.DataFrame:
    '''
    extraction method to get item mapping data from osrs wiki api request
    The API key should be as follows: 'https://prices.runescape.wiki/api/v1/osrs/mapping'
    '''
    # Convert request into json format
    response_data = r.json()

    # Make a list of lists where each entry is a example
    initial_cols = ["Examine", "id", "members", "lowalch", "limit", "highalch", "icon", "name"]

    # Create a df from the extracted data
    df = pd.DataFrame(list(response_data), columns=initial_cols)

    # Replace unknown item limits with 999
    df['limit'] = df['limit'].fillna(999)

    # Organize column positions and remove icon and examine fields
    ideal_cols = ["id","name", "limit", "lowalch", "highalch", "members"]

    df = df[ideal_cols]

    return df

def extract_single_item_data(r: requests.Response) -> pd.DataFrame:
    '''
    Retrieves the time-series report for a single item.
    Can be used for EDA and predicitive modeling
    Response query example: https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id=4151
    '''

    json_data = r.json()

    time_series_data = json_data['data']
    itemID = json_data['itemId']

    # Make a list of lists where each entry is a example
    initial_cols = ["timestamp", "avgHighPrice", "avgLowPrice", "highPriceVolume", "lowPriceVolume"]

    # Create a df from the extracted data
    df = pd.DataFrame(list(time_series_data), columns=initial_cols)

    # Add the id to the df
    df['id'] = itemID

    # Add feature set
    # Convert the Unix timestamp to datetime. By default, the Unix epoch is in UTC.
    df['formatted_timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df['formatted_timestamp']= df['formatted_timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

    # ADD FEATURES
    # Add tax column
    TAX_LIMIT = 5000000
    TAX_RATE = 0.01 # 1% tax for items above 100
    MIN_TAX = 1
    # Tax rounds down to nearest int.
    df['tax'] = round((df["avgHighPrice"] * TAX_RATE).clip(upper=TAX_LIMIT), 0)

    df['tax'] = df['tax'].where(df['tax'] >= MIN_TAX, 0)

    # Add margin column. margin = (sell - tax) - buy
    df['margin'] = round((df["avgHighPrice"] - df["tax"]) - df["avgLowPrice"],0)

    #  Calculate Return on Investment as a percent. High margin item may be very expensive, so holds up more captial. The Margin/sell_price -> ROI
    df['ROI'] = round((df['margin']/df['avgLowPrice']) * 100, 3)
    
    # Calculate total Volume
    df['total_volume'] = df['highPriceVolume'] + df['lowPriceVolume']

    # Calculate percent sold: How much more it is listed at sell price then bought price
    df['percent_sold'] = (df['lowPriceVolume'] / df['total_volume']) * 100

    # Get relationship between the volume and spread. Use the buy volume
    df['margin-volume'] = df['lowPriceVolume'] * df['margin']

    # map out the bottle neck volume
    df['minVol'] = np.minimum(df['highPriceVolume'], df['lowPriceVolume'])

    return df







import requests
import json
import csv
import time
import os.path
import pandas as pd

def get_API_request(url, headers):
    '''
    Uses python's  built in request handler to get an HTTP request.
    For the wiki API, make sure to include a header that lets them know why and who is using the API.
    '''
    # Get get request for json 
    r = requests.get(url, headers=headers)
    # return quest output
    return r

def extract_timeseries_request(r):
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
    ideal_cols = ['id','timestamp', 'avgHighPrice','highPriceVolume','avgLowPrice','lowPriceVolume', 'Tax']
    total_values = []
    total_keys_and_values = []
    for value in value_list:
        sub_values = list(value.values())
        sub_values.append(time_stamp)
        total_values.append(sub_values)
    for key1, value1 in zip(key_list, total_values):
        total_keys_and_values.append([key1] + value1)

    # Create a df from the extracted data
    df = pd.DataFrame(total_keys_and_values, columns=initial_cols, index=False)

    # Interpolate missing values using a linear method
    df = df.interpolate(method='linear', limit_direction='both')

    # Now can add some features to make df make useful
    # Add tax column
    TAX_LIMIT = 5000000
    TAX_RATE = 0.01 # 1% tax for items above 100

    df['Tax'] = (df["avgHighPrice"] * TAX_RATE).clip(upper=TAX_LIMIT)

    # Add margin column. margin = (sell - tax) - buy
    df['Margin'] = (df["avgHighPrice"] - df["Tax"]) - df["avgLowPrice"]


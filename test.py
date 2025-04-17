from modules import utils
import config


# time_step = 'latest'
# outputFile = f"{config.DATA_DIR}time_series_{time_step}.csv"
# url = 'https://prices.runescape.wiki/api/v1/osrs/latest'

# r = utils.get_API_request(url=url, headers=config.HEADERS)

# dailyCSV = utils.extract_latest_timeseries_request(r=r)

# dailyCSV.to_csv(outputFile, index=False)


outputFile = f"{config.DATA_DIR}item_mapping.csv"

url = f"https://prices.runescape.wiki/api/v1/osrs/mapping"

r = utils.get_API_request(url=url, headers=config.HEADERS)

itemMapCSV = utils.extract_item_mapping(r=r)

itemMapCSV.to_csv(outputFile, index=False)

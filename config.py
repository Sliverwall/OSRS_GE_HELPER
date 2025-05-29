'''
Config file used to store constants
'''

OSRS_GE_DB = "data/ge.db"

DATA_DIR = "data/"

HEADERS = {
    'User-Agent': 'Machine Learning Practice',
    'From': 'sliverwall (discord ID)'  # discord ID
}


TAX_LIMIT = 5000000
TAX_RATE = 0.02 # 2% tax for items above 100

'''
Default weight configs
'''
general_profit_weight = 0.94
general_sold_weight = 0.03
general_roi_weight = 0.01
general_vol_weight = 0.02

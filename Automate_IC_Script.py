import numpy as np
import pandas as pd
import sys
from Modules import Data_Processing as dp
from Modules import Data as data
import Dispersion.DispersionAdjustedFunctionality as daf
from datetime import datetime, timedelta
from Modules.Utility import const, ceil_of_division
from Modules.enums import DB

expiry_type = 'I'
end_date = datetime.now() - timedelta(2)
end_date = pd.to_datetime(end_date)


def read_data():
    query_sw = '''
        SELECT *
        FROM strategy_wizard
    '''
    df = data.query_db(DB.GeneralOrQuantiPhi, query_sw)
    df = df.rename(columns = {'id' : 'sid', 'userid': 'uid'})
    return df

def parse_attributes(string):
    import re
    
    index_pattern = re.search(r'Index,TEXT,True,True,(\w+)', string)
    window_for_zscore_of_ic_pattern = re.search(r'Window for ZScore of IC,NUMERIC,True,True,(\d+)', string)
    window_for_zscore_of_std_pattern = re.search(r'Window for ZScore of Std,NUMERIC,True,True,(\d+)', string)
    window_for_std_of_ic_pattern = re.search(r'Window for Std,NUMERIC,True,True,(\d+)', string)
    timeframe_pattern = re.search(r'\|\|(\d+)\|', string)
    
    index = index_pattern.group(1) if index_pattern else None
    print(f"Index: {index}")

    window_for_zscore_of_ic = window_for_zscore_of_ic_pattern.group(1) if window_for_zscore_of_ic_pattern else None
    print(f"Window for ZScore of IC: {window_for_zscore_of_ic}")

    window_for_zscore_of_std = window_for_zscore_of_std_pattern.group(1) if window_for_zscore_of_std_pattern else None
    print(f"Window for ZScore of Std: {window_for_zscore_of_std}")

    window_for_std_of_ic = window_for_std_of_ic_pattern.group(1) if window_for_std_of_ic_pattern else None
    print(f"Window for Std: {window_for_std_of_ic}")

    timeframe = timeframe_pattern.group(1) if timeframe_pattern else None
    print(f"TimeFrame: {timeframe}")
    
    constituents = {}
    for i in range(1, 6):
        constituent_pattern = re.search(fr'Component{i},TEXT,True,True,(\w+)', string)
        constituent = constituent_pattern.group(1) if constituent_pattern else None
        weight_pattern = re.search(fr'Weight{i} \(\%\),NUMERIC,True,True,(\d+\.*\d*)', string)
        weight = weight_pattern.group(1) if weight_pattern else None
        print(f"Constituent: {constituent} | Weight: {weight}")
        constituents[constituent] = float(weight)
    
    return index, constituents, int(window_for_zscore_of_ic), int(window_for_zscore_of_std), int(window_for_std_of_ic), int(timeframe)

def get_ic_string(index_symbol, constituents, look_back_window, timeframe):
    n_points = 2 * look_back_window - 1
    n_days = ceil_of_division(n_points, (375/timeframe))
    start_date = dp.get_date_minus_n_days(pd.to_datetime(end_date), n_days, True).date()

    basket = daf.RawWeightedPortfolio()
    for symbol, weight in constituents.items():
        basket.insert(symbol, 1, weight)
    components = {}
    index = daf.ticker(index_symbol, 15, True, start_date, end_date, expiry_type, True, timeframe, True, 0.1)
    index.initializeDispersion(components, False, 1)
    for stock in basket.Symbols():
        components[stock] = daf.ticker(stock, basket.LotSize(stock), True, start_date, end_date, expiry_type, True, timeframe, False, 0.1)
        components[stock].initializeDispersion({}, True, basket.Weight(stock))

    for timestamp in index.timestamps:
        index.generate_ic(timestamp)

    ic = index.df_futures['ic'].round(4)

    final_string = ic.tail(look_back_window).astype(str).str.cat(sep='|')

    return final_string

def get_std_string(index_symbol, constituents, look_back_window_of_stds, number_of_points_for_std, timeframe):
    ic_string = get_ic_string(index_symbol, constituents, look_back_window_of_stds + number_of_points_for_std, timeframe)
    ic_data = pd.Series([float(x) for x in ic_string.split('|')])
    std_data = pd.Series(ic_data).rolling(window=number_of_points_for_std).std()
    std_data *= 100
    std_data = std_data.round(4)
    std_string = std_data.tail(look_back_window_of_stds).astype(str).str.cat(sep='|')
    return std_string


def write_data(sid, uid, key, value):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date = datetime.now().date().strftime('%Y-%m-%d')
    query_update = f'''
        UPDATE table1
        SET value = '{value}',
            time = '{date}',
            date_time = '{time}'
        WHERE uid = '{uid}'
        AND sid = '{sid}'
        AND key = '{key}';
    '''
    data.query_db(DB.GeneralOrQuantiPhi, query_update)
    return


def main():    
    df = read_data()

    for _, row in df.iterrows():
        index, constituents, window_for_zscore_of_ic, window_for_zscore_of_std, window_for_std_of_ic, timeframe = parse_attributes(row['sstrategyattributes'])
        ic_string_window_for_zscore_of_ic = get_ic_string(index, constituents, window_for_zscore_of_ic, timeframe)
        ic_string_window_for_std_of_ic = get_ic_string(index, constituents, window_for_std_of_ic, timeframe)
        std_string_window_for_std_of_ic = get_std_string(index, constituents, window_for_zscore_of_std, window_for_std_of_ic, timeframe)
        print()
        print()
        print()
        print("==============================================================================================================================================")
        print()
        print(f'UID: {row['uid']} | SID: {row['sid']} | Strategy Name: Hulk | Index: {index} | Window for ZScore of IC: {window_for_zscore_of_ic} | Window for ZScore of Std: {window_for_zscore_of_std} | Window for Std: {window_for_std_of_ic} | Timeframe: {timeframe}')
        print()
        print("IC History for ZScore of IC", ic_string_window_for_zscore_of_ic)
        print("IC History for STD of IC", ic_string_window_for_std_of_ic)
        print("STD History for ZScore of STD", std_string_window_for_std_of_ic)
        print()
        print("==============================================================================================================================================")
        print()
        print()
        print()
        write_data(row['sid'], row['uid'], 'implied_correlation_history_for_zscore_of_ic', ic_string_window_for_zscore_of_ic)
        write_data(row['sid'], row['uid'], 'implied_correlation_history_for_std_of_ic', ic_string_window_for_std_of_ic)
        write_data(row['sid'], row['uid'], 'std_history_for_zscore_of_std', std_string_window_for_std_of_ic)
    
    return    


if __name__ == "__main__":
    original_stdout = const(sys.stdout)

    logs_automation = open(f"Automation LOGS.txt", 'w', buffering=1)
    sys.stdout = logs_automation
    
    try:
        main()
    except Exception as error:
        print("Error Occured |", error)
    
    sys.stdout = original_stdout.value
    logs_automation.close()
        

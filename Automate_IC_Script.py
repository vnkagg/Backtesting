import numpy as np
import pandas as pd
import sys
from Modules import Data_Processing as dp
from Modules import Data as data
import Dispersion.DispersionAdjustedFunctionality as daf
from datetime import datetime, timedelta
from Modules.Utility import const, ceil_of_division
from Modules.enums import DB, Phase

expiry_type = 'I'
end_date = datetime.now() - timedelta(1)
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

    ics = dp.get_resampled_data(rf'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\Parallel_Generation\IC.csv', 1)

    basket = daf.RawWeightedPortfolio()
    for symbol, weight in constituents.items():
        basket.insert(symbol, 1, weight)

    # try:
    components = {}
    index = daf.ticker(index_symbol, 15, True, start_date, end_date, expiry_type, True, timeframe, True, 0.1)
    index.initializeDispersion(components, False, 1)
    for stock in basket.Symbols():
        components[stock] = daf.ticker(stock, basket.LotSize(stock), True, start_date, end_date, expiry_type, True, timeframe, False, 0.1)
        components[stock].initializeDispersion({}, True, basket.Weight(stock))
    # except Exception as e:
    #     print(f"Error in Initializing Data: {e}")
    if 'ic' not in index.df_futures.columns:
        index.df_futures['ic'] = pd.NA
    for timestamp in index.timestamps:
        if timestamp in ics.index:
            print(f"Timestamp: {timestamp} | IC Cache Hit!")
            print(f"Dirty Correlation(IC): {ics.loc[timestamp, 'ic']}")
            print("-------------------------------------------")
            index.df_futures.loc[timestamp, 'ic'] = ics.loc[timestamp, 'ic']
        else:
            ics.loc[timestamp, 'ic'] = index.get_ic_at(timestamp)
            
        if index.df_futures.loc[timestamp, 'ic'] == 0:
            index.df_futures.loc[timestamp, 'ic'] = pd.NA
        if ics.loc[timestamp, 'ic'] == 0:
            ics.loc[timestamp, 'ic'] = pd.NA

    ics = ics.sort_index()
    ics.to_csv(fr'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\Parallel_Generation\IC.csv')
    
    # ic = index.df_futures['ic'].round(4)
    # ic = pd.Series(ic)
    # ic = ic.round(4)
    # final_string = ic.tail(look_back_window).astype(str).str.cat(sep='|')
    ic = index.df_futures['ic']
    final_string = ic.tail(look_back_window).map(lambda x: f"{x:.4f}").str.cat(sep='|')
    return final_string

def get_std_string(index_symbol, constituents, look_back_window_of_stds, number_of_points_for_std, timeframe):
    ic_string = get_ic_string(index_symbol, constituents, look_back_window_of_stds + number_of_points_for_std, timeframe)
    ic_data = pd.Series([float(x) for x in ic_string.split('|')])
    std_data = pd.Series(ic_data).rolling(window=number_of_points_for_std).std()
    std_data *= 100
    # std_data = std_data.round(4)
    std_string = std_data.tail(look_back_window_of_stds).map(lambda x: f"{x:.4f}").str.cat(sep='|')
    return std_string


def write_data(sid, uid, sname, key, value):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date = datetime.now().date().strftime('%Y-%m-%d')
    query_update = f'''
        UPDATE table2
        SET value = '{value}',
            time = '{date}',
            date_time = '{time}'
        WHERE uid = '{uid}'
        AND sid = '{sid}'
        AND sname = '{sname}'
        AND key = '{key}';
    '''
    data.query_db(DB.GeneralOrQuantiPhi, query_update)
    return


def main():    
    df = read_data()
    number_of_rules = len(df)
    print("Number of Rules:", number_of_rules)
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"{i+1} | UID: M{row['uid']} | SID: {row['sid']}")
    print()
    print()
    for i, (_, row) in enumerate(df.iterrows()):
        print(f">> Rule {i+1}/{number_of_rules} | UID: M{row['uid']} | SID: {row['sid']}")
        print()
        print("Parsed Attributes: ")
        index, constituents, window_for_zscore_of_ic, window_for_zscore_of_std, window_for_std_of_ic, timeframe = parse_attributes(row['sstrategyattributes'])
        print("---------    (1/3) Computing IC String for Zscore of IC    ------------")
        ic_string_window_for_zscore_of_ic = get_ic_string(index, constituents, window_for_zscore_of_ic, timeframe)
        print("---------    (2/3) Computing IC String for Standard Deviation of IC    ------------")
        ic_string_window_for_std_of_ic = get_ic_string(index, constituents, window_for_std_of_ic, timeframe)
        print("---------    (3/3) Computing Standard Deviation String for ZScore of Standard Deviation    ------------")
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
        write_data(row['sid'], row['uid'], row['strategyname'], 'implied_correlation_history_for_zscore_of_ic', ic_string_window_for_zscore_of_ic)
        write_data(row['sid'], row['uid'], row['strategyname'],  'implied_correlation_history_for_std_of_ic', ic_string_window_for_std_of_ic)
        write_data(row['sid'], row['uid'], row['strategyname'], 'std_history_for_zscore_of_std', std_string_window_for_std_of_ic)
    
    if df.empty:
        from Modules import ParallelThreads
        ic_path = fr'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\Parallel_Generation\IC.csv'
        ic = pd.read_csv(ic_path, index_col=0, parse_dates=True)
        latest_date = dp.get_nearest_market_valid_day(end_date, Phase.Past)
        if pd.to_datetime(ic.index[-1].date()) == pd.to_datetime(latest_date.date()):
            print(f"IC Values Exists for the latest available QDAP Data: {pd.to_datetime(latest_date.date())}")
            return
        ic_generation_file = fr'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\generate_ic_for_period.py'
        runner = ParallelThreads.ThreadsForScript(ic_generation_file, 150)
        runner.run_script(end_date, end_date)
        ic_missing = fr'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\Parallel_Generation\{end_date.strftime('%Y_%m_%d_%B')}_{end_date.strftime('%Y_%m_%d_%B')}\IC_nearIV.csv'
        missing_ic_vals = pd.read_csv(ic_missing, index_col=0, parse_dates=True)
        ic = pd.concat([ic, missing_ic_vals])
        ic = ic.sort_index()
        ic.to_csv(ic_path)
    return    


if __name__ == "__main__":
    original_stdout = const(sys.stdout)

    logs_automation = open(fr"c:\Users\vinayak\Desktop\Automation_LOGS.txt", 'w', buffering=1)
    sys.stdout = logs_automation
    
    # try:
    main()
    # except Exception as error:
    #     print("Error Occured |", error)
    
    sys.stdout = original_stdout.value
    logs_automation.close()
        

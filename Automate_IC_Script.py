import numpy as np
import pandas as pd
import sys
import Data_Processing as dp
import Data as data
import Dispersion.DispersionAdjustedFunctionality as daf
from datetime import datetime, timedelta
from Utility import const, ceil_of_division
from enums import DB

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
    look_back_window_pattern = re.search(r'Look Back Window,NUMERIC,True,True,(\d+)', string)
    timeframe_pattern = re.search(r'\|\|(\d+)\|', string)
    
    index = index_pattern.group(1) if index_pattern else None
    print(f"Index: {index}")
    look_back_window = look_back_window_pattern.group(1) if look_back_window_pattern else None
    print(f"Look Back Window: {look_back_window}")
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
    
    return index, constituents, int(look_back_window), int(timeframe)

def get_ic_string(index, constituents, look_back_window, timeframe):
    n_points = 2 * look_back_window - 1
    n_days = ceil_of_division(n_points, (375/timeframe))
    start_date = dp.get_date_minus_n_days(pd.to_datetime(end_date), n_days, True).date()
    components = {}

    index = daf.ticker(index, 1, True, start_date, end_date, expiry_type, True, timeframe, False, 0.1, None)
    for constituent in constituents.keys():
        components[constituent] = daf.ticker(constituent, 1, True, start_date, end_date, expiry_type, True, timeframe, False, 0.1, None)

    index.initializeDispersion(True, components, False, 1, look_back_window)
    for component_symbol, component in index.components.items():
        component.initializeDispersion(False, {}, True, constituents[component_symbol]/100, look_back_window)


    index.generate_ic_data()

    ic = index.df_futures['ic'].round(4)

    final_string = ic.tail(look_back_window).astype(str).str.cat(sep='|')

    return final_string


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
        index, constituents, look_back_window, timeframe = parse_attributes(row['sstrategyattributes'])
        ic_string = get_ic_string(index, constituents, look_back_window, timeframe)
        print()
        print()
        print()
        print("==============================================================================================================================================")
        print()
        print(f'UID: {row['uid']} | SID: {row['sid']} | Strategy Name: Hulk | Index: {index} | Look Back Window: {look_back_window} | Timeframe: {timeframe}')
        print()
        print("IC History", ic_string)
        print()
        print("==============================================================================================================================================")
        print()
        print()
        print()
        write_data(row['sid'], row['uid'], 'implied_correlation_history', ic_string)
    
    return    


if __name__ == "__main__":
    original_stdout = const(sys.stdout)

    logs_automation = open(f"Automation LOGS.txt", 'w')
    sys.stdout = logs_automation
    
    try:
        main()
    except Exception as error:
        print("Error Occured |", error)
    
    sys.stdout = original_stdout.value
    logs_automation.close()
        

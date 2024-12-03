import os.path
import sqlite3

import psycopg2
import pandas as pd

from Modules.Data_Processing import get_market_valid_days
from Modules import Data_Processing

from Modules.enums import FNO, DB

host = "192.168.2.23"
port = 5432

user_qdap = "amt"
passwod_qdap = 'amt'
dbname_qdap = "qdap_test"

user_general = "vinayak"
passwod_general = "vinayak#1234"
dbname_general = "vinayak_database"

local_database = r"C:\Users\vinayak\Desktop\Backtesting\Database\database_try.db"

query_check_indexes_in_db = '''
    SELECT name, tbl_name, sql 
    FROM sqlite_master 
    WHERE type = 'index';
'''

query_ic = '''
    SELECT *
    FROM IC
'''

def make_connection_to_db(connect_to):
    if connect_to == DB.QDAP:
        conn = psycopg2.connect(host=host, port=port, user=user_qdap, password=passwod_qdap, dbname=dbname_qdap)
    elif connect_to == DB.LocalDB:
        conn = sqlite3.connect(local_database)
    elif connect_to == DB.GeneralOrQuantiPhi:
        conn = psycopg2.connect(host=host, port=port, user=user_general, password=passwod_general, dbname=dbname_general)

    cursor = conn.cursor()
    return cursor, conn





def query_db(db, query):
    cursor, conn = make_connection_to_db(db)
    df = None
    try:
        cursor.execute(query)
        if cursor.description:
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
        else:
            df = None
            if query.strip().split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                conn.commit()
    except Exception as error:
        print(f"Error Occured in Querying | {error}")
    cursor.close()
    conn.close()
    return df

'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+    FUNCTIONS FOR RAW FETCHING BEGINS     _+_+_+_+_+_+_+_+_+_+_+_+_+_+'

'========================================================================================================='








'---------------------------------------'
'             OPTIONS BEGINS            '
'---------------------------------------'
def fetch_options_data_on(symbol, expiry_type, date, db_type=DB.QDAP, t=0):
    if db_type == DB.LocalDB and t == 0:
        raise ValueError("Please Enter the timeframe of Data when fetching from local database")
    cursor, conn = make_connection_to_db(db_type)
    if db_type == DB.QDAP:
        query = f'''
            SELECT *
            FROM ohlcv_options_per_minute oopm
            WHERE oopm.symbol = '{symbol}'
            AND DATE(oopm.date_timestamp) = '{date}'
            AND oopm.expiry_type = '{expiry_type}'
            ORDER BY date_timestamp ASC;
                ''' 
    else:
        query = f'''
            SELECT *
            FROM ohlcv_options_per_{t}_minute oopm
            WHERE oopm.symbol = '{symbol}'
            AND oopm.date = '{date}'
            AND oopm.expiry_type = '{expiry_type}'
            ORDER BY date_timestamp ASC;
            '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    for ohlc in ["open", "high", "low", "close"]:
        df[ohlc] /= 100
    df['strike'] /= 100
    return df

def fetch_bhav_data_options_on(symbol, expiry_type, date, db_type=DB.QDAP):
    cursor, conn = make_connection_to_db(db_type)
    table_name = "bhav_copy_data_options"
    cursor.execute(
        f'''
                SELECT *
                FROM {table_name} bcdo
                WHERE bcdo.symbol = '{symbol}'
                AND bcdo.data_date = '{date}'
                AND bcdo.expiry_type = '{expiry_type}'
                ORDER BY data_date ASC;
            '''
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    for ohlc in ["open", "high", "low", "close"]:
        df[ohlc] /= 100
    df['strike'] /= 100
    return df
'---------------------------------------'
'             OPTIONS ENDS              '
'---------------------------------------'

'---------------------------------------'
'             FUTURE BEGINS             '
'---------------------------------------'
def fetch_futures_data_on(symbol, expiry_type, date, db_type=DB.QDAP, t=0):
    if db_type == DB.LocalDB and t == 0:
        raise ValueError("Please Enter the timeframe of Data when fetching from local database")
    cursor, conn = make_connection_to_db(db_type)
    if db_type == DB.QDAP:
        query = f'''
            SELECT *
            FROM ohlcv_future_per_minute ofpm
            WHERE ofpm.symbol = '{symbol}'
            AND DATE(ofpm.date_timestamp) = '{date}'
            AND ofpm.expiry_type = '{expiry_type}'
            ORDER BY date_timestamp ASC;
            '''
    else:
        query = f'''
            SELECT *
            FROM ohlcv_futures_per_{t}_minute ofpm
            WHERE ofpm.symbol = '{symbol}'
            AND ofpm.date = '{date}'
            AND ofpm.expiry_type = '{expiry_type}'
            ORDER BY date_timestamp ASC;
            '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    for ohlc in ["open", "high", "low", "close"]:
        df[ohlc] /= 100
    return df

def fetch_bhav_data_futures_on(symbol, expiry_type, date, db_type=DB.QDAP):
    cursor, conn = make_connection_to_db(db_type)
    table_name = "bhav_copy_data_future"
    if db_type == DB.LocalDB:
        table_name += 's'
    cursor.execute(
        f'''
                SELECT *
                FROM {table_name} bcdf
                WHERE bcdf.symbol = '{symbol}'
                AND bcdf.data_date = '{date}'
                AND bcdf.expiry_type = '{expiry_type}'
                ORDER BY data_date ASC;
            '''
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    for ohlc in ["open", "high", "low", "close"]:
        df[ohlc] /= 100
    return df
'---------------------------------------'
'             FUTURES ENDS              '
'---------------------------------------'




'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+ RAW FETCHING +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_'

'========================================================================================================='


















'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+ OPTIMIZED FETCHING +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_'

'========================================================================================================='




'---------------------------------------'
'             OPTIONS BEGINS            '
'---------------------------------------'
def get_options_data_on(symbol, expiry_type, date, t = 1):
    date = pd.to_datetime(date).date()
    try:
        print(f"Checking if Options Data exists for this day in Local DB for timeframe {t}mins")
        df = fetch_options_data_on(symbol, expiry_type, date, DB.LocalDB, t)
        if not df.empty:
            print(f"Data Found in Local DB for timeframe {t} mins")
            print("-----------------------------------------------------------------")
            return df
        else:
            print(f"Local DB does not contain this day's data for timeframe {t}mins")
            raise ValueError(f"empty data returned from local database for {t}min timeframe")
    except:
        if(t!=1):
            print(f"Checking if data exists for this day in Local DB for timeframe {1}mins")
            df = fetch_options_data_on(symbol, expiry_type, date, DB.LocalDB, 1)
            if df.empty:
                print(f"Local DB does not contain this day's data for timeframe {1}mins")
            else:
                print(f"Data Found in Local DB for timeframe 1min")


    if df.empty:
        print(f"Checking if data exists for this day in QDAP for timeframe {1}mins")
        df = fetch_options_data_on(symbol, expiry_type, date, DB.QDAP)
        if df.empty:
            print("Data for this day does not exist in QDAP!")
            print("-----------------------------------------------------------------")
            return    
    
        print(f"Data Found for this day in QDAP for timeframe {1}mins")

        df['date'] = pd.to_datetime(df['date_timestamp']).dt.date
        insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')
            
    if(t != 1):
        df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
        df_resampled['date'] = pd.to_datetime(df_resampled['date_timestamp']).dt.date
        insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
        print("-----------------------------------------------------------------")
        df = df_resampled
    return df


def get_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, t=1):
    print(f"Fetching {symbol}'s-{expiry_type} Options Data, Timeframe = {t}mins, start : {start_date}, end : {end_date}")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    for date in expected_date_range:
        print(date)
        df = get_options_data_on(symbol, expiry_type, date.date(), t)
        if df is not None:
            df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
            dfs.append(df)
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Options Data is empty between {start_date.strftime('%d/%b/%Y'), end_date.strftime('%d/%b/%Y')}")
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
    print("-----------------------------------------------------------------")
    return df

'============================== PER MINUTE ENDS ================================================'
'============================== BHAV DATA BEGINS ================================================'

def get_bhav_data_options_on(symbol, expiry_type, date):
    date = pd.to_datetime(date).date()
    try:
        print(f"Checking if Options Data exists for this day in Local DB")
        df = fetch_bhav_data_options_on(symbol, expiry_type, date, DB.LocalDB)
        if not df.empty:
            print(f"Data Found in Local DB")
            print("-----------------------------------------------------------------")
            df.rename(columns={'data_date': 'date_timestamp'}, inplace=True)
            return df
        else:
            print(f"Local DB does not contain this day's data")
            raise ValueError(f"empty data returned from local database")
    except Exception as e:
        df = pd.DataFrame()
        print(f"Data for this day does not exist in Local DB")
        # print(f"Error occured in fetching BHAV data from Local DB: {e}")


    if df.empty:
        print(f"Checking if data exists for this day in QDAP")
        df = fetch_bhav_data_options_on(symbol, expiry_type, date, DB.QDAP)
        if df.empty:
            print("Data for this day does not exist in QDAP!")
            print("-----------------------------------------------------------------")
            return    
        print(f"Data Found for this day in QDAP")
        insert_data_into_local_database(df, 'bhav_copy_data_options')
    print("-----------------------------------------------------------------")
    df.rename(columns={'data_date': 'date_timestamp'}, inplace=True)
    return df


def get_bhav_data_options_with_timestamps_between(symbol, expiry_type, start_date, end_date):
    print(f"Fetching {symbol}'s-{expiry_type} Options Data, start : {start_date}, end : {end_date}")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    for date in expected_date_range:
        print(date)
        df = get_bhav_data_options_on(symbol, expiry_type, date.date())
        if df is not None:
            df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
            dfs.append(df)
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Options EOD Data is empty between {start_date.strftime('%d/%b/%Y'), end_date.strftime('%d/%b/%Y')}")
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
    print("-----------------------------------------------------------------")
    return df



'---------------------------------------'
'             OPTIONS ENDS              '
'---------------------------------------'




'---------------------------------------'
'             FUTURE BEGINS             '
'---------------------------------------'
def get_futures_data_on(symbol, expiry_type, date, t = 1):
    date = pd.to_datetime(date).date()
    try:
        print(f"Checking if Futures Data exists for this day in Local DB for timeframe {t}min")
        df = fetch_futures_data_on(symbol, expiry_type, date, DB.LocalDB, t)
        if not df.empty:
            print(f"Data Found in Local DB for timeframe {t} min")
            print("-----------------------------------------------------------------")
            return df
        else:
            print(f"Local DB does not contain this day's data for timeframe {t}min")
            raise ValueError(f"empty data returned from local database for {t}min timeframe")
    except:
        if(t != 1):
            print(f"Checking if data exists for this day in Local DB for timeframe {1}min")
            df = fetch_futures_data_on(symbol, expiry_type, date, DB.LocalDB, 1)
            if df.empty:
                print(f"Local DB does not contain this day's data for timeframe {1}min")
            else:
                print(f"Data Found in Local DB for timeframe 1min")


    if df.empty:
        print(f"Checking if data exists for this day in QDAP for timeframe {1}min")
        df = fetch_futures_data_on(symbol, expiry_type, date, DB.QDAP)
        if df.empty:
            print("Data for this day does not exist in QDAP for timeframe 1min!")
            print("-----------------------------------------------------------------")
            return  
        print(f"Data for this day found in QDAP for timeframe {1}min")
        df['date'] = pd.to_datetime(df['date_timestamp']).dt.date
        insert_data_into_local_database(df, 'ohlcv_futures_per_1_minute')
        
    if(t != 1):
        df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
        df_resampled['date'] = pd.to_datetime(df_resampled['date_timestamp']).dt.date
        insert_data_into_local_database(df_resampled, f'ohlcv_futures_per_{t}_minute')
        df = df_resampled
    print("-----------------------------------------------------------------")
    return df


def get_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, t=1):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    print(f"Fetching {symbol}'s-{expiry_type} Futures Data, Timeframe = {t}mins, start : {start_date}, end : {end_date}")

    for date in expected_date_range:
        print("-----------------------------------------------------------------")
        print(date)
        df = get_futures_data_on(symbol, expiry_type, date.date(), t)
        if df is not None:
            df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
            dfs.append(df)
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Futures Data is empty between {start_date.strftime('%d/%b/%Y')}, {end_date.strftime('%d/%b/%Y')} (start and end inclusive)")
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
    print("-----------------------------------------------------------------")
    return df

'============================== PER MINUTE ENDS ================================================'
'============================== BHAV DATE BEGINS ================================================'

def get_bhav_data_futures_on(symbol, expiry_type, date):
    date = pd.to_datetime(date).date()
    try:
        print(f"Checking if BHAV Futures Data exists for this day in Local DB")
        df = fetch_bhav_data_futures_on(symbol, expiry_type, date, DB.LocalDB)
        if not df.empty:
            print(f"Data Found in Local DB")
            print("-----------------------------------------------------------------")
            df.rename(columns={'data_date': 'date_timestamp'}, inplace=True)
            return df
        else:
            print(f"Local DB does not contain this day's data")
            raise ValueError(f"empty data returned from local database")
    except Exception as e:
        df = pd.DataFrame()
        print("Data for this day does not exist in Local DB")
        # print(f"Error occured in fetching BHAV data from Local DB: {e}")


    if df.empty:
        print(f"Checking if data exists for this day in QDAP")
        df = fetch_bhav_data_futures_on(symbol, expiry_type, date, DB.QDAP)
        if df.empty:
            print("Data for this day does not exist in QDAP!")
            print("-----------------------------------------------------------------")
            return    
        
    print(f"Data Found for this day in QDAP")
    insert_data_into_local_database(df, 'bhav_copy_data_future')
            
    print("-----------------------------------------------------------------")
    df.rename(columns={'data_date': 'date_timestamp'}, inplace=True)
    return df


def get_bhav_data_futures_with_timestamps_between(symbol, expiry_type, start_date, end_date):
    print(f"Fetching {symbol}'s-{expiry_type} Futures BHAV Data, start : {start_date}, end : {end_date}")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    for date in expected_date_range:
        print(date)
        df = get_bhav_data_futures_on(symbol, expiry_type, date.date())
        if df is not None:
            df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
            dfs.append(df)
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Futures EOD Data is empty between {start_date.strftime('%d/%b/%Y'), end_date.strftime('%d/%b/%Y')}")
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
    print("-----------------------------------------------------------------")
    return df


def get_bhav_data_futures_for_dates(symbol, expiry_type, dates):
    print(f"Fetching {symbol}'s-{expiry_type} Futures BHAV Data with selective dates")
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)
    # expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    # expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    for date in dates:
        print(date)
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        df = get_bhav_data_futures_on(symbol, expiry_type, date.date())
        if df is not None:
            df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
            dfs.append(df)
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Futures EOD Data is empty for the given dates")
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
    print("-----------------------------------------------------------------")
    return df

'---------------------------------------'
'             FUTURES ENDS              '
'---------------------------------------'





'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_   OPTIMIZED FETCHING ENDS   +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_'

'========================================================================================================='


def save_df_in_folder_as(df, symbol, date, expiry_type, fno):
    prefix_folder_path = os.path.join('Database', fno.name)
    folder_path = os.path.join(prefix_folder_path, symbol)
    file_name = f'{expiry_type}_{date}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{file_name}.csv")
    df.to_csv(f"{file_path}", index=False)
    print(f"Saved dataframe to {file_path}")

def read_df_from_folder(symbol, date, expiry, fno, raiseError = False):
    prefix_folder_path = os.path.join('Database', fno.name)
    folder_path = os.path.join(prefix_folder_path, symbol)
    file_name = f'{expiry}_{date}'
    file_path = os.path.join(folder_path, f"{file_name}.csv")
    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        if raiseError:
            raise FileNotFoundError(f"No file found at {file_path}")
        return None
    print(f"File found at {file_path} >> Reading File...")
    df = pd.read_csv(f"{file_path}")
    print("File Read!")
    return df

def insert_data_into_local_database(df, table_name):
    conn = sqlite3.connect(local_database)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    print(f"Data successfully inserted into Local DB: {table_name} table.")






























    #
























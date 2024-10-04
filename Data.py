import os.path
import sqlite3

import psycopg2
import pandas as pd

import Data_Processing
from Data_Processing import get_market_valid_days

from enums import *
host = "192.168.2.23"
port = 5432
user = "amt"
dbname = "qdap_test"
local_database = r"C:\Users\vinayak\Desktop\Backtesting\database.db"

def make_connection_to_db(qdap):
    if qdap:
        conn = psycopg2.connect(host=host, port=port, user=user, dbname=dbname)
        cursor = conn.cursor()
    else:
        conn = sqlite3.connect(local_database)
        cursor = conn.cursor()
    return cursor, conn




'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+    FUNCTIONS FOR RAW FETCHING BEGINS     _+_+_+_+_+_+_+_+_+_+_+_+_+_+'

'========================================================================================================='







'---------------------------------------'
'             OPTIONS BEGINS            '
'---------------------------------------'

def fetch_options_data_on(symbol, expiry_type, date, qdap = True, t = 0):
    if not qdap and t == 0:
        raise "Please Enter the timeframe of Data when fetching from local database"
    cursor, conn = make_connection_to_db(qdap)
    table_name = "ohlcv_options_per_minute" if t == 0 else f"ohlcv_options_per_{t}_minute"
    cursor.execute(
        f'''
                SELECT *
                FROM {table_name} oopm
                WHERE oopm.symbol = '{symbol}'
                AND DATE(oopm.date_timestamp) = '{date}'
                AND oopm.expiry_type = '{expiry_type}'
                ORDER BY date_timestamp ASC;
            '''
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#
#
# def fetch_options_data(symbol, expiry, expiry_type, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s-{expiry_type} Options Data from {Database}, Timeframe = {t}mins, expiry: {expiry}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_options_per_minute" if t == 0 else f"ohlcv_options_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} oopm
#             WHERE oopm.symbol = '{symbol}'
#             AND DATE(oopm.expiry) = '{expiry}'
#             AND oopm.expiry_type = '{expiry_type}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'HAVING_EXPIRY_{expiry}', FNO.OPTIONS)
#     return df
#
# def fetch_options_data_on_expiry(symbol, expiry, expiry_type, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s 0-DTE {expiry_type}-Options Data from {Database}, Timeframe = {t}mins, expiry {expiry}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_options_per_minute" if t == 0 else f"ohlcv_options_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} oopm
#             WHERE oopm.symbol = '{symbol}'
#             AND DATE(oopm.date_timestamp) = '{expiry}'
#             AND DATE(oopm.expiry) = '{expiry}'
#             AND oopm.expiry_type = '{expiry_type}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'ON_{expiry}', FNO.OPTIONS)
#     return df
#
# def fetch_options_data_with_expiry_between(symbol, expiry_type, start_date, end_date, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s {expiry_type}-Options Data from {Database}, Timeframe = {t}mins, expiries between start date: {start_date} - end date: {end_date}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_options_per_minute" if t == 0 else f"ohlcv_options_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} oopm
#             WHERE oopm.symbol = '{symbol}'
#             AND oopm.expiry_type = '{expiry_type}'
#             AND DATE(oopm.expiry) <= '{end_date}'
#             AND DATE(oopm.expiry) >= '{start_date}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                               f'EXPIRY_BETWEEN_{start_date}_{end_date}', FNO.OPTIONS)
#     return df
#
# def fetch_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s {expiry_type}-Options Data from {Database}, Timeframe = {t}mins, timestamps between start date: {start_date} - end date: {end_date}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_options_per_minute" if t == 0 else f"ohlcv_options_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} oopm
#             WHERE oopm.symbol = '{symbol}'
#             AND oopm.expiry_type = '{expiry_type}'
#             AND DATE(oopm.date_timestamp) <= '{end_date}'
#             AND DATE(oopm.date_timestamp) >= '{start_date}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'TIMESTAMPS_BETWEEN_{start_date}_{end_date}', FNO.OPTIONS)
#     return df
#



'---------------------------------------'
'             OPTIONS ENDS              '
'---------------------------------------'









'---------------------------------------'
'             FUTURE BEGINS             '
'---------------------------------------'


def fetch_futures_data_on(symbol, expiry_type, date, qdap = True, t = 0):
    if not qdap and t == 0:
        raise "Please Enter the timeframe of Data when fetching from local database"
    cursor, conn = make_connection_to_db(qdap)
    table_name = "ohlcv_future_per_minute" if t == 0 else f"ohlcv_future_per_{t}_minute"
    if qdap:
        print("fetching from qdap")
    cursor.execute(
        f'''
                SELECT *
                FROM {table_name} ofpm
                WHERE ofpm.symbol = '{symbol}'
                AND DATE(ofpm.date_timestamp) = '{date}'
                AND ofpm.expiry_type = '{expiry_type}'
                ORDER BY date_timestamp ASC;
            '''
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#
# def fetch_futures_data(symbol, expiry, expiry_type, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s-{expiry_type} Futures Data from {Database}, Timeframe = {t}mins, expiry: {expiry}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_future_per_minute" if t == 0 else f"ohlcv_future_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} ofpm
#             WHERE ofpm.symbol = '{symbol}'
#             AND DATE(ofpm.expiry) = '{expiry}'
#             AND ofpm.expiry_type = '{expiry_type}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'HAVING_EXPIRY_{expiry}', FNO.FUTURES)
#     return df
#
#
# def fetch_futures_data_on_expiry(symbol, expiry, expiry_type, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s 0-DTE {expiry_type}-Futures Data from {Database}, Timeframe = {t}mins, expiry {expiry}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_future_per_minute" if t == 0 else f"ohlcv_future_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} ofpm
#             WHERE ofpm.symbol = '{symbol}'
#             AND DATE(ofpm.date_timestamp) = '{expiry}'
#             AND DATE(ofpm.expiry) = '{expiry}'
#             AND ofpm.expiry_type = '{expiry_type}'
#             ORDER BY date_timstamp ASC;
#             '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} options fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'ON_{expiry}', FNO.FUTURES)
#     return df
#
# # FUNCTION THAT RETRIEVES FUTURES DATA IN BETWEEN A PARTICULAR START DATE AND END DATE
# def fetch_futures_data_with_expiry_between(symbol, expiry_type, start_date, end_date, qdap = True, t=0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s {expiry_type}-Futures Data from {Database}, Timeframe = {t}mins, expiries between start date: {start_date} - end date: {end_date}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_future_per_minute" if t == 0 else f"ohlcv_future_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} ofpm
#             WHERE ofpm.symbol = '{symbol}'
#             AND ofpm.expiry_type = '{expiry_type}'
#             AND DATE(ofpm.expiry) >= '{start_date}'
#             AND DATE(ofpm.expiry) <= '{end_date}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} futures fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                               f'EXPIRY_BETWEEN_{start_date}_{end_date}', FNO.FUTURES)
#     return df
#
# def fetch_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, qdap = True, t = 0):
#     if not qdap and t == 0:
#         raise "Please Enter the timeframe of Data when fetching from local database"
#     Database = "QDAP" if qdap else "Local Database"
#     print(f"Fetching {symbol}'s {expiry_type}-Futures Data from {Database}, Timeframe = {t}mins, timestamps between start date: {start_date} - end date: {end_date}")
#     cursor, conn = make_connection_to_db(qdap)
#     table_name = "ohlcv_future_per_minute" if t == 0 else f"ohlcv_future_per_{t}_minute"
#     cursor.execute(
#         f'''
#             SELECT *
#             FROM {table_name} ofpm
#             WHERE ofpm.symbol = '{symbol}'
#             AND ofpm.expiry_type = '{expiry_type}'
#             AND DATE(ofpm.date_timestamp) >= '{start_date}'
#             AND DATE(ofpm.date_timestamp) <= '{end_date}'
#             ORDER BY date_timestamp ASC;
#         '''
#     )
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()
#     df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
#     if not df.empty:
#         print(f'{symbol} futures fetched')
#         save_df_in_folder_as(df, f'{symbol}_{expiry_type}',
#                          f'TIMESTAMPS_BETWEEN_{start_date}_{end_date}', FNO.FUTURES)
#     return df
#




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
        df = fetch_options_data_on(symbol, expiry_type, date, False, t)
        if not df.empty:
            # print("-----------------------------------------------------------------")
            return df
        else:
            raise f"empty data returned from local database for {t}min timeframe"
    except:
        df = fetch_options_data_on(symbol, expiry_type, date, False, 1)

    if df.empty:
        df = fetch_options_data_on(symbol, expiry_type, date, True)
        insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')

    df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
    insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
    print("-----------------------------------------------------------------")
    return df_resampled
#
# def get_options_data(symbol, expiry_type, expiry, t=1):
#     try:
#         df = fetch_options_data(symbol, expiry_type, expiry, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_options_data(symbol, expiry_type, expiry, False, 1)
#
#     if df.empty:
#         df = fetch_options_data(symbol, expiry_type, expiry, True)
#         insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
#     insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled
#
#
# def get_options_data_on_expiry(symbol, expiry_type, expiry, t=1):
#     try:
#         df = fetch_options_data_on_expiry(symbol, expiry_type, expiry, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_options_data_on_expiry(symbol, expiry_type, expiry, False, 1)
#
#     if df.empty:
#         df = fetch_options_data_on_expiry(symbol, expiry_type, expiry, True)
#         insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
#     insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled
#
#
# def get_options_data_with_expiry_between(symbol, expiry_type, start_date, end_date, t=1):
#     try:
#         df = fetch_options_data_with_expiry_between(symbol, expiry_type, start_date, end_date, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_options_data_with_expiry_between(symbol, expiry_type, start_date, end_date, False, 1)
#
#     if df.empty:
#         df = fetch_options_data_with_expiry_between(symbol, expiry_type, start_date, end_date, True)
#         insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
#     insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled

#
# def get_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, t=1):
#     # expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
#     # expected_date_range = get_market_valid_days(expected_date_range)
#     # missing_dates = expected_date_range
#     # try:
#     #     df = fetch_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, False, t)
#     #     df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
#     #     df_dates = df['date_timestamp'].dt.date
#     #     missing_dates = expected_date_range.difference(pd.to_datetime(df_dates))
#     #     if not missing_dates.empty:
#     #         print("-----------------------------------------------------------------")
#     #         return df
#     #     else:
#     #         raise f"empty data returned from local database for {t}min timeframe"
#     # except:
#     df = fetch_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, False, 1)
#
#     if df.empty:
#         df = fetch_options_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, True)
#         insert_data_into_local_database(df, 'ohlcv_options_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.OPTIONS)
#     # insert_data_into_local_database(df_resampled, f'ohlcv_options_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled

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
        df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
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
        df = fetch_futures_data_on(symbol, expiry_type, date, False, t)
        if not df.empty:
            return df
        else:
            raise f"empty data returned from local database for {t}min timeframe"
    except:
        print("-----------------------------------------------------------------")
        df = fetch_futures_data_on(symbol, expiry_type, date, False, 1)

    if df.empty:
        df = fetch_futures_data_on(symbol, expiry_type, date, True)
        insert_data_into_local_database(df, 'ohlcv_future_per_1_minute')

    df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
    insert_data_into_local_database(df_resampled, f'ohlcv_future_per_{t}_minute')
    print("-----------------------------------------------------------------")
    return df_resampled


# def get_futures_data(symbol, expiry_type, expiry, t=1):
#     try:
#         df = fetch_futures_data(symbol, expiry_type, expiry, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_futures_data(symbol, expiry_type, expiry, False, 1)
#
#     if df.empty:
#         df = fetch_futures_data(symbol, expiry_type, expiry, True)
#         insert_data_into_local_database(df, 'ohlcv_future_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
#     insert_data_into_local_database(df_resampled, f'ohlcv_future_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled
#
#
# def get_futures_data_on_expiry(symbol, expiry_type, expiry, t=1):
#     try:
#         df = fetch_futures_data_on_expiry(symbol, expiry_type, expiry, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_futures_data_on_expiry(symbol, expiry_type, expiry, False, 1)
#
#     if df.empty:
#         df = fetch_futures_data_on_expiry(symbol, expiry_type, expiry, True)
#         insert_data_into_local_database(df, 'ohlcv_future_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
#     insert_data_into_local_database(df_resampled, f'ohlcv_future_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled
#
#
# def get_futures_data_with_expiry_between(symbol, expiry_type, start_date, end_date, t=1):
#     try:
#         df = fetch_futures_data_with_expiry_between(symbol, expiry_type, start_date, end_date, False, t)
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_futures_data_with_expiry_between(symbol, expiry_type, start_date, end_date, False, 1)
#
#     if df.empty:
#         df = fetch_futures_data_with_expiry_between(symbol, expiry_type, start_date, end_date, True)
#         insert_data_into_local_database(df, 'ohlcv_future_per_1_minute')
#
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
#     insert_data_into_local_database(df_resampled, f'ohlcv_future_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled


# def get_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, t = 1):
#     try:
#         df = fetch_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, False, t)
#
#         if not df.empty:
#             print("-----------------------------------------------------------------")
#             return df
#         else:
#             raise f"empty data returned from local database for {t}min timeframe"
#     except:
#         df = fetch_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, False)
#     if df.empty:
#         df = fetch_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, True)
#         insert_data_into_local_database(df, 'ohlcv_future_per_1_minute')
#     df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
#     # insert_data_into_local_database(df_resampled, f'ohlcv_future_per_{t}_minute')
#     print("-----------------------------------------------------------------")
#     return df_resampled

def get_futures_data_with_timestamps_between(symbol, expiry_type, start_date, end_date, t=1):
    expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    expected_date_range = get_market_valid_days(expected_date_range)
    dfs = []
    print(f"Fetching {symbol}'s-{expiry_type} Futures Data, Timeframe = {t}mins, start : {start_date}, end : {end_date}")

    for date in expected_date_range:
        print("-----------------------------------------------------------------")
        print(date)
        df = get_futures_data_on(symbol, expiry_type, date.date(), t)
        df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # df_resampled = Data_Processing.resample_df_to_timeframe(df, t, FNO.FUTURES)
    print("-----------------------------------------------------------------")
    return df





'---------------------------------------'
'             FUTURES ENDS              '
'---------------------------------------'



'========================================================================================================='

'+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_   OPTIMIZED FETCHING ENDS   +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_'

'========================================================================================================='




def save_df_in_folder_as(df, folder_path, file_name, fno, to_excel = False):
    prefix_folder_path = os.path.join('Database', fno.name)
    folder_path = os.path.join(prefix_folder_path, folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if to_excel:
        file_path = os.path.join(folder_path, f"{file_name}.xlsx")
        df.to_excel(f"{file_path}", index=False)
    else:
        file_path = os.path.join(folder_path, f"{file_name}.csv")
        df.to_csv(f"{file_path}", index=False)
    print(f"Saved dataframe to {file_path}")

def read_df_from_folder(folder_path, file_name, fno, raiseError = False, from_excel = True):
    prefix_folder_path = os.path.join('Database', fno.name)
    folder_path = os.path.join(prefix_folder_path, folder_path)
    if from_excel:
        file_path = os.path.join(folder_path, f"{file_name}.xlsx")
    else:
        file_path = os.path.join(folder_path, f"{file_name}.csv")
    if not os.path.exists(file_path):
        print(f"No file found at {file_path}")
        if raiseError:
            raise FileNotFoundError(f"No file found at {file_path}")
        return None
    print(f"File found at {file_path} >> Reading File...")
    if from_excel:
        df = pd.read_excel(f"{file_path}")
    else:
        df = pd.read_csv(f"{file_path}")
    print("File Read!")
    return df

def insert_data_into_local_database(df, table_name):
    conn = sqlite3.connect(local_database)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
    print(f"Data successfully inserted into {table_name} table.")
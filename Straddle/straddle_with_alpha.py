#%%
import psycopg2
import numpy as np
import pandas as pd
#%%
import plotly.express as px
import plotly.graph_objs as go
#%%
from datetime import datetime, time, timedelta
#%%
import copy
#%% md
# Connection Details to QDap Database
#%%
host="192.168.2.23"
port=5432
user="amt"
dbname="qdap_test"
#%%
def make_connection_to_db(host, port, user, dbname):
    conn = psycopg2.connect(host= host, port=port, user=user, dbname=dbname)
    cursor = conn.cursor()
    return cursor, conn
#%%
# FUNCTION THAT RETRIEVES OPTIONS DATA IN BETWEEEN A PARTICULAR START DATE AND END DATE
def fetch_options_data_timeframe(cursor, symbol, expiry_type, start_date, end_date):
    cursor.execute(
        f'''
            SELECT *
            FROM ohlcv_options_per_minute oopm
            WHERE symbol = '{symbol}' 
            AND oopm.expiry_type = '{expiry_type}'
            AND oopm.date_timestamp >= '{start_date}'
            AND oopm.date_timestamp <= '{end_date}'
            ORDER BY date_timestamp ASC;
        '''
    )
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#%%
def fetch_options_data_timeframe_on_expiry_data(cursor, symbol, expiry_type, start_date, end_date):
    cursor.execute(
        f'''
            SELECT *
            FROM ohlcv_options_per_minute oopm
            WHERE symbol = '{symbol}' 
            AND oopm.expiry_type = '{expiry_type}'
            AND oopm.date_timestamp >= '{start_date}'
            AND oopm.date_timestamp <= '{end_date}'
            AND DATE(oopm.date_timestamp) = DATE(oopm.expiry)
            ORDER BY date_timestamp ASC;
        '''
    )
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#%%
# FUNCTION THAT RETRIEVES FUTURES DATA IN BETWEEN A PARTICULAR START DATE AND END DATE
def fetch_futures_data_timeframe(cursor, symbol, expiry_type, start_date, end_date):
    query = f'''
        SELECT *
        FROM ohlcv_future_per_minute ofpm
        WHERE ofpm.symbol = '{symbol}'
        AND ofpm.expiry_type = '{expiry_type}'
        AND date_timestamp >= '{start_date}'
        AND date_timestamp <= '{end_date}'
        ORDER BY date_timestamp ASC;
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#%%
# FUNCTION THAT RETRIEVES FUTURES DATA IN BETWEEN A PARTICULAR START DATE AND END DATE
def fetch_futures_data_timeframe_on_expiry_data(cursor, symbol, expiry_type, start_date, end_date):
    query = f'''
        SELECT *
        FROM ohlcv_future_per_minute ofpm
        WHERE ofpm.symbol = '{symbol}'
        AND ofpm.expiry_type = '{expiry_type}'
        AND date_timestamp >= '{start_date}'
        AND date_timestamp <= '{end_date}'
        AND DATE(ofpm.date_timestamp) = DATE(ofpm.expiry)
        ORDER BY date_timestamp ASC;
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df
#%%
# FUNCTION THAT RETRIEVES OPTIONS AND FUTURES FOR A LIST OF SYMBOLS(0DTE OPTIONS THROUGHOUT THE WEEK IN THIS CASE) AND SHAPES IT INTO A DICTIONARY FOR SMOOTH ACCESSIBILITY
def fetch(host, port, user, dbname, symbols, expiry_type_futures, expiry_type_options, start_date, end_date):
    cursor, conn = make_connection_to_db(host, port, user, dbname)
    dictionary_futures = {}
    dictionary_options = {}
    for symbol in symbols:
        df_futures = fetch_futures_data_timeframe_on_expiry_data(cursor, symbol, expiry_type_futures, start_date, end_date)
        df_futures['date_timestamp'] = pd.to_datetime(df_futures['date_timestamp'])
        df_futures['expiry'] = pd.to_datetime(df_futures['expiry'])
        
        df_options = fetch_options_data_timeframe_on_expiry_data(cursor, symbol, expiry_type_options, start_date, end_date)
        df_options['date_timestamp'] = pd.to_datetime(df_options['date_timestamp'])
        df_options['expiry'] = pd.to_datetime(df_options['expiry'])
        
        df_options = df_options[df_options['date_timestamp'].dt.date == df_options['expiry'].dt.date]
        expiries = pd.to_datetime(df_options['expiry']).dt.date
        df_futures = df_futures[df_futures['date_timestamp'].dt.date.isin(expiries)]
        
        dictionary_futures[symbol] = df_futures
        dictionary_options[symbol] = df_options
    cursor.close()
    conn.close()
    return dictionary_futures, dictionary_options
#%% md
# MIDCAP NIFTY: Monday
# 
# FINNIFTY: Tuesday
# 
# BANKNIFTY: Wednesday
# 
# NIFTY: Thursday
#%%
# ALL THE DESIRED PARAMETERS TO RUN THE STRATEGY
symbols = ["BANKNIFTY", "NIFTY", "FINNIFTY", "MIDCPNIFTY"]
start_date = '2023-09-06'
end_date = '2024-03-01'
expiry_type_options = 'IW1'
expiry_type_futures = 'I'
moneyness_strike = 0
fund_locked = 1000 # inr
fund_locked *= 100
transaction_cost = 11.5
slippage = 10
#%%
DICT_FUTURES, DICT_OPTIONS = fetch(host, port, user, dbname, symbols, expiry_type_futures, expiry_type_options, start_date, end_date)
#%%
dict_futures = DICT_FUTURES.copy()
dict_options = DICT_OPTIONS.copy()
#%%
def drop_duplicates_from_df(df):
    df = df.copy()
    df = df.drop(columns=['id'])
    df.drop_duplicates()
    return df
#%%
# A function that takes any dataframe of options or futures and makes it continuous
def fill_df_and_get_continuous_excluding_market_holidays(df, market_holidays_csv_path = r'C:\Users\user4\Desktop\exchange_holidays.csv'):
    # reading the market holidays from the list provided to us
    df = df.copy()
    market_holidays_df = pd.read_csv(market_holidays_csv_path, parse_dates=['holiday_date'])
    market_holidays = market_holidays_df['holiday_date'].dt.date.tolist()
    # generating a range of all the dates that exists from the first date to the last date
    start_date = df['date_timestamp'].dt.date.iloc[0]
    end_date = df['date_timestamp'].dt.date.iloc[-1]
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    # mask for the invalid days 
    trading_holidays = all_days.to_series().apply(lambda x: x.date() in market_holidays)
    trading_days = all_days[~trading_holidays]
    dates_of_trade = df['date_timestamp'].dt.date.unique()
    dates_of_trade_mask = trading_days.to_series().apply(lambda x: x.date() in dates_of_trade)
    trading_days = trading_days[dates_of_trade_mask]
    # Generate a complete range of the 375 trading minutes for each trading day
    trading_minutes = pd.date_range(start='09:15:00', end='15:29:00', freq='min').time
    # Create a complete index of trading timestamps
    complete_index = pd.DatetimeIndex([pd.Timestamp.combine(day, time) for day in trading_days for time in trading_minutes])
    df = df.set_index('date_timestamp')
    try:
        df = df.reindex(complete_index).ffill()
    except:
        pass
    return df, complete_index
#%%
# A Mega function that takes any dataframe of futures and cleans, structures, and fills the data making it continuous and complete. 
def process_parse_futures(df_futures):
    df_futures = df_futures.copy()
    # dropping duplicate entries
    df_futures = df_futures.drop_duplicates(subset='date_timestamp', keep='first')
    # required information
    info_needed = ['open', 'high', 'low', 'close', "date_timestamp", "symbol"]
    df_futures = df_futures[info_needed]
    # made continuous data if there were some discontinuity in the available data
    _, complete_index = fill_df_and_get_continuous_excluding_market_holidays(df_futures)
    df_futures['date'] = df_futures['date_timestamp'].dt.date
    df_futures = df_futures.set_index('date_timestamp')
    df_futures = df_futures.reindex(complete_index).ffill()
    return df_futures
#%%
# A FUNCTION FOR CONVERTING THE OPTIONS DATA INTO A FORMATE WHERE YOU CAN DIRECTLY ACCESS THE OHLC PRICE OF A CALL/PUT IN O(1). 
# HERE EACH ROW REPRESENTS A TIMESTAMP AND EACH COLUMN IS A DIFFERENT STRIKE
def convert(df, complete_index, ohlc, flag=True):
    if type(ohlc) == int:
        ohlc = ["open", "high", "low", "close"][ohlc]
    df = df.copy()
    df = df.pivot(columns='strike', values=ohlc)
    if flag:
        df = df.reindex(complete_index).ffill()
    return df
#%%
 # THIS FUNCTION STRUCTURES, PROCESSES, AND FORMATS THE OPTIONS DATA SIMILARLY AS OTHERS BUT WITHOUT FILLING MISSING DATA
# THIS IS DONE TO AVOID TRADING AT TIMESTAMPS WHERE THERE WAS NO TRADE IN THE REAL WORLD.
def get_real_put_call_data_ohlc_for_trades(symbols, ohlc, dict_options):
    dict_calls = {}
    dict_puts = {}
    for symbol in symbols:
        df_options = dict_options[symbol]
        df_calls = df_options[(df_options['opt_type'] == 'CE')]
        df_calls = drop_duplicates_from_df(df_calls)
        df_calls = df_calls.set_index('date_timestamp')
        df_calls = convert(df_calls, [], ohlc, False)
        df_puts  = df_options[(df_options['opt_type'] == 'PE')]
        df_puts = drop_duplicates_from_df(df_puts)
        df_puts = df_puts.set_index('date_timestamp')
        df_puts = convert(df_puts, [], ohlc, False)
        dict_calls[symbol] = df_calls
        dict_puts[symbol] = df_puts
    return dict_puts, dict_calls
#%%
# A Mega function that takes any dataframe of options and cleans, structures, and fills the data making it continuous and complete. 
def process_parse_options(df_options):
    df_options = df_options.copy()
    info_needed = ['open', 'high', 'low', 'close']
    # dropping duplicate entries
    df_options = df_options.drop_duplicates(subset=['date_timestamp', 'strike', 'opt_type'], keep='first')
    # processing calls
    df_calls = df_options[(df_options['opt_type'] == 'CE')]
    _, complete_index = fill_df_and_get_continuous_excluding_market_holidays(df_calls)
    df_calls = df_calls.set_index('date_timestamp')
    df_calls = [convert(df_calls, complete_index, info) for info in info_needed]
    # ease of access of a calls open close as a function of timestamp and strike
    # tracking all the existing strikes that were available for the calls
    call_strikes = np.array(df_calls[0].columns, dtype=int)
    # processing puts
    df_puts  = df_options[(df_options['opt_type'] == 'PE')]
    _, complete_index = fill_df_and_get_continuous_excluding_market_holidays(df_puts)
    df_puts = df_puts.set_index('date_timestamp')
    # ease of access of a puts open close as a function of timestamp and strike
    df_puts = [convert(df_puts, complete_index, info) for info in info_needed]
    # tracking all the existing strikes that were available for the puts
    put_strikes = np.array(df_puts[0].columns, dtype=int)
    return df_puts, df_calls, [put_strikes, call_strikes]
#%%
# It is not necessary that calls and puts will have all the strikes exactly the same. 
# this function provides us with the strikes that were present for both call and put options
# PROCESSING ALL THE COMMON STRIKES THAT EXISTED FOR CALLS AND PUTS IN THE TIME FRAME
# THIS IS DONE SO THAT SYNTHETIC FUTURES CAN BE CREATED WITH EASE
def get_common_strikes(symbols, dict_options):
    dict_options = copy.deepcopy(dict_options)
    strikes = {}
    for symbol in symbols:
        df_options = dict_options[symbol]
        _, _, [put_strikes, call_strikes] = process_parse_options(df_options)
        common_strikes = set(put_strikes).intersection(set(call_strikes))
        common_strikes = sorted(list(common_strikes))
        common_strikes = pd.Series(common_strikes)
        strikes[symbol] = common_strikes
    return strikes
#%%
# Since we are dealing with 0 days to expiry 0dte data, for which we are taking weekly expiry options
# we need to get indicators from weekly expiry futures as well
# since they do not exist in the market, we need to synthetically generate them
def get_synthetic_futures(symbols, dict_options, dict_futures):
    dict_options = copy.deepcopy(dict_options)
    dict_futures = copy.deepcopy(dict_futures)
    common_strikes_symbols = get_common_strikes(symbols, dict_options)
    ohlc_list = ['open', 'high', 'low', 'close']
    operated = [False, False, False, False]
    for zz, symbol in enumerate(symbols):
        df_futures = dict_futures[symbol]
        df_options = dict_options[symbol]
        # Process and parse futures and options data
        df_futures = process_parse_futures(df_futures)
        df_calls, df_puts, all_strikes = process_parse_options(df_options)
        if(not (df_futures.shape[0] == df_calls[0].shape[0] == df_puts[0].shape[0])):
            continue
        operated[zz] = True
        common_strikes = np.array(common_strikes_symbols[symbol])  # Ensure common_strikes is a NumPy array
        # Initialize synthetic columns with 0
        df_futures[[('synthetic_' + ohlc) for ohlc in ohlc_list]] = 0
        for time_index in range(df_puts[0].shape[0]):
            for ohlc_i, ohlc in enumerate(ohlc_list):
                future_price = df_futures[ohlc].iloc[time_index]
                # Find the closest strike index to the futures price
                ix = np.argmin(np.abs(common_strikes - future_price))
                c_minus_p = np.inf
                synthetic_future = None
                # Search within a range of moneyness
                for moneyness in range(max(ix - 1, 0), min(ix + 2, len(common_strikes))):
                    strike = common_strikes[moneyness]
                    diff = df_calls[ohlc_i].iloc[time_index][strike] - df_puts[ohlc_i].iloc[time_index][strike]
                    if diff < c_minus_p:
                        c_minus_p = diff
                        synthetic_future = strike + c_minus_p
                # Assign the synthetic future value
                df_futures.at[df_futures.index[time_index], 'synthetic_' + ohlc] = synthetic_future
        dict_futures[symbol] = df_futures
    return dict_futures, operated
#%%
# This is a function that gives us the ATR line (a volatility indicator) typically for the futures data.
# This is a part of my alpha (for more details, please refer to the README file)
def ATR(df_futures, synthetic_nature, period=14):
    df_futures = df_futures.copy()
    if synthetic_nature:
        prefix = 'synthetic_'
    else:
        prefix = ''
    df_futures['previous_close'] = df_futures[prefix+'close'].shift(1)
    df_futures['tr1'] = df_futures[prefix+'high'] - df_futures[prefix+'low']
    df_futures['tr2'] = (df_futures[prefix+'high'] - df_futures['previous_close']).abs()
    df_futures['tr3'] = (df_futures[prefix+'low'] - df_futures['previous_close']).abs()
    df_futures['true_range'] = df_futures[['tr1', 'tr2', 'tr3']].max(axis=1)
    # Calculate the ATR using Exponential Moving Average
    df_futures['atr'] = df_futures['true_range'].ewm(span=period, adjust=False).mean()
    # Drop the intermediate columns used for calculation
    df_futures.drop(columns=['previous_close', 'tr1', 'tr2', 'tr3', 'true_range'], inplace=True)
    return df_futures
#%%
# EMA or exponentially moving average
# I am taking an EMA on ATR
# This is a part of my alpha (for more details, please refer to the README file)
def ema(line, window_short, window_long, df):
    df = df.copy()
    col_short, col_long, col_crossover = f'ema_{window_short}', f'ema_{window_long}', f'crossover_{window_short}_{window_long}'
    df[col_short] = df[line].ewm(span=window_short).mean()
    df[col_long] = df[line].ewm(span=window_long).mean() 
    df[col_crossover] = 0
    polarity = df[col_short] - df[col_long]
    polarity = polarity > 0
    signals = []
    position_polarity_positive = polarity.iloc[window_long]
    for i in range(window_long, df.shape[0]):
        if((i+1 != df.shape[0]) and (df.index[i].date() != df.index[i+1].date())):
            position_polarity_positive = polarity.iloc[i+1]
            continue
        if(polarity.iloc[i] != position_polarity_positive):
            position_polarity_positive = polarity.iloc[i]
            df.at[df.index[i], col_crossover] = [-1, 1][int(position_polarity_positive)]
    return df
#%%
# This function utilizes all the functions till now to generate the synthetic futures and add the alpha data processed which can be used to find the trades
def get_futures_data_with_alpha_ideas_processed(symbols, entry_window_short, entry_window_long, exit_window_short, exit_window_long, dict_futures, dict_options):
    df_list = []
    for zz, symbol in enumerate(symbols):
        dict_futures_with_synthetic, operated = get_synthetic_futures(symbols, dict_options, dict_futures)
        if not operated[zz]:
            continue
        df_futures_with_synthetic = dict_futures_with_synthetic[symbol]
        df_futures_synthetic_atr = ATR(df_futures_with_synthetic, True)
        df = ema('atr', entry_window_short, entry_window_long, df_futures_synthetic_atr)
        df = ema('atr', exit_window_short, exit_window_long, df)
        df_list.append(df)
    df_merged = pd.concat(df_list, axis=0)
    return df_merged
#%%
# just a structuring step for easy processing later
def get_put_call_strikes_dict(symbols, dict_options):
    dict_calls = {}
    dict_puts = {}
    for symbol in symbols:
        _, _, [put_strikes, call_strikes] = process_parse_options(dict_options[symbol])
        dict_calls[symbol] = call_strikes
        dict_puts[symbol] = put_strikes
    return dict_puts, dict_calls
#%%
# just a structuring step for easy processing later
def get_processed_put_call_data_ohlc(symbols, ohlc, dict_options):
    dict_calls = {}
    dict_puts = {}
    for symbol in symbols:
        df_puts, df_calls, _ = process_parse_options(dict_options[symbol])
        dict_calls[symbol] = df_calls[ohlc]
        dict_puts[symbol] = df_puts[ohlc]
    return dict_puts, dict_calls
#%%
# This function incorporates the alpha logic to find the signalling time stamps in the working time range.
# It takes the data, and gives us the signals for when to buy what, and when to sell what
def signal_intraday(df_futures, dict_put_strikes, dict_call_strikes, window_short_entry, window_long_entry, window_short_exit, window_long_exit, threshold=1, smoothening_factor=0.9):
    cross_over_short_long_entry, cross_over_short_long_exit = f'crossover_{window_short_entry}_{window_long_entry}', f'crossover_{window_short_exit}_{window_long_exit}'
    signals = []
    entries = []
    enter, exit = 1, 0
    for i, (index_timestamp, bar) in enumerate(df_futures.iterrows()):
        if bar.name.time() < time(15, 30) and bar.name.time() > time(15, 20):
            continue  
        if bar.name.time() == time(15, 20):
            for entry in entries:
                [symbol, strike_call, strike_put, entry_futures_price_synthetic, running_avg, holding_time_points] = entry
                signals.append([exit, symbol, bar.name, strike_call, strike_put, current_futures_price_synthetic, "3:20 square off"])
            entries = []
            continue
        strikes_call = dict_call_strikes[bar['symbol']]
        strikes_put = dict_put_strikes[bar['symbol']]
        current_futures_price_synthetic = bar['synthetic_close']
        ix_call, ix_put = np.argmin(abs(strikes_call - current_futures_price_synthetic)), np.argmin(abs(strikes_put - current_futures_price_synthetic))
        current_atm_call_strike, current_atm_put_strike = strikes_call[ix_call], strikes_put[ix_put]
        for ii, entry in enumerate(entries[:]):
            [symbol, strike_call, strike_put, entry_futures_price_synthetic, running_avg, holding_time_points] = entry
            percentage = (current_futures_price_synthetic - entry_futures_price_synthetic)/entry_futures_price_synthetic
            percentage *= 100
            running_avg = (running_avg * holding_time_points + current_futures_price_synthetic)/(holding_time_points+1)
            holding_time_points += 1
            entry[4] = running_avg
            entry[5] = holding_time_points
            entries[ii] = entry
            # check 1% move here 
            if(abs(percentage) >= threshold):
                # if 1% move then check if there is a shift in the ATM available
                # no shift in atm, no square off
                if current_atm_call_strike == strike_call and current_atm_put_strike == strike_put:
                    continue
                # if 1% move then check the value of the metric.
                #if metric > 0.9, no square off
                gap = current_futures_price_synthetic - entry_futures_price_synthetic
                metric = (running_avg + gap)/current_futures_price_synthetic
                if metric > smoothening_factor:
                    continue
                # ADJUSTMENT LOGIC
                # sell net delta position
                signals.append([exit, symbol, df_futures.index[i+1], strike_call, strike_put, current_futures_price_synthetic, "Hedging squareoff"])
                entries[ii] = []
                # buy delta neutral
                signals.append([enter, symbol, df_futures.index[i+1], current_atm_call_strike, current_atm_put_strike, current_futures_price_synthetic, "Hedging position"])
                entries[ii] = [symbol, current_atm_call_strike, current_atm_put_strike, current_futures_price_synthetic, current_futures_price_synthetic, 1]
        # ENTRY LOGIC
        if bar[cross_over_short_long_entry] == 1:
            # Create a datetime object for comparison
            dummy_date = datetime.combine(datetime.today(), time(9, 15)) + timedelta(minutes=window_long_entry)
            dummy_date = dummy_date.time()
            if bar.name.time() < dummy_date:
                continue
            signals.append([enter, bar['symbol'], df_futures.index[i+1], current_atm_call_strike, current_atm_put_strike, current_futures_price_synthetic, "Entry crossover"])
            entries.append([bar['symbol'], current_atm_call_strike, current_atm_put_strike, current_futures_price_synthetic, current_futures_price_synthetic, 1])
        # EXIT LOGIC
        if bar[cross_over_short_long_exit] == -1:
            dummy_date = datetime.combine(datetime.today(), time(9, 15)) + timedelta(minutes=window_long_exit)
            dummy_date = dummy_date.time()
            if bar.name.time() < dummy_date:
                continue
            for entry in entries:
                [symbol, strike_call, strike_put, entry_futures_price_synthetic, running_avg, holding_time_points] = entry
                signals.append([exit, symbol, df_futures.index[i+1], strike_call, strike_put, current_futures_price_synthetic, "Exit crossover"])
            entries = []
    df_signals = pd.DataFrame(signals, columns=["Position", "Symbol", "Valid_Tradable_Time", "Strike_Call", "Strike_Put", "Futures_Price_Here", "Remarks"])
    df_signals.set_index('Valid_Tradable_Time', inplace=True)
    return signals, df_signals          
#%%
# based on the signals provided,
# this function executes trades and provides us the prices at which we do what according to our signals
def make_trades(df_signals, moneyness_strike, dict_puts, dict_calls, dict_put_strikes, dict_call_strikes):
    df_trades = df_signals.copy()
    df_trades['Straddle_Price'] = 0
    for i, (index_timestamp, signal) in enumerate(df_signals.iterrows()):
        symbol, timestamp, position, strike_call, strike_put = signal['Symbol'], index_timestamp, signal['Position'], signal['Strike_Call'], signal['Strike_Put']
        df_calls = dict_calls[symbol]
        df_puts = dict_puts[symbol]
        try:
            straddle_price = df_calls[strike_call].loc[timestamp] + df_puts[strike_put].loc[timestamp]
            df_trades.loc[timestamp, "Straddle_Price"] = straddle_price
        except:
            print(f"Either Call of strike {strike_call} or Put of Strike {strike_put} was not traded at {timestamp}")
    # df_trades.rename(columns={"Valid_Tradable_Time": "date_timestamp"}, inplace=True)
    # # df_trades = pd.DataFrame(trades, columns=['Position', 'Price', 'date_timestamp', 'strike_price_put', 'strike_price_call', 'futures_price', ])
    # df_trades = df_trades.set_index('date_timestamp')
    df_trades.sort_index(inplace=True)
    df_trades['cashflow'] = ((-1)*df_trades['Position'] + (1-df_trades['Position'])*(1))*df_trades['Straddle_Price']
    return df_trades
#%%
# alpha parameters
short_window_entry = 50
long_window_entry = 60
short_window_exit = 10
long_window_exit = 20
square_threshold = 0.5
sharpness_threshold = 0.99
#%%
df_merged = get_futures_data_with_alpha_ideas_processed(symbols, short_window_entry, long_window_entry, short_window_exit, long_window_exit, dict_futures, dict_options)
#%%
dict_put_strikes, dict_call_strikes = get_put_call_strikes_dict(symbols, dict_options)
#%%
signals, df_signals = signal_intraday(df_merged, dict_put_strikes, dict_call_strikes, short_window_entry, long_window_entry, short_window_exit, long_window_exit, square_threshold, sharpness_threshold)
#%%
df_signals
#%%
dict_puts, dict_calls = get_real_put_call_data_ohlc_for_trades(symbols, 0, dict_options)
#%%
df_trades = make_trades(df_signals, 0, dict_puts, dict_calls, dict_put_strikes, dict_call_strikes)
#%%
df_trades.head(50)
#%%
# Another custom class for metrics, used to calculate various statistics that are to be reported
class metrics: 
    def __init__(self, df_trades, fund_locked, risk_free_rate=12, transaction_costs=11.5, slippage = 10):
        self.fund_locked = fund_locked
        self.risk_free_rate = risk_free_rate
        self.df_trades = df_trades
        self.transaction_costs = transaction_costs
        self.slippage = slippage
    
    def get_expense_cost(self, amount):
        transaction_costs = self.transaction_costs
        slippage = self.slippage
        return amount * (transaction_costs + slippage)* 1/100 * 1/100
        
    def number_of_trades(self):
        return self.df_trades.count().iloc[0]

    def PNL(self):
        df_trades = self.df_trades
        profit, net_profit = 0, 0
        profits = []
        open_position = False
        for i, trade in df_trades.iterrows():
            price = trade['Straddle_Price']
            position = trade['Position']
            cash_flow_nature = 1
            if position == 1: # long -> pos = 1, short -> pos = 0
                cash_flow_nature = -1
            net_profit += cash_flow_nature * price - self.get_expense_cost(price)
            profit += cash_flow_nature * price - self.get_expense_cost(price)
            if open_position and not position:
                profits.append(profit)
                profit = 0
            open_position = position
        return net_profit, profits

    def net_turnover(self):
        prices = self.df_trades['Straddle_Price']
        return prices.sum()
        
    def net_expenditure(self):
        # 1% = 100 basis points => total_turnover * 0.01/100 * total_basis_points
        turnover = self.net_turnover()
        return self.get_expense_cost(turnover)

    def net_return(self):
        net_profit, _ = self.PNL()
        return 100 * net_profit/ self.fund_locked

    def sharpe(self):
        profits_per_day = self.per_day_pnl()
        profits_per_day = pd.Series(profits_per_day['cashflow'])
        sharpe_ratio = profits_per_day.mean()
        sharpe_ratio -= self.fund_locked * self.risk_free_rate * 1/100 * 1/365
        sharpe_ratio /= profits_per_day.std()
        return sharpe_ratio

    def max_drawdown(self):
        _, profits = self.PNL()
        increments = [(profits[i] - profits[i - 1]) for i in range(1, len(profits))]
        dd = 0
        max_dd = 0
        for inc in increments:
            dd += inc
            dd = min(0, dd)
            max_dd = min(dd, max_dd)
        return max_dd
        
    def per_day_pnl(self):
        df_trades = self.df_trades
        x = df_trades.groupby('date')['cashflow'].sum()
        pnl_per_day = pd.DataFrame(x)
        return pnl_per_day
#%%
Metrics = metrics(df_trades, 0)
#%%
Metrics.sharpe()
#%%
Metrics.per_day_pnl()
#%%
Metrics.net_turnover()/100
#%%
Metrics.net_expenditure()/100
#%%
Metrics.per_day_pnl().sum().iloc[0]/100
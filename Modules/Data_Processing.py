
import pandas as pd
import numpy as np
import QuantLib as ql
from collections import deque
from Modules.enums import Option, LongShort, DB, GreeksParameters, Leg, FNO, OHLC, Phase
from Modules.Utility import is_invalid_value


def drop_duplicates_from_df(df):
    df = df.copy()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    # very important step. it might happen that same timestamp same entries but different volumes might be present
    # if 'volume' in df.columns:
    #     df = df.drop(columns=['volume'])
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True, inplace = False)
    return df

def get_resampled_data(path, t):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.reset_index()
    df['group'] = df.index//t
    df = df.groupby('group', group_keys=False).first()
    df = df.set_index('index')
    return df 

def get_closest_index_binary_search(array, target):
    from bisect import bisect_left
    pos = bisect_left(array, target)
    if pos == 0:
        return 0
    if pos == len(array):
        return len(array) - 1
    # Return the closer of the two neighboring values
    before = array[pos - 1]
    after = array[pos]
    return pos if (after - target) < (target - before) else pos-1



# There are instances of same contract traded multiple times 
# with different volumes at same timestamp
def aggregate_df(df, fno):
    df = df.copy()
    df = drop_duplicates_from_df(df)
    # df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
    # df.set_index('date_timestamp', inplace=True)
    
    # Define aggregation rules for OHLCV columns
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'expiry': 'first',
        'expiry_type': 'first',
        'symbol': 'first'
    }
    
    # Group by non-OHLCV columns plus 'opt_type' and 'strike' for OPTIONS
    if fno == FNO.OPTIONS:
        df_grouped = df.groupby(['date_timestamp', 'opt_type', 'strike']).agg(aggregation_rules)
    else:
        # If not OPTIONS, handle other cases or default behavior
        df_grouped = df.groupby(['date_timestamp']).agg(aggregation_rules)
    
    df_grouped = df_grouped.reset_index()

    return df_grouped


def resample_df_to_timeframe(df, t, fno):
    df = df.copy()
    df = drop_duplicates_from_df(df)
    df['date_timestamp'] = pd.to_datetime(df['date_timestamp'])
    df.set_index('date_timestamp', inplace=True)
    
    # Define aggregation rules for OHLCV columns
    aggregation_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'expiry': 'first',
        'expiry_type': 'first',
        'symbol': 'first'
    }
    
    # Group by non-OHLCV columns plus 'opt_type' and 'strike' for OPTIONS
    if fno == FNO.OPTIONS:
        df_grouped = df.groupby(['opt_type', 'strike'])
        df_resampled = df_grouped.resample(f'{t}min').agg(aggregation_rules)
    else:
        # If not OPTIONS, handle other cases or default behavior
        df_resampled = df.resample(f'{t}min').agg(aggregation_rules)
    
    df_resampled = df_resampled.reset_index()
    
    # Define trading hours
    start_time = pd.to_datetime('09:15').time()
    end_time = pd.to_datetime('15:29').time()
    
    # Filter out rows outside trading hours
    df_resampled = df_resampled[df_resampled['date_timestamp'].dt.time.between(start_time, end_time)]
    all_days = df_resampled['date_timestamp'].dt.date
    market_valid_days = get_market_valid_days(all_days)
    df_resampled = df_resampled[df_resampled['date_timestamp'].dt.date.isin(market_valid_days)]
    df_resampled.reset_index(drop=True, inplace=True)
    df_resampled.sort_values(by='date_timestamp', inplace=True)
    print(f"Successfully resampled data to {t}mins timeframe")
    return df_resampled


def trading_minutes_between(t1, t2):
    daterange = pd.date_range(start=t1.date(), end=t2.date(), freq='B')
    marketdays = get_market_valid_days(daterange)
    if(len(marketdays) < 1):
        return 0
    return (len(marketdays)-1)*375 + (t2.hour-t1.hour)*60  + (t2.minute-t1.minute)

def get_market_valid_days(all_days, market_holidays_csv_path = r"C:\Users\vinayak\Desktop\Backtesting\Database\exchange_holidays.csv"):
    market_holidays_df = pd.read_csv(market_holidays_csv_path, parse_dates=['holiday_date'])
    market_holidays = market_holidays_df['holiday_date'].dt.date.tolist()
    all_days = pd.Series(all_days)
    trading_holidays = all_days.apply(lambda x: pd.to_datetime(x).date() in market_holidays)
    trading_days = all_days[~trading_holidays]
    return trading_days

def is_market_day(day, market_holidays_csv_path = r"C:\Users\vinayak\Desktop\Backtesting\Database\exchange_holidays.csv"):
    market_holidays_df = pd.read_csv(market_holidays_csv_path, parse_dates=['holiday_date'])
    market_holidays = market_holidays_df['holiday_date'].dt.date.tolist()
    verdict = pd.to_datetime(day).date() not in market_holidays 
    if verdict:
        print(f'{pd.to_datetime(day).date()} was a Market Day')
    else:
        print(f'{pd.to_datetime(day).date()} was a Marktet Close Day')
    return verdict 

def get_date_minus_n_days(start_date, days, include = False):
    from datetime import timedelta
    number_of_days_done = 0
    day_delta = 0 if include else 1
    start_date = pd.to_datetime(start_date)
    previous = None
    while number_of_days_done != days:
        previous = start_date - timedelta(day_delta)
        if len(get_market_valid_days(previous)):
            number_of_days_done += 1
        day_delta += 1
    # print(start_date, days, previous, day_delta, len(get_market_valid_days(previous)))
    nature = "" if include else "without "
    print(f"Market Valid arithmetic for {start_date.date().strftime('%d/%b/%Y')} - {days}days = {previous.date().strftime('%d/%b/%Y')} ({nature}including the start_date)")
    return previous

def get_latest_market_valid_day(date):
    from datetime import timedelta
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    latest_date = date
    while not is_market_day(latest_date):
        latest_date -= timedelta(1)
    return pd.to_datetime(latest_date)

def get_next_market_valid_day(date):
    from datetime import timedelta
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    latest_date = date
    while not is_market_day(latest_date):
        latest_date += timedelta(1)
    return pd.to_datetime(latest_date)

def get_nearest_market_valid_day(date, phase):
    from datetime import timedelta
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    latest_date = date
    while not is_market_day(latest_date):
        latest_date += timedelta(1) * phase.value
    return pd.to_datetime(latest_date)

def isLastNdays(timestamp, n, *portfolio, **kwargs):
    isLastN = False
    trading_days = True
    general_days = False 
    if 'trading' in kwargs:
        trading_days=True
        general_days=False
    if 'general' in kwargs:
        general_days=True
        trading_days=False
    for ticker in portfolio:
        expiry = ticker.get_expiry(timestamp)
        boolean = False
        if trading_days:
            boolean = trading_minutes_between(timestamp, expiry)/375 <= n
        if general_days:
            (expiry - timestamp).days <= n
        isLastN = isLastN | boolean
    return isLastN

'''
1. Takes the start and end date from the data frame
2. creates a range of all business days between the start and end date
3. Reads all the market holidays from the exchange holidays
4. removes all the non-trading days from this range using market holidays data provided to us
5. if 0DTE, then keeps only the dates having a trade in the dataframe
6. fills all the minutes in those days using forward fill
'''
def get_continuous_excluding_market_holidays(df, is_zero_DTE = False, t = 1, market_holidays_csv_path = r"C:\Users\vinayak\Desktop\Backtesting\Database\exchange_holidays.csv"):
    # reading the market holidays from the list provided to us
    df = df.copy()
    # generating a range of all the dates that exists from the first date to the last date
    if 'date_timestamp' in df.columns:
        start_date = df['date_timestamp'].dt.date.iloc[0]
        end_date = df['date_timestamp'].dt.date.iloc[-1]
    elif isinstance(df.index, pd.DatetimeIndex):
        start_date = df.index[0]
        end_date = df.index[-1]
    else:
        raise ValueError("Date Timestamp Information not provided")
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    # mask for the invalid days
    trading_days = get_market_valid_days(all_days, market_holidays_csv_path)
    if is_zero_DTE:
        dates_of_trade = df['date_timestamp'].dt.date.unique()
        dates_of_trade_mask = trading_days.apply(lambda x: x.date() in dates_of_trade)
        # dates_of_trade_mask = trading_days.to_series().apply(lambda x: x.date() in dates_of_trade)
        trading_days = trading_days[dates_of_trade_mask]
    # Generate a complete range of the 375 trading minutes for each trading day
    trading_minutes = pd.date_range(start='09:15:00', end='15:29:00', freq=f'{t}min').time
    # Create a complete index of trading timestamps
    complete_index = pd.DatetimeIndex([pd.Timestamp.combine(day, time) for day in trading_days for time in trading_minutes])
    try:
        if 'date_timestamp' in df.columns:
            df = df.set_index('date_timestamp')
            df = df.reindex(complete_index)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reindex(complete_index)
        else:
            raise ValueError("Date Timestamp Information not provided")
    except:
        pass
    return df, complete_index

'''
Processing step for futures data
drops duplicates
sets index
keeps needed columns
fills and makes continuous
'''
def process_parse_futures(df_futures, is_zero_DTE=False, t=1, toFill = True, isEod = False):
    df_futures = df_futures.copy()
    df_futures = drop_duplicates_from_df(df_futures)
    df_futures['date_timestamp'] = pd.to_datetime(df_futures['date_timestamp'])
    df_futures['expiry'] = pd.to_datetime(df_futures['expiry'])
    df_futures = fill_expiry(df_futures, False)
    if df_futures['expiry_type'].iloc[0] == 'I':
        df_futures['days_to_expiry'] = abs((df_futures['expiry'] - df_futures['date_timestamp']))
        df_futures = df_futures.loc[df_futures.groupby('date_timestamp')['days_to_expiry'].idxmin()]
        df_futures = df_futures.drop(columns=['days_to_expiry'])
        # df_futures = df_futures[df_futures['date_timestamp'].dt.month == df_futures['expiry'].dt.month]
    if not isEod:
        if t == 1:
            df_futures = aggregate_df(df_futures, FNO.FUTURES)
        else:
            df_futures = resample_df_to_timeframe(df_futures, t, FNO.FUTURES)
    # dropping duplicate entries
    # made continuous data if there were some discontinuity in the available data
    if not isEod:
        _, complete_index = get_continuous_excluding_market_holidays(df_futures, is_zero_DTE, t)
    # df_futures['date_timestamp'] = pd.to_datetime(df_futures['date_timestamp'])
    df_futures = drop_duplicates_from_df(df_futures)
    df_futures.set_index('date_timestamp', inplace=True)
    df_futures = df_futures.sort_index()
    if df_futures.index.duplicated().any():
        print("Duplicates found after setting index:", df_futures.index[df_futures.index.duplicated()])
    # df_futures['expiry'] = pd.to_datetime(df_futures['expiry'])
    if not isEod:
        df_futures = df_futures.reindex(complete_index)
    df_futures['date'] = df_futures.index.date
    df_futures['expiry_type'] = df_futures['expiry_type'].ffill()
    df_futures['expiry_type'] = df_futures['expiry_type'].bfill()
    df_futures = fill_expiry(df_futures)
    if toFill:
        df_futures = df_futures.groupby('date', group_keys=False).apply(lambda contract: contract.ffill().bfill())
    return df_futures

'''
processing step for options data
takes input raw option dataframe
outputs arrays of puts, calls, and put_strikes, calls_strikes
puts[0] = df_puts of that symbol with open prices
open, high, low, close = 0, 1, 2, 3
'''
def process_parse_options(df_options, is_zero_DTE=False, t=1, toFill = True):
    # df_options = df_options.copy()
    df_options = drop_duplicates_from_df(df_options)
    df_options['date_timestamp'] = pd.to_datetime(df_options['date_timestamp'])
    df_options['expiry'] = pd.to_datetime(df_options['expiry'])
    df_options = fill_expiry(df_options, False)
    if df_options['expiry_type'].iloc[0] == 'I':
        df_options['days_to_expiry'] = abs((df_options['expiry'] - df_options['date_timestamp']))
        df_options = df_options.loc[df_options.groupby(['date_timestamp', 'opt_type', 'strike'])['days_to_expiry'].idxmin()]
        df_options = df_options.drop(columns=['days_to_expiry'])
    if t == 1:
        df_options = aggregate_df(df_options, FNO.OPTIONS)
    else:
        df_options = resample_df_to_timeframe(df_options, t, FNO.OPTIONS)

    # df_options = drop_duplicates_from_df(df_options)
    info_needed = ['open', 'high', 'low', 'close']

    
    
    
    # processing calls
    df_calls = df_options[(df_options['opt_type'] == 'CE')]
    _, complete_index = get_continuous_excluding_market_holidays(df_calls, is_zero_DTE, t)
    # ease of access of a calls open close as a function of timestamp and strike
    df_call_strikes = (
        df_options[df_options['opt_type'] == 'CE']
        .groupby('expiry')['strike']
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )
    df_calls = df_calls.set_index('date_timestamp')
    df_calls = df_calls.sort_index()
    df_calls_arr = [pivot(df_calls, complete_index, info, True) for info in info_needed]
    if toFill:
        df_calls_arr = [df.groupby('date', group_keys=False).apply(lambda contract : contract.ffill().bfill()) for df in df_calls_arr]
    for df in df_calls_arr:
        df['expiry'] = df_calls.groupby(df_calls.index)['expiry'].first()
    df_calls_arr = [fill_expiry(df_calls_ohlc) for df_calls_ohlc in df_calls_arr]


    # processing puts
    df_puts  = df_options[(df_options['opt_type'] == 'PE')]
    _, complete_index = get_continuous_excluding_market_holidays(df_puts, is_zero_DTE, t)
    # ease of access of a puts open close as a function of timestamp and strike
    df_put_strikes = (
        df_options[df_options['opt_type'] == 'PE']
        .groupby('expiry')['strike']
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )
    df_puts = df_puts.set_index('date_timestamp')
    df_puts = df_puts.sort_index()
    df_puts_arr = [pivot(df_puts, complete_index, info, True) for info in info_needed]
    # tracking all the existing strikes that were available for the puts
    if toFill:
        df_puts_arr = [df.groupby('date', group_keys=False).apply(lambda contract : contract.ffill().bfill()) for df in df_puts_arr]
    for df in df_puts_arr:
        df['expiry'] = df_puts.groupby(df_puts.index)['expiry'].first()
    df_puts_arr = [fill_expiry(df_puts_ohlc) for df_puts_ohlc in df_puts_arr]


    expiry_date_map = {}
    for expiry in df_options['expiry'].unique():
        df = df_options[df_options['expiry'] == expiry]
        start_date = df['date_timestamp'].min().date()
        end_date = df['date_timestamp'].max().date()
        expiry_date_map[expiry] = (pd.to_datetime(start_date), pd.to_datetime(end_date))
    expiry_date_map = expiry_date_map

    # tracking all the existing strikes that were available for the puts
    return df_puts_arr, df_calls_arr, [df_put_strikes, df_call_strikes], expiry_date_map

'''
A FUNCTION FOR CONVERTING THE OPTIONS DATA INTO A FORMAT WHERE 
YOU CAN DIRECTLY ACCESS THE OHLC PRICE OF A CALL/PUT IN O(1). 
HERE EACH ROW REPRESENTS A TIMESTAMP AND EACH COLUMN IS A DIFFERENT STRIKE
'''
def pivot(df, complete_index, ohlc, flag=True, toFill=True):
    if type(ohlc) == int:
        ohlc = ["open", "high", "low", "close"][ohlc]
    df = df.copy()
    df = df.pivot(columns='strike', values=ohlc)  # Perform pivot
    if flag:
        df = df.reindex(complete_index)  # Align to complete_index
    df['date'] = df.index.date
    return df


def fill_expiry(df, TimestampIsIndex=True):
    df = df.copy()
    df['expiry'] = df['expiry'].ffill()
    df['expiry'] = df['expiry'].bfill()
    if TimestampIsIndex:
        df.loc[pd.to_datetime(df['expiry']).dt.date < (df.index).date, 'expiry'] = None
    else:
        df.loc[pd.to_datetime(df['expiry']).dt.date < (pd.to_datetime(df['date_timestamp']).dt.date), 'expiry'] = None
    df['expiry'] = df['expiry'].bfill()
    df['expiry'] = df['expiry'].ffill()
    return df

# PROCESSING ALL THE COMMON STRIKES THAT EXISTED FOR CALLS AND PUTS IN THE TIME FRAME
# THIS IS DONE SO THAT SYNTHETIC FUTURES CAN BE CREATED WITH EASE

# from functools import lru_cache
# @lru_cache(maxsize=128)
def get_greeks(greek_parameters: GreeksParameters, raiseError=True):
    try:
        evaluation_date = pd.to_datetime(greek_parameters.timestamp)
        expiry_date = pd.to_datetime(greek_parameters.expiry_date)
        
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)
        
        # Construct the option
        # Construct the option
        if greek_parameters.option_type == Option.Call:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(greek_parameters.option_strike))
        elif greek_parameters.option_type == Option.Put:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(greek_parameters.option_strike))

        exercise = ql.EuropeanExercise(ql.Date(expiry_date.day, expiry_date.month, expiry_date.year))
        option = ql.VanillaOption(payoff, exercise)

        # Construct the Black-Scholes process
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(greek_parameters.underlying_price))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year), greek_parameters.risk_free_rate, ql.Actual365Fixed())
        )
        dividend_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year), greek_parameters.dividend_yield, ql.Actual365Fixed())
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year), ql.NullCalendar(), 0.20, ql.Actual365Fixed())
        )
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)
        
        # Set the pricing engine for the option
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)

        # Calculate implied volatility
        implied_vol = option.impliedVolatility(greek_parameters.option_price, bsm_process)

        # Update the volatility term structure with the implied volatility
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(ql.Settings.instance().evaluationDate, ql.NullCalendar(), implied_vol, ql.Actual365Fixed())
        )
        bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

        # Calculate the Greeks
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega() / 100
        theta = option.thetaPerDay()
        rho = option.rho()

        result = {
            'implied_volatility': implied_vol,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
        return result
    
    except Exception as error:
        print(f"Error in Generating Greeks from Quantlib | {error}")
        if raiseError:
            raise ValueError(error)
        return None


def get_optimized_greeks(greek_parameters : GreeksParameters, log_check_db = False, raiseError = True):
    import Modules.Data as data
    time_to_expiry = (greek_parameters.expiry_date - greek_parameters.timestamp).days
    read_query = f'''
        SELECT implied_volatility, delta, gamma, vega, theta, rho
        FROM Greeks_2
        WHERE underlying_price = '{greek_parameters.underlying_price}'
        AND option_strike = '{greek_parameters.option_strike}'
        AND time_to_expiry = '{time_to_expiry}'
        AND option_type = '{greek_parameters.option_type.name}'
    '''
    result = data.query_db(DB.LocalDB, read_query)

    if isinstance(result, pd.DataFrame) and not result.empty:
        greeks = {
            'implied_volatility': float(result['implied_volatility'].iloc[0]),
            'delta': float(result['delta'].iloc[0]),
            'gamma': float(result['gamma'].iloc[0]),
            'vega': float(result['vega'].iloc[0]),
            'theta': float(result['theta'].iloc[0]),
            'rho': float(result['rho'].iloc[0])
        }
        if log_check_db:
            print("Greeks Found in DB")
        return greeks
    
    greeks = get_greeks(greek_parameters, raiseError)
    if greeks is None:
        if log_check_db:
            print("Greeks Calculated, Error Occured. Greeks are None")
        return None
    
    insert_query = f'''
        INSERT INTO Greeks_2 (
            underlying_price, 
            option_strike, 
            option_price, 
            time_to_expiry, 
            option_type,  
            implied_volatility, delta, gamma, vega, theta, rho
        ) VALUES (
            {greek_parameters.underlying_price}, 
            {greek_parameters.option_strike}, 
            {greek_parameters.option_price}, 
            {time_to_expiry}, 
            '{greek_parameters.option_type.name}', 
            {greeks['implied_volatility']}, 
            {greeks['delta']}, 
            {greeks['gamma']}, 
            {greeks['vega']}, 
            {greeks['theta']}, 
            {greeks['rho']}
        )
    '''
    result = data.query_db(DB.LocalDB, insert_query)
    if log_check_db:
        print("Greeks Calculated, Inserted into DB")
    return greeks


def get_correlations(dict_futures, ohlc):
    df_merged = pd.concat([df[ohlc].rename(stock) for stock, df in dict_futures.items()], axis = 1)
    correlation_matrix = df_merged.corr()
    return correlation_matrix


class Trades:
    def __init__(self):
        # for a dataframe sorted based on timestamps to see what was traded at what time
        self.tradesArr = []
        # for matching trades using tokenID so that metrics can be calculated
        self.tradesDict = {}

        self.tokens = {}

        self.token_count = 0
    def make_trade(self, timestamp, remarks, *legs):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        if len(legs) == 0:
            print(f"Length of Legs is 0 || from take_position (from dp.ticker or outside) -> Trades.make_trade()")
            return
        self.addTradeDict(timestamp, remarks, *legs)
        self.addTradeArr(timestamp, remarks, *legs)

    def addTradeArr(self, timestamp, remarks, *legs):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        # Timestamp is added here and not in the ticker.take_position
        # because timestamp is not of a leg, it is of a trade
        trade_obj = {'Timestamp': timestamp, 'Remarks': remarks}
        for leg_number, leg in enumerate(legs):
            for key, value in leg.__dict__.items():
                # leg_number = self.tokens[leg.id()]
                trade_obj[f'{key} Leg({leg_number})'] = value
        self.tradesArr.append(trade_obj)

    def addTradeDict(self, timestamp, remarks, *legs):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        for leg in legs:
            obj = {'Timestamp': timestamp, 'Remarks': remarks}
            if leg.id() in self.tokens.keys():
                leg_number = self.tokens[leg.id()]
            else:
                self.token_count += 1
                leg_number = self.token_count
                self.tokens[leg.id()] = leg_number
            for key, value in leg.__dict__.items():
                obj[f'{key} Leg({leg_number})'] = value
            token_id = leg.id()
            if token_id not in self.tradesDict:
                self.tradesDict[token_id] = deque()
            self.tradesDict[token_id].append(obj)


class ticker:
    def __init__(self, symbol, lot_size, FetchData, start_date, end_date, expiry_type, toFill = True, timeframe = 1, is_index = False, risk_free=0.1):
        if symbol is None:
            raise ValueError("Symbol can't be None/Empty")
        if lot_size is None:
            raise ValueError("Lot Size can't be None/Empty")
        self.symbol = symbol
        self.lot_size = lot_size
        self.FetchData = FetchData
        self.start_date = start_date
        self.end_date = end_date
        self.expiry_type = expiry_type
        self.toFill = toFill
        self.resampling_timeframe = timeframe
        self.is_index = is_index
        self.is_stock = not is_index
        self.risk_free = risk_free
        self.tokens = {}
        self.Trades = Trades()
        self.ohlc = OHLC.close
        # self.is_zero_DTE = is_zero_DTE
        if isinstance(FetchData, bool) and FetchData == True:
            self.initializeFNOdata()
        else:
            if FetchData[0] == True:
                self.InitializeFuturesData()
            if FetchData[1] == True:
                self.InitializeOptionsData()

    def InitializeFuturesData(self):
        from Modules import Data as data
        self.df_futures = data.get_futures_data_with_timestamps_between(self.symbol, self.start_date, self.end_date, self.expiry_type, self.resampling_timeframe)
        self.df_futures = process_parse_futures(self.df_futures, False, self.resampling_timeframe, self.toFill)
        self.timestamps = self.df_futures.index

    def InitializeOptionsData(self):
        from Modules import Data as data
        self.df_options = data.get_options_data_with_timestamps_between(self.symbol, self.start_date, self.end_date, self.expiry_type, self.resampling_timeframe)
        self.arr_df_puts, self.arr_df_calls, [self.df_put_strikes, self.df_call_strikes], self.expiry_date_map = process_parse_options(self.df_options, False, self.resampling_timeframe, self.toFill)
    
    def initializeFNOdata(self):
        from Modules import Data as data
        self.InitializeFuturesData()
        self.InitializeOptionsData()
    
    def StartEnd(self, start, end):
        if not isinstance(start, pd.Timestamp):
            if isinstance(start, int):
                start = self.timestamps[start]
            else:
                start = pd.to_datetime(start)
        if not isinstance(end, pd.Timestamp):
            if isinstance(end, int):
                end = self.timestamps[end]
            else:
                end = pd.to_datetime(end)
        return self.timestamps[(self.timestamps >= start) & (self.timestamps <= end)]

    def set_ohlc(self, ohlc):
        self.ohlc = ohlc
        print(f"OHLC for {self.symbol} is set to {self.ohlc}")
        return

    def get_opts(self, opt_type):
        if opt_type == Option.Put:
            return self.arr_df_puts[self.ohlc.value]
        elif opt_type == Option.Call:
            return self.arr_df_calls[self.ohlc.value]
        else:
            print(f"Make consistent import for Option from enums")
            print(f"opt_type: {opt_type}, type: {type(opt_type)}")
            print(f"Option.Call: {Option.Call}, type: {type(Option.Call)}")
            print(f"id(opt_type): {id(opt_type)}")
            print(f"id(Option.Call): {id(Option.Call)}")
            print(f"id(opt_type.__class__): {id(opt_type.__class__)}")
            print(f"id(Option.Call.__class__): {id(Option.Call.__class__)}")
            print(f"opt_type module: {opt_type.__class__.__module__}")
            print(f"Option module: {Option.__module__}")
        return None

    def get_strikes(self, opt_type, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        if opt_type == Option.Put:
            df_strikes = self.df_put_strikes
        else:
            df_strikes = self.df_call_strikes
        for expiry, strikes in df_strikes.items():
            start_date, end_date = self.expiry_date_map.get(expiry, (None, None))
            if start_date.date() <= timestamp.date() <= end_date.date():
                return np.array(strikes)
        print("Timestamp does not exist in the given data. Strike Data not Fetched")
        return None
    
    def get_strike_gap(self):
        all_strikes = list(self.df_call_strikes.values())[0] + list(self.df_put_strikes.values())[0]
        all_strikes = pd.Series(all_strikes)
        return (all_strikes - all_strikes.shift(1)).median()
    
    def get_opts_price(self, timestamp, opt_type, strike):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        options = self.get_opts(opt_type)
        try:
            price = options.loc[timestamp, strike]
            return float(price)
        except Exception as e:
            print(f"Exception Occured | {e}")
            if timestamp.time() < pd.Timestamp('09:15').time() or timestamp.time() > pd.Timestamp('15:29').time():
                raise ValueError(f"Invalid Market Timestamp || {timestamp.time()}")
            if timestamp.date() > pd.to_datetime(self.end_date).date() or timestamp.date() > pd.to_datetime(self.start_date).date():
                raise ValueError(f"{timestamp.date()} is not covered in the ticker definition ({pd.to_datetime(self.start_date).date()} to {pd.to_datetime(self.end_date).date()})")
            if strike not in self.get_strikes(opt_type, timestamp):
                raise ValueError(f"Not a Valid Strike")             
            raise ValueError("Error in getting price of the option. Either the timestamp is invalid or strike didn't exist in this period")

    def inspect(self, timestamp, moneyness_index=0):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        moneyness_call_strike, _ = self.find_moneyness_strike(timestamp, moneyness_index, Option.Call)
        price_call = self.get_opts_price(timestamp, Option.Call, moneyness_call_strike)
        moneyness_put_strike, _ = self.find_moneyness_strike(timestamp, moneyness_index, Option.Put)
        price_put = self.get_opts_price(timestamp, Option.Put, moneyness_put_strike)
        price_futures = self.get_futures_price(timestamp)
        if moneyness_index > 0:
            moneyness = "OTM"
        elif moneyness_index == 0:
            moneyness = "ATM"
        elif moneyness_index < 0:
            moneyness = "ITM"
        print(f"{self.symbol} | Timestamp: {timestamp.strftime('%d/%b/%Y %H:%M:%S')}")
        print(f">> Future | Price: {price_futures}")
        detail = "" if moneyness_index == 0 else f"-{abs(moneyness_index)}"
        print(f">> {moneyness}{detail} Put Option | Strike: {moneyness_put_strike} | Price: {price_put}")
        print(f">> {moneyness}{detail} Call Option | Strike: {moneyness_call_strike} | Price: {price_call}")
        print()
        return

    def get_futures_price(self, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        return self.df_futures.loc[timestamp, self.ohlc.name]

    def get_expiry(self, timestamp, string = False):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        expiry = self.df_futures.loc[timestamp, 'expiry']
        if string:
            return str(expiry)
        return expiry

    def is_expiry(self, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        return timestamp.date() == self.get_expiry(timestamp).date()

    def get_futures_data_in_window(self, timestamp, window=None):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        if window is None:
            window = self.look_back_window
        df_futures_before_timestamp = self.df_futures.loc[:timestamp, self.ohlc.name]
        df_futures_in_look_back_period = df_futures_before_timestamp.iloc[-window:]
        return df_futures_in_look_back_period

    def get_futures_data(self, start, end):
        return self.df_futures.loc[start:end, self.ohlc.name]

    def get_common_strikes(self, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        callstrikes, putstrikes = self.get_strikes(Option.Call, timestamp), self.get_strikes(Option.Put, timestamp)
        common_strikes = set(callstrikes).intersection(putstrikes)
        common_strikes = sorted(common_strikes)
        return pd.Series(common_strikes)

    def get_strike_for_synthetic(self, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        strikes = self.get_common_strikes(timestamp)
        futures_price = self.df_futures.loc[timestamp, self.ohlc.name]
        atm_strike_index = get_closest_index_binary_search(strikes, int(futures_price))
        # atm_strike_index = np.argmin(np.abs(strikes - int(futures_price)))
        return strikes[atm_strike_index]

    def take_position(self, timestamp, remarks, *legs : Leg):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        legs_objects = []
        for leg in legs:
            if leg.Instrument == FNO.FUTURES:
                PriceLeg = self.get_futures_price(timestamp)
            else:
                PriceLeg = self.get_opts(leg.Instrument).loc[timestamp, leg.Strike]
            legs_objects.append(
                Leg(
                    leg.Position, 
                    leg.Lots, 
                    leg.Instrument, 
                    leg.Strike, 
                    PriceLeg, 
                    leg.LegName
                )
            )
        self.Trades.make_trade(timestamp, remarks, *legs_objects)
        return

    def view_trades(self):
        return pd.DataFrame(self.Trades.tradesArr)

    def find_optprice_strike(self, timestamp, price, opt_type):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        strikes_prices_df = self.get_opts(opt_type).select_dtypes(include=[np.number])
        if timestamp not in strikes_prices_df.index:
            raise ValueError(f"Timestamp {timestamp} not found in data.")
        prices = strikes_prices_df.loc[timestamp].values
        strikes = strikes_prices_df.columns.values

        # Filter out NaN values
        valid_mask = ~np.isnan(prices)
        valid_prices = prices[valid_mask]
        valid_strikes = strikes[valid_mask]

        # Find the index of the closest price
        price_index = np.argmin(np.abs(valid_prices - price))
        strike = valid_strikes[price_index]

        return strike

    def find_nearest_strike(self, timestamp, underlying_price, opt_type):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        strikes = np.array(self.get_strikes(opt_type, timestamp))
        strike_index = get_closest_index_binary_search(strikes, underlying_price)
        nearest_strike = strikes[strike_index]
        return nearest_strike, strike_index

    def find_underlyingPriceMovement_strike(self, timestamp, moneyness_direction, percentage, opt_type):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        underlying = self.get_futures_price(timestamp)
        strike_desired = underlying * (1 + opt_type.value * moneyness_direction * percentage/100)
        strikes = np.array(self.get_strikes(opt_type, timestamp))
        strike_index = get_closest_index_binary_search(strikes, strike_desired)
        strike = strikes[strike_index]
        return strike

    # OTM is positive, ITM is negative
    def find_moneyness_strike(self, timestamp, moneyness, opt_type):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        try:
            strikes = np.array(self.get_strikes(opt_type, timestamp))
            # strikes.sort()
            futures_price = self.df_futures.loc[timestamp, self.ohlc.name]
            # atm_strike_index = np.argmin(np.abs(strikes - int(futures_price)))
            atm_strike_index = get_closest_index_binary_search(strikes, futures_price)
            moneyness_index = atm_strike_index
            if opt_type == Option.Put:
                moneyness_index -= moneyness
            elif opt_type == Option.Call:
                moneyness_index += moneyness
            moneyness_index = min(moneyness_index, len(strikes)-1)
            moneyness_index = max(0, moneyness_index)
            return strikes[moneyness_index], moneyness_index
        except Exception as e:
            print(f"from ticker.find_moneyness_strike(): {e}")

    # def generate_synthetic_futures(self):
    #     print(f"Generating synthetic futures data for {self.symbol}")
    #     if not (self.df_futures.shape[0] == self.get_opts(Option.Call).shape[0] == self.get_opts(Option.Put).shape[0]):
    #         return

    #     # MODIFIED get_common_strikes function with timestamp parameter
    #     # Initialize synthetic columns with 0
    #     self.df_futures['synthetic_' + self.ohlc.name] = 0
    #     for time_index, timestamp in enumerate(self.df_futures.index):
    #         common_strikes = self.get_common_strikes(timestamp)
    #         future_price = self.get_futures_price(timestamp)
    #         # Find the closest strike index to the futures price
    #         ix = np.argmin(np.abs(common_strikes - future_price))
    #         c_minus_p = np.inf
    #         synthetic_future = None
    #         # Search within a range of moneyness
    #         for moneyness in range(max(ix - 1, 0), min(ix + 2, len(common_strikes))):
    #             strike = common_strikes[moneyness]
    #             diff = self.get_opts(Option.Call).iloc[time_index][strike] - self.get_opts(Option.Put).iloc[time_index][strike]
    #             if diff < c_minus_p:
    #                 c_minus_p = diff
    #                 synthetic_future = strike + c_minus_p
    #         # Assign the synthetic future value
    #         self.df_futures.at[self.df_futures.index[time_index], 'synthetic_' + self.ohlc.name] = synthetic_future
    #     print(f"Synthetic futures data for {self.symbol} generated successfully")
    #     return

    def get_iv_at(self, timestamp, log_sanity=False):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        if 'iv' not in self.df_futures.columns:
            self.generate_iv(timestamp, log_sanity)
        elif is_invalid_value(self.df_futures.loc[timestamp, 'iv']):
            self.generate_iv(timestamp, log_sanity)
        return self.df_futures.loc[timestamp, 'iv']
    
    def get_iv_during(self, start, end):
        if not isinstance(start, pd.Timestamp):
            start = pd.to_datetime(start)
        if not isinstance(end, pd.Timestamp):
            start = pd.to_datetime(end)
        timestamps = self.StartEnd(start, end)
        for timestamp in timestamps:
            self.get_iv_at(timestamp)
        return self.df_futures.loc[timestamps, 'iv']

    def generate_iv(self, timestamp, log_sanity = False, log_error = True):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        if log_sanity:
            print(f"Timestamp: {timestamp}")
        self.df_futures.loc[timestamp, 'iv'] = pd.NA
        try:
            atm_call_strike, _ = self.find_moneyness_strike(timestamp, 0, Option.Call)
            atm_put_strike, _ = self.find_moneyness_strike(timestamp, 0, Option.Put)
            futures_price = self.get_futures_price(timestamp)
            if log_sanity:
                print(f"  IV Sanity Parameters")
                print(f"  >> Symbol: {self.symbol}")
                print(f"  >> Spot Price(XX): {futures_price}")
                print(f"  >> ATM Put Strike: {atm_put_strike}")
                print(f"  >> ATM Call Strike: {atm_call_strike}")
            if pd.isna(futures_price) or futures_price <= 0:
                if log_error:
                    if not log_sanity:
                        print(f"Timestamp: {timestamp}")
                    print(f"Invalid futures price {futures_price}")
                return

            greeks_call = self.Greeks(timestamp, Option.Call, atm_call_strike, log_sanity)
            greeks_put = self.Greeks(timestamp, Option.Put, atm_put_strike, log_sanity)
            greeks_call_result = "FAILURE" if greeks_call is None else "SUCCESS"
            greeks_put_result = "FAILURE" if greeks_put is None else "SUCCESS"
            if (greeks_call is None) or (greeks_put is None):
                if log_error:
                    if not log_sanity:
                        print(f"Timestamp: {timestamp}")
                    print(f"Failed to Generate IV")
                    print(f">> Greeks from Call: {greeks_call_result}")
                    print(f">> Greeks from Put: {greeks_put_result}")
                    print('------------------------------------------')
                return
            iv_calculated_from_call = greeks_call['implied_volatility']
            iv_calculated_from_put = greeks_put['implied_volatility']
            iv = (iv_calculated_from_call + iv_calculated_from_put) / 2
            self.df_futures.loc[timestamp, 'iv'] = iv
            if log_sanity:
                print(f"Successfully Generated IV = {float(np.round(iv*100, 2))}%")
                print(f"  IV Sanity Values")
                print(f"  IV from Call: {float(np.round(iv_calculated_from_call*100, 2))}%")
                print(f"  IV from Put: {float(np.round(iv_calculated_from_put*100, 2))}%")
        except Exception as e:
            if log_error:
                if not log_sanity:
                    print(f"Timestamp: {timestamp}")
                print(f"{self.symbol}'s error: {e}")
        if log_sanity:
            print('------------------------------------------')

    def generate_complete_iv_data(self):
        print(f"Generating {self.symbol} Implied Volatility(IV) data...")
        for timestamp, _ in self.df_futures.iterrows():
            self.generate_iv(timestamp)
        print(f"Implied Volatility data for {self.symbol} generated successfully!")
        print()

    def Greeks(self, timestamp, opt_type, strike, log_sanity = False, log_error = True):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        futures_price = self.df_futures.loc[timestamp, self.ohlc.name] 
        strike = float(strike)
        options_price = self.get_opts_price(timestamp, opt_type, strike)
        expiry = self.get_expiry(timestamp)
        time_to_expiry = (expiry - timestamp).days
        if log_sanity:
            print(f"  Greeks Sanity Parameters")
            print(f"  >> Symbol: {self.symbol}")
            print(f"  >> Spot(xx): {futures_price}")
            print(f"  >> Strike: {strike}")
            print(f"  >> Option Price: {options_price}")
            print(f"  >> Time to Expiry: {time_to_expiry}days")
            print(f"  >> Option Type: {opt_type.name}")
            print(f"  >> Expiry: {expiry.date()}")
        try:
            greeks = get_greeks(
                GreeksParameters(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    expiry_date=expiry,
                    option_type=opt_type,
                    option_strike=strike,
                    option_price=options_price,
                    underlying_price=futures_price,
                    risk_free_rate=0,
                    dividend_yield=0,
                ),
                log_sanity
            )
            if log_sanity:
                print(f"  Greeks Sanity Values (Per Option)")
                print(f"  >> IV: {greeks['implied_volatility']}")
                print(f"  >> Delta: {greeks['delta']}")
                print(f"  >> Gamma: {greeks['gamma']}")
                print(f"  >> Theta: {greeks['theta']}")
                print(f"  >> Vega: {greeks['vega']}")
                print(f"  >> Rho: {greeks['rho']}")
            return greeks
        except Exception as e:
            if log_error:
                print(f"  Could not Generate Greeks for {self.symbol}")
                print(f"  Problem | {e}")
                if not log_sanity:
                    print(f"  Greeks Sanity Parameters")
                    print(f"  >> Symbol: {self.symbol}")
                    print(f"  >> Spot(xx): {futures_price}")
                    print(f"  >> Strike: {strike}")
                    print(f"  >> Option Price: {options_price}")
                    print(f"  >> Time to Expiry: {time_to_expiry}days")
                    print(f"  >> Option Type: {opt_type.name}")
                    print(f"  >> Expiry: {expiry.date()}")
            # print(f"dates from dp.ticker.Greeks: {pd.to_datetime(timestamp).date(), pd.to_datetime(expiry).date()}")

    def get_net_delta(ticker, timestamp):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        net_delta = 0
        for _, token in ticker.tokens.items():
            net_delta += token.stats.loc[timestamp, f'net_delta']
        return net_delta

    def generate_vix_data(self):
        print(f"Generating {self.symbol} VIX data...")

        # Iterate over each timestamp in the futures data
        for timestamp, _ in self.df_futures.iterrows():
            futures_price = self.df_futures.loc[timestamp, self.ohlc.name]

            if pd.isna(futures_price) or futures_price <= 0:
                print(f"Skipping {timestamp}: Invalid futures price {futures_price}")
                continue
            print(f"Timestamp: {timestamp}")
            common_strikes = self.get_common_strikes(timestamp)

            # Initialize variables for VIX calculation
            days_to_expiry = (pd.to_datetime(self.get_opts(Option.Call).loc[timestamp, 'expiry']) - pd.to_datetime(timestamp)).days
            time_to_expiry_in_years = days_to_expiry/365
            if time_to_expiry_in_years <= 0:
                continue
            first_strike_below, _ = self.find_moneyness_strike(timestamp, 1, Option.Call)
            strikes_used_for_vix = []
            counter_could_not_use, counter_could_use = 0, 0
            # Calculate contribution from each strike (both calls and puts)
            sigma_squared = 2/days_to_expiry
            product_term = 0
            strike_gap = pd.Series(np.diff(common_strikes)).median()
            for strike in common_strikes:
                try:
                    option = Option.Call if strike >= futures_price else Option.Put
                    liq_check = self.Greeks(timestamp, option, strike, False)
                    numerator = strike_gap * np.exp(self.risk_free * time_to_expiry_in_years) * self.get_opts_price(timestamp, option, strike)
                    denominator = strike * strike
                    product_term += numerator/denominator

                    strikes_used_for_vix.append(int(strike))
                    counter_could_use += 1
                except Exception as e:
                    print(f"Error calculating VIX for {self.symbol} at strike {strike}: {e}")
                    counter_could_not_use += 1
                    continue
            sigma_squared *= product_term
            minus_term = (futures_price/first_strike_below - 1)**2
            minus_term /= time_to_expiry_in_years
            sigma_squared -= minus_term
            print(f"sigma_squared = {sigma_squared}")
            try:
                vix = np.sqrt(sigma_squared) * 100
            except Exception as e:
                print(e)
            print(f"VIX = {vix}")
            print(f"Futures Price = {futures_price}")
            print(f"Strikes Used for VIX Calculation (Greeks were Calculated): {strikes_used_for_vix}")
            print(f"Number of Strikes that could not contribute to VIX = {counter_could_not_use} || Could contribute = {counter_could_use}")


            # Store the VIX value in the futures dataframe
            self.df_futures.loc[timestamp, 'vix'] = vix
            print('-----------------------------------')

        print(f"VIX data for {self.symbol} generated successfully!")

    def reset_trades(self):
        self.Trades = Trades()
        self.tokens = {}


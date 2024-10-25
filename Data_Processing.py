
import pandas as pd
import numpy as np
import QuantLib as ql
from collections import deque
from enums import *



def drop_duplicates_from_df(df):
    df = df.copy()
    df = df.drop(columns=['id'])
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(inplace = False)
    return df

import pandas as pd

def resample_df_to_timeframe(df, t, fno):
    df = df.copy()
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
        'symbol': 'first',
    }
    
    # Identify OHLCV columns and non-OHLCV columns
    ohlcv_columns = list(aggregation_rules.keys())
    
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





def get_market_valid_days(all_days, market_holidays_csv_path = r"C:\Users\vinayak\Desktop\Backtesting\exchange_holidays.csv"):
    market_holidays_df = pd.read_csv(market_holidays_csv_path, parse_dates=['holiday_date'])
    market_holidays = market_holidays_df['holiday_date'].dt.date.tolist()
    all_days = pd.Series(all_days)
    trading_holidays = all_days.apply(lambda x: pd.to_datetime(x).date() in market_holidays)
    trading_days = all_days[~trading_holidays]
    return trading_days

def get_date_minus_n_days(start_date, days, include = False):
    from datetime import timedelta
    number_of_days_done = 0
    day_delta = 0 if include else 1
    start_date = pd.to_datetime(start_date)
    previous = None
    while number_of_days_done != days:
        previous = start_date - timedelta(day_delta)
        print(start_date, days, previous, day_delta, len(get_market_valid_days(previous)))
        if len(get_market_valid_days(previous)):
            number_of_days_done += 1
        day_delta += 1
    return previous
'''
Reads all the market holidays from the csv provided to us
takes the start and end date from the data frame
creates a range of all business days between the start and end date
removes all the non-trading days from this range using market holidays data provided to us
if 0DTE, then keeps only the dates having a trade in the dataframe
fills all the minutes in those days using forward fill
'''
def get_continuous_excluding_market_holidays(df, is_zero_DTE = False, t = 1, market_holidays_csv_path = r"C:\Users\vinayak\Desktop\Backtesting\exchange_holidays.csv"):
    # reading the market holidays from the list provided to us
    df = df.copy()
    # generating a range of all the dates that exists from the first date to the last date
    start_date = df['date_timestamp'].dt.date.iloc[0]
    end_date = df['date_timestamp'].dt.date.iloc[-1]
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    # mask for the invalid days
    trading_days = get_market_valid_days(all_days, market_holidays_csv_path)
    if is_zero_DTE:
        dates_of_trade = df['date_timestamp'].dt.date.unique()
        dates_of_trade_mask = trading_days.to_series().apply(lambda x: x.date() in dates_of_trade)
        trading_days = trading_days[dates_of_trade_mask]
    # Generate a complete range of the 375 trading minutes for each trading day
    trading_minutes = pd.date_range(start='09:15:00', end='15:29:00', freq=f'{t}min').time
    # Create a complete index of trading timestamps
    complete_index = pd.DatetimeIndex([pd.Timestamp.combine(day, time) for day in trading_days for time in trading_minutes])
    df = df.set_index('date_timestamp')
    # try:
    #     df = df.reindex(complete_index).ffill()
    # except:
    #     pass
    return df, complete_index




'''
Processing step for futures data
drops duplicates
sets index
keeps needed columns
fills and makes continuous
'''
def process_parse_futures(df_futures, is_zero_DTE=False, t=1, toFill = True, isEod = False):
    info_needed = ['open', 'high', 'low', 'close', 'date_timestamp', 'symbol', 'expiry']
    df_futures = df_futures.copy()
    # dropping duplicate entries
    df_futures['date_timestamp'] = pd.to_datetime(df_futures['date_timestamp'])

    # df_futures = df_futures.drop_duplicates(subset='date_timestamp', keep='first')
    df_futures = drop_duplicates_from_df(df_futures)
    # required information
    df_futures = df_futures[info_needed]
    # made continuous data if there were some discontinuity in the available data
    if not isEod:
        _, complete_index = get_continuous_excluding_market_holidays(df_futures, is_zero_DTE, t)
        df_futures['date'] = df_futures['date_timestamp'].dt.date
    df_futures = df_futures.set_index('date_timestamp')
    if not isEod:
        df_futures = df_futures.reindex(complete_index)
    if toFill:
        df_futures = df_futures.ffill()
    df_futures = merge_prior_info(df_futures, df_futures)
    return df_futures




'''
A FUNCTION FOR CONVERTING THE OPTIONS DATA INTO A FORMATE WHERE 
YOU CAN DIRECTLY ACCESS THE OHLC PRICE OF A CALL/PUT IN O(1). 
HERE EACH ROW REPRESENTS A TIMESTAMP AND EACH COLUMN IS A DIFFERENT STRIKE
'''
def convert(df, complete_index, ohlc, flag=True, toFill = True):
    if type(ohlc) == int:
        ohlc = ["open", "high", "low", "close"][ohlc]
    df = df.copy()
    df = df.pivot(columns='strike', values=ohlc)
    if flag:
        df = df.reindex(complete_index)
    if toFill:
        df = df.ffill()
    return df




# '''
# THIS FUNCTION STRUCTURES, PROCESSES, AND FORMATS THE OPTIONS DATA SIMILARLY 
# AS OTHERS BUT WITHOUT FILLING MISSING DATA
# THIS IS DONE TO AVOID TRADING AT TIMESTAMPS WHERE THERE WAS NO TRADE IN THE REAL WORLD.
# '''
# def get_real_put_call_data_ohlc_for_trades(symbols, ohlc, dict_options):
#     dict_calls = {}
#     dict_puts = {}
#     for symbol in symbols:
#         df_options = dict_options[symbol]
#         df_calls = df_options[(df_options['opt_type'] == 'CE')]
#         df_calls = drop_duplicates_from_df(df_calls)
#         df_calls = df_calls.set_index('date_timestamp')
#         df_calls = convert(df_calls, [], ohlc, False)
#         df_puts  = df_options[(df_options['opt_type'] == 'PE')]
#         df_puts = drop_duplicates_from_df(df_puts)
#         df_puts = df_puts.set_index('date_timestamp')
#         df_puts = convert(df_puts, [], ohlc, False)
#         dict_calls[symbol] = df_calls
#         dict_puts[symbol] = df_puts
#     return dict_puts, dict_calls




def merge_prior_info(df_info, df_to):
    df_to = df_to.copy()
    df_to['expiry'] = df_info.groupby(df_info.index)['expiry'].first()
    df_to['expiry'] = df_to['expiry'].ffill()
    df_to = df_to[pd.to_datetime(df_to['expiry']).dt.date >= (df_to.index).date]
    df_to['expiry'] = df_to['expiry'].bfill()
    return df_to

'''
processing step for options data
takes input raw option dataframe
outputs arrays of puts, calls, and put_strikes, calls_strikes
puts[0] = df_puts of that symbol with open prices
open, high, low, close = 0, 1, 2, 3
'''
def process_parse_options(df_options, is_zero_DTE=False, t=1, toFill = True):
    # df_options = df_options.copy()
    df_options['date_timestamp'] = pd.to_datetime(df_options['date_timestamp'])
    info_needed = ['open', 'high', 'low', 'close']
    # dropping duplicate entries
    df_options = df_options.drop_duplicates(subset=['date_timestamp', 'strike', 'opt_type'], keep='first')

    # print("printing from process_parse_options")
    # print("raw options timestamps", df_options['date_timestamp'].unique())


    # processing calls
    df_calls = df_options[(df_options['opt_type'] == 'CE')]
    _, complete_index = get_continuous_excluding_market_holidays(df_calls, is_zero_DTE, t)
    # ease of access of a calls open close as a function of timestamp and strike
    df_call_strikes = (df_options[(df_options['opt_type'] == 'CE')]).groupby('expiry')['strike'].unique().to_dict()
    df_calls = df_calls.set_index('date_timestamp')
    df_calls_arr = [convert(df_calls, complete_index, info, True, toFill) for info in info_needed]
    # print("converted calls timestamps", df_calls_arr[0].index.unique())
    # tracking all the existing strikes that were available for the calls
    call_strikes = np.sort(np.array(df_calls_arr[0].columns, dtype=int))
    df_calls_arr = [merge_prior_info(df_calls, df_calls_ohlc) for df_calls_ohlc in df_calls_arr]
    


    # processing puts
    df_puts  = df_options[(df_options['opt_type'] == 'PE')]
    _, complete_index = get_continuous_excluding_market_holidays(df_puts, is_zero_DTE, t)
    # ease of access of a puts open close as a function of timestamp and strike
    df_put_strikes = (df_options[(df_options['opt_type'] == 'PE')]).groupby('expiry')['strike'].unique().to_dict()
    df_puts = df_puts.set_index('date_timestamp')
    df_puts_arr = [convert(df_puts, complete_index, info, True, toFill) for info in info_needed]
    # tracking all the existing strikes that were available for the puts
    put_strikes = np.sort(np.array(df_puts_arr[0].columns, dtype=int))
    df_puts_arr = [merge_prior_info(df_puts, df_puts_ohlc) for df_puts_ohlc in df_puts_arr]

    # tracking all the existing strikes that were available for the puts
    return df_puts_arr, df_calls_arr, [put_strikes, call_strikes], [df_put_strikes, df_call_strikes]





# PROCESSING ALL THE COMMON STRIKES THAT EXISTED FOR CALLS AND PUTS IN THE TIME FRAME
# THIS IS DONE SO THAT SYNTHETIC FUTURES CAN BE CREATED WITH EASE





def find_closest_strike(strikes, value):
    strikes = np.array(strikes)
    strikes.sort()
    closest_strike_index = np.argmin(np.abs(strikes - int(value)))
    closest_strike = strikes[closest_strike_index]
    return closest_strike, closest_strike_index


def get_greeks(underlying_price, strike_price, option_price, risk_free_rate, dividend_yield, evaluation_date,
               expiry_date, type):
    evaluation_date = pd.to_datetime(evaluation_date)
    expiry_date = pd.to_datetime(expiry_date)

    # Set the evaluation date (current date)
    ql.Settings.instance().evaluationDate = ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year)

    # Construct the option
    if type == Option.Call:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike_price))
    elif type == Option.Put:
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(strike_price))

    exercise = ql.EuropeanExercise(ql.Date(expiry_date.day, expiry_date.month, expiry_date.year))

    # Create the option object
    option = ql.VanillaOption(payoff, exercise)

    # Construct the Black-Scholes process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(underlying_price))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year), risk_free_rate,
                       ql.Actual365Fixed()))
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year), dividend_yield,
                       ql.Actual365Fixed()))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ql.Date(evaluation_date.day, evaluation_date.month, evaluation_date.year),
                            ql.NullCalendar(), 0.20, ql.Actual365Fixed()))  # Initial guess for volatility

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # Set the pricing engine for the option
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    option.setPricingEngine(engine)

    # Calculate implied volatility
    implied_vol = option.impliedVolatility(option_price, bsm_process)


    # Update the volatility term structure with the implied volatility
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ql.Settings.instance().evaluationDate, ql.NullCalendar(), implied_vol, ql.Actual365Fixed()))
    bs_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, vol_handle)

    # Reassign the pricing engine with the updated process
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bs_process))

    # Calculate the Greeks
    delta = option.delta()
    gamma = option.gamma()
    vega = option.vega()/100
    theta = option.thetaPerDay()  # Theta is per day
    rho = option.rho()

    # Return implied volatility and Greeks
    return {
        'implied_volatility': implied_vol,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }

def get_smooth_iv_horizontally(iv1, iv2, timestamp, expiry):
    time_to_expiry = int((pd.to_datetime(expiry) - pd.to_datetime(timestamp)).days)
    if time_to_expiry > 10:
        return iv1
    weight1 = time_to_expiry/ 10
    weight2 = 1 - weight1
    return iv1 * weight1 ** 2 + iv2 * weight2 ** 2/ (weight1 ** 2 + weight2 ** 2)

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

    def make_trade(self, timestamp, remarks, *legs):
        if len(legs) == 0:
            print(f"Length of Legs is 0 || from take_position (from dp.ticker or outside) -> Trades.make_trade()")
            return
        self.addTradeArr(timestamp, remarks, *legs)
        self.addTradeDict(timestamp, remarks, *legs)

    def addTradeArr(self, timestamp, remarks, *legs):
        # Timestamp is added here and not in the ticker.take_position
        # because timestamp is not of a leg, it is of a trade
        trade_obj = {'timestamp': timestamp, 'remarks': remarks}
        for i, leg in enumerate(legs):
            for key, value in leg.items():
                trade_obj[f'{key}{i}'] = value
        self.tradesArr.append(trade_obj)

    def addTradeDict(self, timestamp, remarks, *legs):
        for leg in legs:
            obj = {'timestamp': timestamp, 'remarks': remarks}
            for key, value in leg.items():
                obj[key] = value
            token_id = f"{leg['StrikeLeg']}{leg['OptTypeLeg'].name}"
            if token_id not in self.tradesDict:
                self.tradesDict[token_id] = deque()
            self.tradesDict[token_id].append(obj)

class ticker:
    def __init__(self, symbol, df_futures, df_options, is_index, components, is_component, weight, lot_size, toFill = True, t = 1, is_zero_DTE = False, risk_free=0.1):
        if symbol is None:
            raise ValueError("symbol can't be None")
        if df_futures is None:
            raise ValueError("df_futures can't be None")
        if df_options is None:
            raise ValueError("df_options can't be None")
        if is_index is None:
            raise ValueError("is_index can't be None")
        if components is None:
            raise ValueError("components can't be None")
        if is_component is None:
            raise ValueError("is_component can't be None")
        if weight is None:
            raise ValueError("weight can't be None")
        if lot_size is None:
            raise ValueError("lot_size can't be None")
        self.symbol = symbol
        self.resampling_timeframe = t
        self.expiry_type_futures = df_futures['expiry_type'].iloc[0]
        self.expiry_type_options = df_options['expiry_type'].iloc[0]
        self.df_futures = process_parse_futures(df_futures, is_zero_DTE, t, toFill)
        self.df_options = df_options
        self.arr_df_puts, self.arr_df_calls, [self.putstrikes, self.callstrikes], [self.df_put_strikes, self.df_call_strikes] = process_parse_options(df_options, is_zero_DTE, t, toFill)
        self.is_index = is_index
        self.is_component = is_component
        self.weight = weight
        self.components = components
        self.hasActivePosition = False
        self.last_trade = None
        self.tokens = {}
        self.Trades = Trades()
        self.hedging = pd.DataFrame(columns = ['date_timestamp', 'lots', 'position', 'futures_price', 'delta_added'])
        self.hedging.set_index('date_timestamp', inplace=True)
        self.ohlc = OHLC.close
        self.lot_size = lot_size
        self.risk_free = risk_free
        self.toFill = toFill

    def set_ohlc(self, ohlc):
        self.ohlc = ohlc
        print(f"OHLC for {self.symbol} is set to {self.ohlc}")
        return

    def get_opts(self, opt_type):
        if opt_type == Option.Put:
            return self.arr_df_puts[self.ohlc.value]
        elif opt_type == Option.Call:
            return self.arr_df_calls[self.ohlc.value]
        return None

    def get_strikes(self, opt_type, timestamp):
        if opt_type == Option.Put:
            df_strikes = self.df_put_strikes
            for expiry, strikes in df_strikes.items():
                df = self.df_options[self.df_options['expiry'] == expiry]
                start_date = df['date_timestamp'].min()
                end_date = df['date_timestamp'].max()
                if start_date.date() <= timestamp.date() <= end_date.date():
                    return np.sort(np.array(strikes))
        else:
            df_strikes = self.df_call_strikes
            for expiry, strikes in df_strikes.items():
                df = self.df_options[self.df_options['expiry'] == expiry]
                start_date = df['date_timestamp'].min()
                end_date = df['date_timestamp'].max()
                if start_date.date() <= timestamp.date() <= end_date.date():
                    return np.sort(np.array(strikes))

    def get_opts_price(self, timestamp, opt_type, strike):
        return self.get_opts(opt_type).loc[timestamp, strike]

    def get_futures_price(self, timestamp):
        return self.df_futures.loc[timestamp, self.ohlc.name]

    def get_expiry(self, timestamp, string = False):
        expiry = self.df_futures.loc[timestamp, 'expiry']
        if string:
            return expiry
        return pd.to_datetime(expiry)

    def get_futures_data_in_window(self, timestamp, window=None):
        if window is None:
            window = self.look_back_window
        df_futures_before_timestamp = self.df_futures.loc[:timestamp, self.ohlc.name]
        df_futures_in_look_back_period = df_futures_before_timestamp.iloc[-window:]
        return df_futures_in_look_back_period

    def get_common_strikes(self, timestamp):
        callstrikes, putstrikes = self.get_strikes(Option.Call, timestamp), self.get_strikes(Option.Put, timestamp)
        common_strikes = set(callstrikes).intersection(putstrikes)
        common_strikes = sorted(common_strikes)
        return pd.Series(common_strikes)

    def get_correlations_in_look_back_period(self, timestamp):
        futures_data_look_back_period = []
        for stock, component in self.components.items():
            component_futures_data_look_back_period = component.get_futures_data_look_back_period(timestamp).rename(stock)
            if len(component_futures_data_look_back_period) < self.look_back_window:
                return None
            futures_data_look_back_period.append(component_futures_data_look_back_period)
        df_merged = pd.concat(futures_data_look_back_period, axis=1)
        # print(f"Correlations Calculation between {self.symbol} components completed")
        correlation_matrix = df_merged.corr()
        return correlation_matrix

    # def get_correlations(self):
    #     df_merged = pd.concat([component.df_futures[self.ohlc.name].rename(stock) for stock, component in self.components.items()], axis=1)
    #     self.correlation_matrix = df_merged.corr()
    #     print(f"Correlations Calculation between {self.symbol} components completed")
    #     return self.correlation_matrix

    # leg = (position_leg1 = LongShort.type, opt_type_leg1 = Option.type, moneyness = int, lots = int)
    def take_position(self, timestamp, remarks, *legs):
        legs_objects = []
        for leg in legs:
            # print(leg)
            # print(type(leg))
            # print(f"leg['position'], leg['moneyness'], leg['opt_type'], leg['lots']: {type(leg['position']), type(leg['moneyness']), type(leg['opt_type']), type(leg['lots'])}")
            # StrikeLeg, _ = self.find_moneyness_strike(timestamp, leg['moneyness'], leg['opt_type'])
            StrikeLeg = leg['strike']
            if leg['opt_type'] == FNO.FUTURES:
                PriceLeg = self.get_futures_price(timestamp)
            else:
                PriceLeg = self.get_opts(leg['opt_type']).loc[timestamp, StrikeLeg]
            leg_obj = {
                'PriceLeg': PriceLeg, 
                'StrikeLeg': StrikeLeg, 
                'OptTypeLeg': leg['opt_type'], 
                'PositionLeg': leg['position'], 
                'LotsLeg': leg['lots']}
            legs_objects.append(leg_obj)
            # print(f"PRINTING FROM DP.TICKER. {self.symbol} lots: {leg['lots']}")
        self.Trades.make_trade(timestamp, remarks, *legs_objects)
        return

    def find_moneyness_strike(self, timestamp, moneyness, opt_type):
        try:
            strikes = np.array(self.get_strikes(opt_type, timestamp))
            strikes.sort()
            futures_price = self.df_futures.loc[timestamp, self.ohlc.name]
            atm_strike_index = np.argmin(np.abs(strikes - int(futures_price)))
            moneyness_index = atm_strike_index
            if opt_type == Option.Put:
                moneyness_index += moneyness
            elif opt_type == Option.Call:
                moneyness_index -= moneyness
            moneyness_index = min(moneyness_index, len(strikes)-1)
            moneyness_index = max(0, moneyness_index)
            return strikes[moneyness_index], moneyness_index
        except Exception as e:
            print(f"from ticker.find_moneyness_strike(): {e}")

    def generate_synthetic_futures(self):
        print(f"Generating synthetic futures data for {self.symbol}")
        if not (self.df_futures.shape[0] == self.get_opts(Option.Call).shape[0] == self.get_opts(Option.Put).shape[0]):
            return

    # MODIFIED get_common_strikes function with timestamp parameter
        # Initialize synthetic columns with 0
        self.df_futures['synthetic_' + self.ohlc.name] = 0
        for time_index, timestamp in enumerate(self.df_futures.index):
            common_strikes = self.get_common_strikes(timestamp)
            future_price = self.get_futures_price(timestamp)
            # Find the closest strike index to the futures price
            ix = np.argmin(np.abs(common_strikes - future_price))
            c_minus_p = np.inf
            synthetic_future = None
            # Search within a range of moneyness
            for moneyness in range(max(ix - 1, 0), min(ix + 2, len(common_strikes))):
                strike = common_strikes[moneyness]
                diff = self.get_opts(Option.Call).iloc[time_index][strike] - self.get_opts(Option.Put).iloc[time_index][strike]
                if diff < c_minus_p:
                    c_minus_p = diff
                    synthetic_future = strike + c_minus_p
            # Assign the synthetic future value
            self.df_futures.at[self.df_futures.index[time_index], 'synthetic_' + self.ohlc.name] = synthetic_future
        print(f"Synthetic futures data for {self.symbol} generated successfully")
        return

    def generate_iv_data(self):
        print(f"Generating {self.symbol} Implied Volatility(IV) data...")
        for timestamp, _ in self.df_futures.iterrows():
            print(timestamp)
            try:
                atm_call_strike, _ = self.find_moneyness_strike(timestamp, 0, Option.Call)
                atm_put_strike, _ = self.find_moneyness_strike(timestamp, 0, Option.Put)
                futures_price = self.df_futures.loc[timestamp, self.ohlc.name]

                self.df_futures.loc[timestamp, 'iv'] = 0
                if pd.isna(futures_price) or futures_price <= 0:
                    print(f"Skipping {timestamp}: Invalid futures price {futures_price}")
                    continue

                greeks_call = self.greeks(timestamp, Option.Call, atm_call_strike, True)
                greeks_put = self.greeks(timestamp, Option.Put, atm_put_strike, True)
                greeks_call_result = "FAILURE" if greeks_call is None else "SUCCESS"
                greeks_put_result = "FAILURE" if greeks_put is None else "SUCCESS"
                greeks_result = f"Failed to Generate IV || Greeks from Call: {greeks_call_result}, Greeks from Put: {greeks_put_result}"
                if greeks_call is None or greeks_put is None:
                    print(greeks_result)
                    print('------------------------------------------')
                    continue
                iv_calculated_from_call = greeks_call['implied_volatility']
                iv_calculated_from_put = greeks_put['implied_volatility']
                iv = (iv_calculated_from_call + iv_calculated_from_put) / 2
                self.df_futures.loc[timestamp, 'iv'] = iv
                print(f"Successfully Generated IV")
            except Exception as e:
                self.df_futures.loc[timestamp, 'iv'] = 0
                print(f"{self.symbol}'s error: {e}")
            print('------------------------------------------')
        print(f"Implied Volatility data for {self.symbol} generated successfully!")
        print()

    # def generate_ic_data(self):
    #     if not self.is_index:
    #         print("Implied correlation of a non index asset can't be generated")
    #         return
    #     print(f"Generating {self.symbol} Implied Correlation(IC) data...")
    #     for timestamp, _ in self.df_futures.iterrows():
    #         print(timestamp)
    #         numerator, denominator = (self.df_futures.loc[timestamp, 'iv'])**2, 0
    #         correlations = self.get_correlations_in_look_back_period(timestamp)
    #         if correlations is None:
    #             self.df_futures.loc[timestamp, 'ic'] = 0
    #             print(f"Insufficient timestamp to calculate Correlations between Stocks")
    #             print('---------------------')
    #             continue
    #         for s1, component1 in self.components.items():
    #             for s2, component2 in self.components.items():
    #                 if s1 == s2:
    #                     numerator -= (component1.weight * component1.df_futures.loc[timestamp, 'iv'])**2
    #                 else:
    #                     # denominator += component1.weight * component2.weight * component1.df_futures.loc[
    #                     #     timestamp, 'iv'] * component2.df_futures.loc[timestamp, 'iv'] * correlations.loc[s1, s2]
    #                     denominator += component1.weight * component2.weight * component1.df_futures.loc[
    #                         timestamp, 'iv'] * component2.df_futures.loc[timestamp, 'iv']
    #         if denominator == 0:
    #             print(f"IC Denominator was 0. Invalidating this timestamp")
    #             self.df_futures.loc[timestamp, 'ic'] = 0
    #         else:
    #             self.df_futures.loc[timestamp, 'ic'] = numerator / denominator
    #         # self.df_futures.loc[timestamp, 'ic'] = numerator / max(0.01, denominator)
    #         print(f"IV({self.symbol}): {self.df_futures.loc[timestamp, 'iv']}")
    #         for symbol, component in self.components.items():
    #             print(f"IV({symbol}): {component.df_futures.loc[timestamp, 'iv']} | weight: {component.weight}")
    #         number_of_components = len(self.components)
    #         components = list(self.components.keys())
    #         for i in range(number_of_components):
    #             for j in range(i+1, number_of_components):
    #                 s1 = components[i]
    #                 s2 = components[j]
    #                 print(f"Correlation({s1}, {s2}) = {correlations.loc[s1, s2]}")
    #         print(f"Dirty Correlation(IC): {self.df_futures.loc[timestamp, 'ic']}")
    #         print('---------------------')
    #     print(f"Implied correlation data for {self.symbol} generated successfully")
    #     return

    def greeks(self, timestamp, opt_type, strike, log = True):
        futures_price = self.df_futures.loc[timestamp, self.ohlc.name] 
        strike = float(strike)
        options_price = self.get_opts_price(timestamp, opt_type, strike)
        expiry = self.get_expiry(timestamp)
        try:
            greeks = get_greeks(
                futures_price,
                strike,
                options_price,
                0,
                0,
                timestamp.date(),
                expiry.date(),
                opt_type)
            return greeks
        except Exception as e:
            if log:
                time_to_expiry = (expiry - timestamp).days
                print(f"Could not Generate Greeks (dp.ticker.greeks) for {self.symbol}")
                print(f"  {e}")
                print(f"  Sanity Parameters || spot(xx): {futures_price} | strike: {strike} | option price: {options_price} | time to expiry: {time_to_expiry}days | option type: {opt_type} | expiry: {expiry.date()} || error: {e}")
            # print(f"dates from dp.ticker.greeks: {pd.to_datetime(timestamp).date(), pd.to_datetime(expiry).date()}")

    def update_token(self, key, token):
        self.tokens[key] = token
    #
    def get_net_delta(ticker, timestamp):
        net_delta = 0
        for _, token in ticker.tokens.items():
            net_delta += token.stats.loc[timestamp, f'net_delta'] * ticker.lot_size_options
        return net_delta

    def hedge_futures_trade(self, lots, position, timestamp):
        self.hedging.loc[timestamp, 'lots'] = lots
        self.hedging.loc[timestamp, 'position'] = position
        self.hedging.loc[timestamp, 'futures_price'] = self.get_futures_price(timestamp)
        delta_added = lots * self.lot_size * position.value
        self.hedging.loc[timestamp, 'delta_added'] = delta_added

    
    def generate_vix_data(self):
        print(f"Generating {self.symbol} VIX data...")

        # Iterate over each timestamp in the futures data
        for timestamp, _ in self.df_futures.iterrows():
            futures_price = self.df_futures.loc[timestamp, self.ohlc.name]

            if pd.isna(futures_price) or futures_price <= 0:
                print(f"Skipping {timestamp}: Invalid futures price {futures_price}")
                continue
            print(timestamp)
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
                    liq_check = self.greeks(timestamp, option, strike, False)
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




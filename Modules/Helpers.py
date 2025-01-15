from Modules.enums import Option, LongShort, Leg, FNO, Period
from Modules import TradeAndLogics as TL
import pandas as pd
import numpy as np


def get_leg_future(ticker, timestamp, lots, position, asArray=False):
    legs = Leg(position, lots, FNO.FUTURES, None, ticker.get_futures_price(timestamp))
    if asArray:
        legs = [legs]
    return legs

def get_price_legs(*legs):
    price = 0
    for leg in legs:
        price += leg.Price * leg.Position.value
    return abs(price)


'v-v-v-v-v-v-  STRADDLES  -v-v-v-v-v-v-'
# Straddle that has a 
# call option of moneyness m and
# put option of moneyness m
def get_legs_straddle(ticker, timestamp, lots, position):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_call_price = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_put_price = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)
    legs = [Leg(position, lots, Option.Call, atm_call_strike, atm_call_price, 'StraddleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, atm_put_strike, atm_put_price, 'StraddlePut', ticker.lot_size)]
    return legs

# Straddle that has a 
# call option of strike close to strike_input and
# put option of strike close to strike_input


def get_legs_straddle_near(ticker, timestamp, lots, position, desired_strike):
    call_strike, _ = ticker.find_nearerst_strike(timestamp, desired_strike, Option.Call)
    put_strike, _ = ticker.find_nearerst_strike(timestamp, desired_strike, Option.Put)
    call_price = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, put_strike)
    legs = [Leg(position, lots, Option.Call, call_strike, call_price, 'StraddleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, put_strike, put_price, 'StraddlePut', ticker.lot_size)]
    return legs

'^-^-^-^-^-^-  STRADDLES  -^-^-^-^-^-^-'




'v-v-v-v-v-v-  STRANGLES  -v-v-v-v-v-v-'

# Strangle that has a 
# call option of moneyness m1
# put option of moneyness m2
def get_legs_strangle_moneyness(ticker, timestamp, lots, position, desired_moneyness_call, desired_moneyness_put = None):
    desired_moneyness_put = desired_moneyness_call if desired_moneyness_put is None else desired_moneyness_put
    call_strike, _ = ticker.find_moneyness_strike(timestamp, desired_moneyness_call, Option.Call)
    put_strike, _ = ticker.find_moneyness_strike(timestamp, desired_moneyness_put, Option.Put)
    call_price = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, put_strike)
    legs = [Leg(position, lots, Option.Call, call_strike, call_price, 'StrangleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, put_strike, put_price, 'StranglePut', ticker.lot_size)]
    return legs


def get_legs_strangle_WithFarOptionsOfPrice_ATMpriceXfactor(ticker, timestamp, lots, position, factor):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_call_price = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_put_price = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)
    
    desired_call_price = atm_call_price/factor
    desired_put_price = atm_put_price/factor
    
    call_strike = ticker.find_optprice_strike(timestamp, desired_call_price, Option.Call)
    put_strike = ticker.find_optprice_strike(timestamp, desired_put_price, Option.Put)
    call_price = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, put_strike)
    
    legs = [Leg(position, lots, Option.Call, call_strike, call_price, 'StrangleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, put_strike, put_price, 'StranglePut', ticker.lot_size)]
    return legs


# Strangle that has a 
# call option with strike of call_strike + call_price/factor and 
# put option with strike of put_strike - put_price/factor

def get_legs_strangle_StrikesDistantBy_OptPriceFactor(ticker, timestamp, lots, position, factor):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_price_call = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_price_put = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)

    desired_call_strike = atm_call_strike + atm_price_call/factor
    desired_put_strike = atm_put_strike - atm_price_put/factor

    call_strike, _ = ticker.find_nearest_strike(timestamp, desired_call_strike, Option.Call)
    put_strike, _ = ticker.find_nearest_strike(timestamp, desired_put_strike, Option.Put)
    call_price = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, put_strike)

    legs = [Leg(position, lots, Option.Call, call_strike, call_price, 'StrangleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, put_strike, put_price, 'StranglePut', ticker.lot_size)]
    return legs

# Strangle that has a 
# call option with strike of futures_price(1 + factor/100) and 
# put option with strike of futures_price(1 - factor/100)
def get_legs_strangle_ToStayIn_UnderlyingPriceRange(ticker, timestamp, lots, position, factor):
    price_underlying = ticker.get_futures_price(timestamp)

    desired_call_strike = price_underlying * (1 + factor/100)
    desired_put_strike = price_underlying * (1 - factor/100)

    call_strike, _ = ticker.find_nearest_strike(timestamp, desired_call_strike, Option.Call)
    put_strike, _ = ticker.find_nearest_strike(timestamp, desired_put_strike, Option.Put)
    call_price = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, put_strike)

    legs = [Leg(position, lots, Option.Call, call_strike, call_price, 'StrangleCall', ticker.lot_size), 
            Leg(position, lots, Option.Put, put_strike, put_price, 'StranglePut', ticker.lot_size)]
    return legs

'^-^-^-^-^-^-  STRANGLES  -^-^-^-^-^-^-'





'v-v-v-v-v-v-  IRON FLY  -v-v-v-v-v-v-'

# Iron Fly that has the far legs
# Call option that has a price = price(main_call)/factor
# Put option that has a price = price(main_put)/factor
def get_legs_ironfly_WithFarOptionsOfPrice_ATMpriceXfactor(ticker, timestamp, lots, position, factor):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_price_call = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_price_put = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)

    desired_far_price_call = atm_price_call/factor
    desired_far_price_put  = atm_price_put/factor

    far_call_strike = ticker.find_optprice_strike(timestamp, desired_far_price_call, Option.Call)
    far_put_strike = ticker.find_optprice_strike(timestamp, desired_far_price_put, Option.Put)
    far_call_price = ticker.get_opts_price(timestamp, Option.Call, far_call_strike)
    far_put_price = ticker.get_opts_price(timestamp, Option.Put, far_put_strike)
    
    legs = [
        Leg(position.opposite(), lots, Option.Call, far_call_strike, far_call_price, 'IronFly/FarCall', ticker.lot_size),
        Leg(position, lots, Option.Call, atm_call_strike, atm_price_call, 'IronFly/NearCall', ticker.lot_size),
        Leg(position, lots, Option.Put, atm_put_strike, atm_price_put, 'IronFly/NearPut', ticker.lot_size),
        Leg(position.opposite(), lots, Option.Put, far_put_strike, far_put_price, 'IronFly/FarPut', ticker.lot_size)
    ]
    return legs



# Iron Fly that has the far legs  
# call option that has a strike of call strike * (1 + factor%) 
# put option that has a strike of put strike * (1 - factor%)
def get_legs_ironfly_ToStayIn_UnderlyingPriceRange(ticker, timestamp, lots, position, factor):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_price_call = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_price_put = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)
    
    underlying_price = ticker.get_futures_price(timestamp)
    desired_far_call_strike = underlying_price * (1 + factor/100)
    desired_far_put_strike = underlying_price * (1 - factor/100)

    far_call_strike, _ = ticker.find_nearest_strike(timestamp, desired_far_call_strike, Option.Call)
    far_put_strike, _ = ticker.find_nearest_strike(timestamp, desired_far_put_strike, Option.Put)
    far_call_price = ticker.get_opts_price(timestamp, Option.Call, far_call_strike)
    far_put_price = ticker.get_opts_price(timestamp, Option.Put, far_put_strike)

    legs = [
        Leg(position.opposite(), lots, Option.Call, far_call_strike, far_call_price, 'IronFly/FarCall', ticker.lot_size),
        Leg(position, lots, Option.Call, atm_call_strike, atm_price_call, 'IronFly/NearCall', ticker.lot_size),
        Leg(position, lots, Option.Put, atm_put_strike, atm_price_put, 'IronFly/NearPut', ticker.lot_size),
        Leg(position.opposite(), lots, Option.Put, far_put_strike, far_put_price, 'IronFly/FarPut', ticker.lot_size)
    ]
    return legs


def get_legs_ironfly_StrikesDistantBy_OptPriceFactor(ticker, timestamp, lots, position, factor):
    atm_call_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
    atm_put_strike, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Put)
    atm_call_price = ticker.get_opts_price(timestamp, Option.Call, atm_call_strike)
    atm_put_price = ticker.get_opts_price(timestamp, Option.Put, atm_put_strike)

    desired_far_call_strike = atm_call_strike + atm_call_price/factor
    desired_far_put_strike = atm_put_strike - atm_put_price/factor

    far_call_strike, _ = ticker.find_nearest_strike(timestamp, desired_far_call_strike, Option.Call)
    far_put_strike, _ = ticker.find_nearest_strike(timestamp, desired_far_put_strike, Option.Put)
    far_call_price = ticker.get_opts_price(timestamp, Option.Call, far_call_strike)
    far_put_price = ticker.get_opts_price(timestamp, Option.Put, far_put_strike)
    legs = [
        Leg(position.opposite(), lots, Option.Call, far_call_strike, far_call_price, 'IronFly/FarCall', ticker.lot_size),
        Leg(position, lots, Option.Call, atm_call_strike, atm_call_price, 'IronFly/NearCall', ticker.lot_size),
        Leg(position, lots, Option.Put, atm_put_strike, atm_put_price, 'IronFly/NearPut', ticker.lot_size),
        Leg(position.opposite(), lots, Option.Put, far_put_strike, far_put_price, 'IronFly/FarPut', ticker.lot_size)
    ]
    return legs


'^-^-^-^-^-^-  IRON FLY  -^-^-^-^-^-^-'


'v-v-v-v-v-v-  IRON CONDOR  -v-v-v-v-v-v-'

# Iron Condor that has the far legs  
# call option that has a strike of call strike * (1 + factor%) 
# put option that has a strike of put strike * (1 - factor%)
def get_legs_ironcondor_underlyingmovement(ticker, timestamp, lots, position, moneyness, factor):
    call_strike, _ = ticker.find_moneyness_strike(timestamp, moneyness, Option.Call)
    put_strike, _ = ticker.find_moneyness_strike(timestamp, moneyness, Option.Put)
    price_put = ticker.get_opts_price(timestamp, Option.Put, put_strike)
    price_call = ticker.get_opts_price(timestamp, Option.Call, call_strike)

    desired_far_call_strike = call_strike * (1 + factor/100)
    desired_far_put_strike = put_strike * (1 - factor/100)
    
    far_call_strike = ticker.find_underlyingPriceMovement_strike(timestamp, 1, factor, Option.Call)
    far_put_strike = ticker.find_underlyingPriceMovement_strike(timestamp, 1, factor, Option.Put)
    far_call_price = ticker.get_opts_price(timestamp, Option.Call, far_call_strike)
    far_put_price = ticker.get_opts_price(timestamp, Option.Put, far_put_strike)
    legs = [
        Leg(position.opposite(), lots, Option.Call, far_call_strike, far_call_price, 'IronCondor/FarCall', ticker.lot_size),
        Leg(position, lots, Option.Call, call_strike, price_call, 'IronCondor/NearCall', ticker.lot_size),
        Leg(position, lots, Option.Put, put_strike, price_put, 'IronCondor/NearPut', ticker.lot_size),
        Leg(position.opposite(), lots, Option.Put, far_put_strike, far_put_price, 'IronCondor/FarPut', ticker.lot_size)
    ]
    return legs

# Iron Condor that has the far legs
# Call option that has a price = price(main_call)/factor
# Put option that has a price = price(main_put)/factor
def get_legs_ironcondor_price(ticker, timestamp, lots, position, moneyness, factor):
    if not isinstance(moneyness, int):
        moneyness_call = moneyness[0]
        moneyness_put = moneyness[1]
    else:
        moneyness_call = moneyness_put = moneyness
    call_strike, _ = ticker.find_moneyness_strike(timestamp, moneyness_call, Option.Call)
    put_strike, _ = ticker.find_moneyness_strike(timestamp, moneyness_put, Option.Put)
    price_call = ticker.get_opts_price(timestamp, Option.Call, call_strike)
    price_put = ticker.get_opts_price(timestamp, Option.Put, put_strike)

    far_call_strike = ticker.find_optprice_strike(timestamp, price_call/factor, Option.Call)
    far_put_strike = ticker.find_optprice_strike(timestamp, price_put/factor, Option.Put)
    far_call_price = ticker.get_opts_price(timestamp, Option.Call, far_call_strike)
    far_put_price = ticker.get_opts_price(timestamp, Option.Put, far_put_strike)
    
    legs = [
        Leg(position.opposite(), lots, Option.Call, far_call_strike, far_call_price, 'IronCondor/FarCall', ticker.lot_size),
        Leg(position, lots, Option.Call, call_strike, price_call, 'IronCondor/NearCall', ticker.lot_size),
        Leg(position, lots, Option.Put, put_strike, price_put, 'IronCondor/NearPut', ticker.lot_size),
        Leg(position.opposite(), lots, Option.Put, far_put_strike, far_put_price, 'IronCondor/FarPut', ticker.lot_size)
    ]
    return legs

'^-^-^-^-^-^-  IRON CONDOR  -^-^-^-^-^-^-'




def get_legs_synthetic_future(ticker, timestamp, lots, position):
    strike = ticker.get_strike_for_synthetic(timestamp)
    call_price = ticker.get_opts_price(timestamp, Option.Call, strike)
    put_price = ticker.get_opts_price(timestamp, Option.Put, strike)
    legs = [Leg(position, lots, Option.Call, strike, call_price, 'Hedging/Synthetic Future (Call)', ticker.lot_size), 
            Leg(position.opposite(), lots, Option.Put, strike, put_price, 'Hedging/Synthetic Future (Put)', ticker.lot_size)]
    return legs

# Normalize IV Horizontally 
# Useful for near expiry data
def get_normallized_iv(timestamp, ticker_near, ticker_next, log=True, log_sanity=False):
    if log:
        print(f"{ticker_near.symbol} | Normallized IV")
    
    iv_near = ticker_near.get_iv_at(timestamp, log_sanity)
    iv_next = ticker_next.get_iv_at(timestamp, log_sanity)
    if log:
        print(f">>  IV Near: {iv_near} || IV Next: {iv_next}")
        if log_sanity:
            print()
    
    expiry_near = pd.to_datetime(ticker_near.df_futures.loc[timestamp, 'expiry'])
    expiry_next = pd.to_datetime(ticker_next.df_futures.loc[timestamp, 'expiry'])
    if log:
        info = []
        if (not pd.isna(expiry_near)):
            info.append(f"Expiry Near: {expiry_near.strftime('%d/%b/%Y')}")
        if (not pd.isna(expiry_next)):
            info.append(f"Expiry Next: {expiry_next.strftime('%d/%b/%Y')}")
        print(f">> {' || '.join(info)}")
        if log_sanity:
            print()
    
    if (not pd.isna(expiry_next)):
        time_to_expiry_next = (expiry_next - timestamp).days
    else:
        time_to_expiry_next = np.inf
    if (not pd.isna(expiry_near)):
        time_to_expiry_near = (expiry_near - timestamp).days
    else:
        time_to_expiry_near = np.inf
    print(f">>  Time to Expiry Near: {time_to_expiry_near}days || Time to Expiry Next: {time_to_expiry_next}days")
    if log_sanity:
        print()
    
    if pd.isna(iv_near) and pd.isna(iv_next):
        if log:
            print(f"Invalidating timestamp because of NA IV's.")
        if log_sanity:
            print()
        return pd.NA
    
    if pd.isna(iv_near):
        weight1 = 0
        weight2 = 1
        normalized_iv = weight2 * iv_next
    elif pd.isna(iv_next):
        weight1 = 1
        weight2 = 0
        normalized_iv = weight1 * iv_near
    else:
        weight1 = (time_to_expiry_next - 30)/(time_to_expiry_next - time_to_expiry_near)
        weight2 = 1 - weight1
        normalized_iv = weight1 * iv_near + weight2 * iv_next
    if log:
        print(f">>  Weight1 Near: {weight1} || Weight2 Next: {weight2}")
        print(f">>  Normallized IV = {normalized_iv}")
        if log_sanity:
            print()
    return normalized_iv





# Visualise the entire state of a TOKEN from start to end
def get_summary_token(token, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = token.stats.index[0]
        end = token.stats.index[-1]
    df = token.stats.copy()
    df = df.loc[start:end]
    df['Underlying Price'] = token.data.loc[start:end]
    df['net_delta'] /= token.lot_size
    df['underlying_iv'] *= 100
    df.rename(columns = {
                        'instrument' : 'Instrument',
                        'net_lots' : 'Net Lots', 
                        'position' : 'Position', 
                        'running_pnl' : 'Running PNL', 
                        'net_delta' : 'Net Delta (L)', 
                        'net_vega' : 'Net Vega', 
                        'underlying_iv': 'Underlying IV (%)', 
                        'expenses' : 'Cumulative Expenses', 
                        'running_pnl_without_expenses' : 'Running PNL (without expenses)'
    }, inplace=True)

    return df

# Visualise the entire state of a TICKER from start to end
def get_summary_ticker(ticker, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = ticker.df_futures.index[0]
        end = ticker.df_futures.index[-1]
    timestamps = ticker.df_futures.loc[start:end].index

    data = {
        'Running PNL (without expenses)': [],
        'Cumulative Expenses': [],
        'Running PNL': [],
        'Net Delta (L)': [],
        'Net Vega': [],
        'Net Lots': [],
        'Active Position?': [],
        'Underlying IV (%) (Average of all contracts present)': []
    }
    for timestamp in timestamps:
        pnl_without_expenses = 0
        expenses = 0
        pnl = 0
        delta = 0
        vega = 0
        lots = 0
        underlying_iv = 0
        n_valid_iv_tokens = 0

        for _, token in ticker.tokens.items():
            pnl_without_expenses += token.stats.loc[timestamp, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']
            previous = token.stats.index.get_loc(start)
            previous = previous - 1 if previous else None
            expenses += token.stats.loc[timestamp, 'expenses']
            pnl += token.stats.loc[timestamp, 'running_pnl']
            delta += (token.stats.loc[timestamp, 'net_delta'])/token.lot_size
            vega += token.stats.loc[timestamp, 'net_vega'] 
            lots += token.stats.loc[timestamp, 'net_lots'] 
            if previous:
                expenses -= token.stats['expenses'].iloc[previous]
                pnl -= token.stats['running_pnl'].iloc[previous]
                delta -= token.stats['net_delta'].iloc[previous]/token.lot_size
                vega -= token.stats['net_vega'].iloc[previous]
                lots -= token.stats['net_lots'].iloc[previous]
            if token.stats.loc[timestamp, 'underlying_iv']:
                underlying_iv += token.stats.loc[timestamp, 'underlying_iv']
                n_valid_iv_tokens += 1

        active_position = int(abs(lots) > 0)
        
        data['Running PNL (without expenses)'].append(pnl_without_expenses) 
        data['Cumulative Expenses'].append(expenses) 
        data['Running PNL'].append(pnl) 
        data['Net Delta (L)'].append(delta)
        data['Net Vega'].append(vega)
        data['Net Lots'].append(lots)
        data['Active Position?'].append(active_position)
        if n_valid_iv_tokens:
            data['Underlying IV (%) (Average of all contracts present)'].append(underlying_iv * 100 / n_valid_iv_tokens)
        else:
            data['Underlying IV (%) (Average of all contracts present)'].append(pd.NA)
    result_df = pd.DataFrame(data, index=timestamps)
    
    return result_df

# Visualise the entire state of a PORTFOLIO OF TICKERS from start to end
def get_summary_portfolio(*tickers, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    if len(tickers) == 0:
        raise ValueError("Please provide the tickers")
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = tickers[0].df_futures.index[0]
        end = tickers[0].df_futures.index[-1]

    timestamps = tickers[0].df_futures.loc[start:end].index
    
    for ticker in tickers:
        if len(ticker.df_futures.loc[start:end].index) != len(timestamps):
            raise ValueError(f"Length of {ticker.symbol} data does not match with Length of {tickers[0].symbol} data")

    data = pd.DataFrame(0.0, index=timestamps, columns=[
            'Running PNL (without expenses)', 'Cumulative Expenses', 'Running PNL', 
            'Net Delta (L)', 'Net Vega', 'Net Lots', 'Active Position?'
        ])    
    for timestamp in timestamps:
        for ticker in tickers:
            for _, token in ticker.tokens.items():
                data.loc[timestamp, 'Running PNL (without expenses)'] += token.stats.loc[timestamp, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']
                data.loc[timestamp, 'Cumulative Expenses'] += token.stats.loc[timestamp, 'expenses']
                data.loc[timestamp, 'Running PNL'] += token.stats.loc[timestamp, 'running_pnl']
                data.loc[timestamp, 'Net Delta (L)'] += (token.stats.loc[timestamp, 'net_delta'])/token.lot_size
                data.loc[timestamp, 'Net Vega'] += token.stats.loc[timestamp, 'net_vega'] 
                data.loc[timestamp, 'Net Lots'] += token.stats.loc[timestamp, 'net_lots'] 
                previous = token.stats.index.get_loc(start)
                previous = previous - 1 if previous else None
                if previous:
                    data.loc[timestamp, 'Cumulative Expenses'] -= token.stats['expenses'].iloc[previous]
                    data.loc[timestamp, 'Running PNL'] -= token.stats['running_pnl'].iloc[previous]
                    # delta -= token.stats['net_delta'].iloc[previous]/token.lot_size
                    # vega -= token.stats['net_vega'].iloc[previous]
                # if token.stats.loc[timestamp, 'position'] != LongShort.Neutral and token.stats.loc[timestamp, 'position'] != 0:
                #     data[token.secDesc] = token.data.loc[timestamp]
            data.loc[timestamp, 'Active Position?'] = int(abs(data.loc[timestamp, 'Net Lots']) > 0)

    return data





# Excel Formatted Tickers Performance Sheet by Sheet
def zoom_tokens_performance_bar_by_bar(*portfolio, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = portfolio[0].df_futures.index[0]
        end = portfolio[0].df_futures.index[-1]

    timestamps = portfolio[0].df_futures.loc[start:end].index
    data_to_save_in_excel = {}
    symbol_wise_pnl = pd.DataFrame()
    token_wise_pnl = pd.DataFrame()
    for ticker in portfolio:
        symbol = ticker.symbol
        # zoom_symbol = pd.DataFrame(index=timestamps)
        # zoom_symbol[['Stock Running PNL (without expenses)', 'Cumulative Expenses', 
        #      'Stock Running PNL (with expenses)', 'Net Delta (L)']] = 0.0
        zoom_symbol = []

        token_legs = {}
        tokens = {}
        token_leg_count = 1
        for timestamp in timestamps:
            timestamp_data = {}
            timestamp_data['timestamp'] = timestamp
            timestamp_data['Stock Running PNL (without expenses)'] = 0
            timestamp_data['Cumulative Expenses'] = 0
            timestamp_data['Stock Running PNL (with expenses)'] = 0
            timestamp_data['Net Delta (L)'] = 0
            timestamp_data['Underlying IV'] = ticker.get_iv_at(timestamp) * 100
            timestamp_data['Underlying Price (XX)'] = ticker.get_futures_price(timestamp)
            for _, token in ticker.tokens.items():
                if (token.stats.loc[timestamp, 'position'] == LongShort.Neutral or token.stats.loc[timestamp, 'position'] == 0) and (token.secDesc not in token_legs.keys()):
                    continue
                if token.secDesc in token_legs.keys():
                    leg = token_legs[token.secDesc]
                else:
                    leg = token_leg_count
                    token_legs[token.secDesc] = token_leg_count
                    tokens[token.secDesc] = token
                    token_leg_count+= 1
                timestamp_data[f'LegName (Leg {leg})'] = token.legname
                timestamp_data['Stock Running PNL (without expenses)'] += token.stats.loc[timestamp, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']
                timestamp_data['Cumulative Expenses'] += token.stats.loc[timestamp, 'expenses'] 
                timestamp_data['Stock Running PNL (with expenses)'] += token.stats.loc[timestamp, 'running_pnl']
                timestamp_data['Net Delta (L)'] += token.stats.loc[timestamp, 'net_delta']/token.lot_size
                previous = token.stats.index.get_loc(start)
                previous = previous - 1 if previous else None
                if previous:
                    timestamp_data['Cumulative Expenses'] -= token.stats['expenses'].iloc[previous]
                    timestamp_data['Stock Running PNL (with expenses)'] -= token.stats['running_pnl'].iloc[previous]
                    # timestamp_data[timestamp, 'Net Delta (L)'] -= token.stats['net_delta'].iloc[previous]/token.lot_size

                if token.instrument == FNO.FUTURES:
                    timestamp_data[f'Futures Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
                    timestamp_data[f'Futures Price (Leg {leg})'] = token.data.loc[timestamp]
                    timestamp_data[f'Futures Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
                    timestamp_data[f'Futures PNL (Leg {leg})'] = token.stats.loc[timestamp, 'running_pnl_without_expenses']
                elif token.instrument == Option.Call:
                    timestamp_data[f'Call Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
                    timestamp_data[f'Call Price (Leg {leg})'] = token.data.loc[timestamp]
                    timestamp_data[f'Call Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
                    timestamp_data[f'Call PNL (Leg {leg})'] = token.stats.loc[timestamp, 'running_pnl_without_expenses']
                    timestamp_data[f'Call Strike (Leg {leg})'] = token.strike
                    timestamp_data[f'Call Delta (L) (Leg {leg})'] = token.stats.loc[timestamp, 'net_delta']/(token.lot_size) 
                elif token.instrument == Option.Put:
                    timestamp_data[f'Put Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
                    timestamp_data[f'Put Price (Leg {leg})'] = token.data.loc[timestamp]
                    timestamp_data[f'Put Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
                    timestamp_data[f'Put PNL (Leg {leg})'] = token.stats.loc[timestamp, 'running_pnl_without_expenses']
                    timestamp_data[f'Put Strike (Leg {leg})'] = token.strike
                    timestamp_data[f'Put Delta (L) (Leg {leg})'] = token.stats.loc[timestamp, 'net_delta']/(token.lot_size)
            zoom_symbol.append(timestamp_data)

        pnl_without_expense_symbol = 0 
        expenses_symbol = 0
        pnl_with_expense_symbol = 0
        for token in tokens.values():
            pnl_without_expense_token = (token.stats.loc[end, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']) 
            expenses_token = token.stats.loc[end, 'expenses'] 
            pnl_with_expense_token = pnl_without_expense_token 
            previous = token.stats.index.get_loc(start) 
            previous = previous - 1 if previous else None
            if previous:
                expenses_token -= token.stats['expenses'].iloc[previous]
                pnl_with_expense_token -= expenses_token

            pnl_without_expense_symbol += pnl_without_expense_token
            expenses_symbol += expenses_token
            pnl_with_expense_symbol += pnl_with_expense_token
            if token.instrument != FNO.FUTURES:
                token_id = f"{symbol}{int(token.strike)}{token.instrument.name}"
            else:
                token_id = f"{symbol}{token.instrument.name}"
            token_wise_pnl.loc[f'{token_id}', 'Leg Name'] = token.legname
            token_wise_pnl.loc[f'{token_id}', 'PNL (without expenses)'] = pnl_without_expense_token
            token_wise_pnl.loc[f'{token_id}', 'Expenses'] = expenses_token
            token_wise_pnl.loc[f'{token_id}', 'PNL (with expenses)'] = pnl_with_expense_token
    
        symbol_wise_pnl.loc[f'{symbol}', 'PNL (without expenses)'] = pnl_without_expense_symbol
        symbol_wise_pnl.loc[f'{symbol}', 'Expenses'] = expenses_symbol
        symbol_wise_pnl.loc[f'{symbol}', 'PNL (with expenses)'] = pnl_with_expense_symbol
        zoom_symbol = pd.DataFrame(zoom_symbol)
        zoom_symbol.set_index('timestamp', inplace=True)
        data_to_save_in_excel[f'{symbol}'] = zoom_symbol

    data_to_save_in_excel['Combined Summary'] = symbol_wise_pnl
    data_to_save_in_excel['Token Wise PNL'] = token_wise_pnl
    return data_to_save_in_excel


# # Excel Formatted Tickers Performance Sheet by Sheet
# def zoom_tokens_performance_bar_by_bar(*portfolio, **kwargs):
#     start = kwargs['start'] if 'start' in kwargs.keys() else None
#     end = kwargs['end'] if 'end' in kwargs.keys() else None
#     try:
#         start = pd.to_datetime(start)
#         end = pd.to_datetime(end)
#     except:
#         start = portfolio[0].df_futures.index[0]
#         end = portfolio[0].df_futures.index[-1]

#     timestamps = portfolio[0].df_futures.loc[start:end].index
#     data_to_save_in_excel = {}
#     symbol_wise_pnl = pd.DataFrame()
#     for ticker in portfolio:
#         symbol = ticker.symbol
#         zoom_symbol = pd.DataFrame(index=timestamps)
#         zoom_symbol[['Stock Running PNL (without expenses)', 'Cumulative Expenses', 
#              'Stock Running PNL (with expenses)', 'Net Delta (L)']] = 0.0
#         token_legs = {}
#         tokens = {}
#         token_leg_count = 1
#         for timestamp in timestamps:
#             timestamp_data[timestamp, 'Underlying IV'] = ticker.get_iv_at(timestamp) * 100
#             timestamp_data[timestamp, 'Underlying Price (XX)'] = ticker.get_futures_price(timestamp)
#             for _, token in ticker.tokens.items():
#                 if (token.stats.loc[timestamp, 'position'] == LongShort.Neutral or token.stats.loc[timestamp, 'position'] == 0) and (token.secDesc not in token_legs.keys()):
#                     continue
#                 if token.secDesc in token_legs.keys():
#                     leg = token_legs[token.secDesc]
#                 else:
#                     leg = token_leg_count
#                     token_legs[token.secDesc] = token_leg_count
#                     tokens[token.secDesc] = token
#                     token_leg_count+= 1
#                 timestamp_data[timestamp, f'LegName (Leg {leg})'] = token.legname
#                 timestamp_data[timestamp, 'Stock Running PNL (without expenses)'] += token.stats.loc[timestamp, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']
#                 timestamp_data[timestamp, 'Cumulative Expenses'] += token.stats.loc[timestamp, 'expenses'] 
#                 timestamp_data[timestamp, 'Stock Running PNL (with expenses)'] += token.stats.loc[timestamp, 'running_pnl']
#                 timestamp_data[timestamp, 'Net Delta (L)'] += token.stats.loc[timestamp, 'net_delta']/token.lot_size
#                 previous = token.stats.index.get_loc(start)
#                 previous = previous - 1 if previous else None
#                 if previous:
#                     timestamp_data[timestamp, 'Cumulative Expenses'] -= token.stats['expenses'].iloc[previous]
#                     timestamp_data[timestamp, 'Stock Running PNL (with expenses)'] -= token.stats['running_pnl'].iloc[previous]
#                     # timestamp_data[timestamp, 'Net Delta (L)'] -= token.stats['net_delta'].iloc[previous]/token.lot_size

#                 if token.instrument == FNO.FUTURES:
#                     timestamp_data[timestamp, f'Futures Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
#                     timestamp_data[timestamp, f'Futures Price (Leg {leg})'] = token.data.loc[timestamp]
#                     timestamp_data[timestamp, f'Futures Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
#                 elif token.instrument == Option.Call:
#                     timestamp_data[timestamp, f'Call Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
#                     timestamp_data[timestamp, f'Call Price (Leg {leg})'] = token.data.loc[timestamp]
#                     timestamp_data[timestamp, f'Call Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
#                     timestamp_data[timestamp, f'Call Strike (Leg {leg})'] = token.strike
#                     timestamp_data[timestamp, f'Call Delta (L) (Leg {leg})'] = token.stats.loc[timestamp, 'net_delta']/(token.lot_size) 
#                 elif token.instrument == Option.Put:
#                     timestamp_data[timestamp, f'Put Position (Leg {leg})'] = token.stats.loc[timestamp, 'position'].name
#                     timestamp_data[timestamp, f'Put Price (Leg {leg})'] = token.data.loc[timestamp]
#                     timestamp_data[timestamp, f'Put Lots (Leg {leg})'] = token.stats.loc[timestamp, 'net_lots']
#                     timestamp_data[timestamp, f'Put Strike (Leg {leg})'] = token.strike
#                     timestamp_data[timestamp, f'Put Delta (L) (Leg {leg})'] = token.stats.loc[timestamp, 'net_delta']/(token.lot_size)

#         pnl_without_expense_symbol = 0 
#         expenses_symbol = 0
#         pnl_with_expense_symbol = 0
#         for token in tokens.values():
#             pnl_without_expense_token = (token.stats.loc[end, 'running_pnl_without_expenses'] - token.stats.loc[start, 'running_pnl_without_expenses']) 
#             expenses_token = token.stats.loc[end, 'expenses'] 
#             pnl_with_expense_token = pnl_without_expense_token 
#             previous = token.stats.index.get_loc(start) 
#             previous = previous - 1 if previous else None
#             if previous:
#                 expenses_token -= token.stats['expenses'].iloc[previous]
#                 pnl_with_expense_token -= expenses_token

#             pnl_without_expense_symbol += pnl_without_expense_token
#             expenses_symbol += expenses_token
#             pnl_with_expense_symbol += pnl_with_expense_token
        
    
#         symbol_wise_pnl.loc[f'{symbol}', 'PNL (without expenses)'] = pnl_without_expense_symbol
#         symbol_wise_pnl.loc[f'{symbol}', 'Expenses'] = expenses_symbol
#         symbol_wise_pnl.loc[f'{symbol}', 'PNL (with expenses)'] = pnl_with_expense_symbol
#         data_to_save_in_excel[f'{symbol}'] = zoom_symbol

#     data_to_save_in_excel['Combined Summary'] = symbol_wise_pnl
#     return data_to_save_in_excel







# View the Chronologically ordered trades (Aggregate and Sheet Seggregated) performed in a Portfolio of securities
def get_trades_portfolio(*portfolio, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    if len(portfolio) == 0:
        raise ValueError("Please provide the tickers")
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = portfolio[0].df_futures.index[0]
        end = portfolio[0].df_futures.index[-1]

    arr = []
    for ticker in portfolio:
        arr += [{**trade, 'Symbol': ticker.symbol} for trade in ticker.Trades.tradesArr if pd.to_datetime(trade['Timestamp']) <= end and pd.to_datetime(trade['Timestamp']) >= start]

    df_trades = pd.DataFrame(arr)
    if df_trades.empty:
        raise ValueError("Trades is Empty")
    
    df_trades = df_trades.sort_values(by='Timestamp')
    cols = ['Timestamp', 'Symbol', 'Remarks']
    df_trades = df_trades.reindex(columns=cols+[col for col in df_trades.columns if col not in cols])
    df_trades.set_index('Timestamp', inplace=True)
    return df_trades

# View the Trades in a chronological order for any particular asset
def get_trades_ticker(ticker, **kwargs):
    start = kwargs['start'] if 'start' in kwargs.keys() else None
    end = kwargs['end'] if 'end' in kwargs.keys() else None
    try:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    except:
        start = ticker.df_futures.index[0]
        end = ticker.df_futures.index[-1]
    
    arr = [{**trade, 'Symbol': ticker.symbol} for trade in ticker.Trades.tradesArr if pd.to_datetime(trade['Timestamp']) <= end and pd.to_datetime(trade['Timestamp']) >= start]
    df_trades = pd.DataFrame(arr)
    if df_trades.empty:
        raise ValueError("Trades is Empty")
    df_trades = df_trades.sort_values(by='Timestamp')
    cols = ['Timestamp', 'Symbol', 'Remarks']
    df_trades = df_trades.reindex(columns=cols+[col for col in df_trades.columns if col not in cols])
    df_trades.set_index('Timestamp', inplace=True)
    return df_trades





# Get the Payoff Statistics Dataframe for a combination of legs
def get_payoff_stats(*legs, **kwargs):
    lotsize_include=False
    if 'lotsize' in kwargs:
        lotsize_include=True

    sum_premiums = 0
    for leg in legs:
        if leg.Instrument != FNO.FUTURES:
            sum_premiums += leg.Price * leg.Position.value
    x_max = 0
    x_min = 100000
    for leg in legs:
        if leg.Instrument == FNO.FUTURES:
            x_max = max(x_max, leg.Price)
            x_min = min(x_min, leg.Price)
        else:
            x_max = max(x_max, leg.Strike)
            x_min = min(x_min, leg.Strike)
    x_min -= abs(sum_premiums)
    x_max += abs(sum_premiums)
    left = max(int(x_min)/5, int(x_min)-50, 0)
    right = min(5*int(x_max), int(x_max)+50)
    n_samples = (right - left)*10
    x_values = np.linspace(int(left), int(right), int(n_samples)) 
    df = pd.DataFrame(index=x_values)
    for i, leg in enumerate(legs):
        y_values = leg.payoff(x_values, lotsize_include)
        df[f'{leg.Instrument.name} (Leg {i+1})'] = y_values
    df['Net Payoff'] = df.sum(axis=1)
    df_bep = df.copy()
    df_bep['Rounded Underlying'] = np.round(df.index)
    df_bep['Rounded Net Payoff'] = np.round(df_bep['Net Payoff'])
    bep = df_bep[df_bep['Rounded Net Payoff'] == 0].groupby('Rounded Underlying').mean()
    bep['Break Even Points'] = bep.index
    bep.reset_index(drop=True, inplace=True)
    return df, bep[['Break Even Points', 'Net Payoff']]




# Theta Neutral Entry (Index, Components)
def get_sqrtp_for_theta_neutral(timestamp, ticker):
    index = ticker
    if ticker.is_component:
        index = ticker.index
    numerator, denominator = index.weight * index.find_moneyness_strike(timestamp, 0, Option.Call)[0], 0
    for component in index.components.values():
        denominator += component.weight * component.find_moneyness_strike(timestamp, 0, Option.Call)[0]
    if denominator == 0:
        raise ValueError("Either No components found or the weighted sum of their ATM strikes is 0")
    return numerator/denominator


class Metrics:
    def __init__(self, input, fund_blocked, risk_free_rate):
        if isinstance(input, pd.Series) or isinstance(input, pd.DataFrame):
            self._daily_profit(input)
        elif isinstance(input, str):
            self._summary_pnl_path(input)
        else:
            print(f"Either Provide Daily Profit Pandas Series DateTimeIndexed using the ._daily_profit() method")
            print(f"or Provide Path to the Summary PNL CSV using the ._summary_pnl_path() containing the Running PNL (without expenses), Cumulative Expenses, Running PNL columns")
        self.fund_blocked = fund_blocked
        self.risk_free_rate = risk_free_rate

    def _daily_profit(self, daily_profit):
        self.daily_profit = daily_profit
    
    def _summary_pnl_path(self, path):
        from Modules.Data_Processing import get_continuous_excluding_market_holidays
        self.summary_pnl_path = path
        self.summary_pnl = pd.read_csv(self.summary_pnl_path, index_col=0, parse_dates=True)
        self.summary_pnl = self.normalize(self.summary_pnl)
        _, ci = get_continuous_excluding_market_holidays(self.summary_pnl)
        self.summary_pnl = self.summary_pnl.reindex(ci)
        
        self.summary_pnl = self.summary_pnl.ffill()
        self.summary_pnl['date'] = self.summary_pnl.index.date
        running_pnl_last = self.summary_pnl.groupby('date')['Running PNL (without expenses)'].last()
        cumulative_expenses_last = self.summary_pnl.groupby('date')['Cumulative Expenses'].last()
        daily_profit = running_pnl_last - running_pnl_last.shift(1).fillna(0)
        daily_profit -= (cumulative_expenses_last - cumulative_expenses_last.shift(1).fillna(0))
        self.daily_profit = pd.Series(daily_profit)
        self.daily_profit.index = pd.to_datetime(self.daily_profit.index)
        self.daily_profit.name = 'Daily PNL including Expenses'

    def normalize(self, df):
        df = df.copy()
        df = df[['Running PNL (without expenses)', 'Cumulative Expenses']]
        df = df.reset_index()
        ix_zeroes = df.index[df['Running PNL (without expenses)'] == 0]
        ix_last_vals = ix_zeroes.to_numpy() - 1
        if(ix_last_vals[0] == -1):
            ix_last_vals[0] = 0
        df['last_vals_pnl'] = None
        df['last_vals_expenses'] = None
        df.loc[ix_zeroes, 'last_vals_pnl'] = df.loc[ix_last_vals, 'Running PNL (without expenses)'].values
        df.loc[ix_zeroes, 'last_vals_expenses'] = df.loc[ix_last_vals, 'Cumulative Expenses'].values
        df.loc[0, ['last_vals_pnl', 'last_vals_expenses']] = 0
        df['last_vals_pnl'] = df['last_vals_pnl'].fillna(0)
        df['last_vals_pnl'] = df['last_vals_pnl'].cumsum()
        df['last_vals_expenses'] = df['last_vals_expenses'].fillna(0)
        df['last_vals_expenses'] = df['last_vals_expenses'].cumsum()
        df = df.groupby(df['index']).agg({
            'Running PNL (without expenses)': 'sum',
            'Cumulative Expenses': 'sum',
            'last_vals_pnl': 'first',
            'last_vals_expenses': 'sum'
        })
        df['Running PNL (without expenses)'] += df['last_vals_pnl']
        df['Cumulative Expenses'] += df['last_vals_expenses'] 
        df = df.drop(columns=['last_vals_pnl', 'last_vals_expenses'])
        df['Running PNL'] = df['Running PNL (without expenses)'] - df['Cumulative Expenses']
        return df
    
    def output_series(self, series, period=Period.Annual):
        if period == Period.Annual:
            series.index = series.index.strftime('%Y')
        if period == Period.Monthly:
            series.index = series.index.strftime('%B %Y')
        return series

    def scale_period(self, period):
        if period == Period.Annual:
            return np.sqrt(252)
        if period == Period.Monthly:
            return np.sqrt(21)
        return 1
    
    # def focus_time_interval(self, from_date, till_date):
    #     self.backup_daily_profit = self.daily_profit
    #     self.daily_profit = self.daily_profit[from_date:till_date]
    
    # def reset_time_interval(self):
    #     self.daily_profit = self.backup_daily_profit

    def returns(self, period=Period.Annual, all_data=False):
        period_profits = self.daily_profit.resample(period.value)
        if all_data:
            period_profits = self.daily_profit
        period_returns = period_profits.sum()/ self.fund_blocked * 100
        period_returns = np.round(period_returns, 2)
        if all_data:
            return period_returns
        period_returns.name = f"{period.name} Returns (%)"
        return self.output_series(period_returns, period)
        
    def sharpe(self, period=Period.Annual, all_data=False):
        if all_data:
            period_profits = self.daily_profit
        else:
            period_profits = self.daily_profit.resample(period.value)
        sharpe = (period_profits.mean() - self.risk_free_rate * self.fund_blocked/365)/period_profits.std()
        sharpe *= self.scale_period(period)
        sharpe = np.round(sharpe, 2)
        if all_data:
            return sharpe
        sharpe.name = f"{period.name} Sharpe"
        return self.output_series(sharpe, period)

    def sortino(self, period=Period.Annual, all_data=False):
        if all_data:
            period_profits = self.daily_profit
            negative_performance = period_profits[period_profits < 0]
        else:
            period_profits = self.daily_profit.resample(period.value)
            negative_performance = period_profits.apply(lambda group: group[group < 0])
            negative_performance = negative_performance.resample(period.value)
        sortino = (period_profits.mean() - self.risk_free_rate * self.fund_blocked/365)/negative_performance.std()
        sortino *= self.scale_period(period)
        sortino = np.round(sortino, 2)
        if all_data:
            return sortino
        sortino.name = f"{period.name} Sortino"
        return self.output_series(sortino, period)

    def information_ratio(self, period=Period.Annual, all_data=False):
        if all_data:
            period_profits = self.daily_profit
            points_with_trades = period_profits[period_profits != 0]
        else:
            period_profits = self.daily_profit.resample(period.value)
            points_with_trades = period_profits.apply(lambda group: group[group != 0])
            points_with_trades = points_with_trades.resample(period.value)
        ir = (period_profits.mean() - self.risk_free_rate * self.fund_blocked/365)/points_with_trades.std()
        ir *= self.scale_period(period)
        ir = np.round(ir, 2)
        if all_data:
            return ir
        ir.name = f"{period.name} Information Ratio"
        return self.output_series(ir, period)

    def drawdown_intervals(self, period=Period.Annual):
        period_profits = self.daily_profit.resample(period.value)
        cum_period_profits = period_profits.apply(lambda group: group.cumsum())
        cum_period_profits = cum_period_profits.resample(period.value)
        def helper(group):
            peaks = group.cummax()
            drawdowns = (group - peaks)/peaks * 100
            drawdowns.fillna(0)
            till_date = drawdowns.idxmin()
            peak_value = peaks[till_date]
            peaks_all = peaks[peaks == peak_value]
            peak_date = peaks_all.index[0]
            interval = (peak_date, till_date)
            return interval
        intervals = cum_period_profits.apply(helper)
        intervals.name = f"{period.name} Time Interval"
        return self.output_series(intervals, period)

    def max_drawdowns(self, period=Period.Annual):
        period_profits = self.daily_profit.resample(period.value)
        cum_period_profits = period_profits.apply(lambda group: group.cumsum())
        cum_period_profits = cum_period_profits.resample(period.value)
        def helper(group):
            peaks = group.cummax()
            drawdowns = (group - peaks)/peaks * 100
            drawdowns.fillna(0)
            return drawdowns
        drawdowns = cum_period_profits.apply(helper)
        drawdowns = drawdowns.resample(period.value)
        max_drawdown = drawdowns.min()
        max_drawdown = np.round(max_drawdown, 2)
        max_drawdown.name = f"{period.name} Max Drawdown (%)"
        return self.output_series(max_drawdown, period)
        
    def profits(self, period=Period.Annual):
        profits_series = self.daily_profit.resample(period.value).sum()
        profits_series = np.round(profits_series, 2)
        profits_series.name = f"{period.name} Profits"
        return self.output_series(profits_series, period)

    def cum_profit(self, period=Period.Annual, all_data=False):
        from Modules import Plot
        if all_data:
            period_profits = self.daily_profit
            cum_profits = period_profits.cumsum()
            df_cum_profits = cum_profits.to_frame()
            return (cum_profits, Plot.plot_df(df_cum_profits, *df_cum_profits.columns))
        period_profits = self.daily_profit.resample(period.value)
        data_series = {}
        graph_series = {}
        for key, value in period_profits:
            value = value.cumsum()
            data_series[key] = value
            value = value.to_frame()
            graph_series[key] = Plot.plot_df(value, *value.columns)
        data_series = pd.Series(data_series)
        graph_series = pd.Series(graph_series)
        return (self.output_series(data_series, period), self.output_series(graph_series, period))
    
    def risk_free_profit(self):
        rfp = len(self.daily_profit) * self.fund_blocked * self.risk_free_rate / 365
        rfp = np.round(rfp, 2)
        return rfp
    
    def win_percentage(self, period=Period.Annual):
        period_profits = self.daily_profit.resample(period.value)
        win_periods = period_profits.apply(lambda group: group[group > 0])
        win_periods = win_periods.resample(period.value)
        win_periods = win_periods.count()
        lose_periods = period_profits.apply(lambda group: group[group < 0])
        lose_periods = lose_periods.resample(period.value)
        lose_periods = lose_periods.count()
        total_periods = win_periods + lose_periods
        win_percentage = win_periods/ total_periods if total_periods > 0 else 0
        win_percentage = np.round(win_percentage * 100, 2)
        win_percentage.name = f"{period.name} Win Ratio (%)"
        return win_percentage


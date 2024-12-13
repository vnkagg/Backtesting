


import numpy as np
import pandas as pd
import os
import importlib
import sys
import copy
import diskcache


backtesting_path = r'C:\Users\vinayak\Desktop\Backtesting'
# print("path before", sys.path)
if backtesting_path not in sys.path:
    # print("The backtesting folder was not in the systems path")
    sys.path.append(backtesting_path)
# print("path after", sys.path)
# print()


# from scipy.optimize import minimize
from Modules import Plot
from Modules import Helpers
from Modules import Data as data
from Modules import Utility as util
from Modules import TradeAndLogics as TL
from Modules import Data_Processing as dp
from Modules.enums import Option, LongShort, DB, FNO, Leg, OHLC, PTSL


original_stdout = util.const(sys.stdout)
original_stdout.value



# start_date = '2023-07-01' 
# end_date = '2023-08-01'


# start_date = pd.to_datetime(start_date)
# end_date = pd.to_datetime(end_date)


# entry = '2023-10-16 09:15:00'
entry = input("Enter Entry Time: ")
entry = pd.to_datetime(entry)

# exit = '2023-10-26 11:30:00'
exit = input("Enter Exit Time: ")
exit = pd.to_datetime(exit)

from datetime import timedelta

start = pd.to_datetime(entry - timedelta(1))
end = pd.to_datetime(exit + timedelta(1))


ICView = input("Enter Short/ Long IC: ")
# ICView = 'Long'
if ICView == "Long":
    ICView = LongShort.Long
else:
    ICView = LongShort.Short

strategy_desc = fr"{entry.strftime('%d%b_%H_%M')}_{exit.strftime('%d%b_%H_%M')}"

folder_path = fr"C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\CheckTradable\Signaled_from_ICNearIV\{strategy_desc}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path) 

logs_path = fr'/c/Users/vinayak/Desktop/Backtesting/Dispersion/Optimizations/CheckTradable/Signaled_from_ICNearIV/{strategy_desc}/LOGS_Trading_Logic.txt'

file = open(fr'{folder_path}\LOGS_Trading_Logic.txt', 'w', buffering = 1)
print(f"tail -f {logs_path}")

notional_vega_buying = 10000
notional_vega_selling = 10000 # rs
# notional_vega_buying = int(input("Notional Vega Buy: "))
# notional_vega_selling = int(input("Notional Vega Sell: ")) # rs
# profit_target = int(input("Profit Target: "))
# stop_loss = int(input("Stop Loss: "))
profit_target = 100000
stop_loss = 100000




index_symbol = 'BANKNIFTY'
expiry_type_near = 'I'
expiry_type_next = 'II'
risk_free_rate = 0.1 # (10% interest rate)
timeframe = 1 # mins
look_back_window = 25*4
# Trade/ Strategy Parameters
buying_delta_threshold_per_lot = 5
selling_delta_threshold_per_lot = 1
zscore_threshold_long = 2
zscore_threshold_short = -2
ic_threshold_long = 0.8
ic_threshold_short = 0.2
epsilon = 0.1
moneyness_ironfly = 0
price_factor = 5


def tell_delta_threshold_per_lot(ticker):
    if ticker.intention == LongShort.Long:
        return buying_delta_threshold_per_lot
    return selling_delta_threshold_per_lot

def tell_notional_vega(ticker):
    if ticker.intention == LongShort.Long:
        return notional_vega_buying
    return notional_vega_selling





index_symbol = 'BANKNIFTY'
index_lot_size = 15



import Dispersion.DispersionAdjustedFunctionality as daf


basket = daf.RawWeightedPortfolio()
basket.insert('HDFCBANK', 550, 27.04)
basket.insert('ICICIBANK', 700, 23.03)
basket.insert('KOTAKBANK', 400, 11.72)
basket.insert('SBIN', 750, 11.27)
basket.insert('AXISBANK', 625, 11.18)


logs_near = open(fr'{folder_path}\LOGS_Fetching_Near_Month_Data.txt', 'w')
sys.stdout = logs_near
constituents_near = {}
ohlc = OHLC.close
index_near = daf.ticker(index_symbol, index_lot_size, True, start.date(), end.date(), expiry_type_near, True, timeframe, True, 0.1)
index_near.initializeDispersion(constituents_near, False, 1)
index_near.set_ohlc(ohlc)
index_near.set_intention(ICView)
for stock in basket.Symbols():
    constituents_near[stock] = daf.ticker(stock, basket.LotSize(stock), True, start.date(), end.date(), expiry_type_near, True, timeframe, False, 0.1)
    constituents_near[stock].initializeDispersion({}, True, basket.Weight(stock))
    constituents_near[stock].set_ohlc(ohlc)
    constituents_near[stock].set_intention(ICView.opposite())
sys.stdout = original_stdout.value
logs_near.close()


index = copy.deepcopy(index_near)
portfolio = [index, *[component for component in index.components.values()]]

def get_vega_legs(ticker, timestamp, *Legs):
    vega = 0
    for leg in Legs:
        if leg.Instrument == FNO.FUTURES:
            continue
        greeks = ticker.Greeks(timestamp, leg.Instrument, leg.Strike)
        if greeks is None:
            return None
        vega += greeks['vega'] * leg.Position.value
    return abs(vega)

def entry_signal(timestamp):
    if timestamp == entry:
        return True
    return False
def exit_signal(timestamp):
    if TL.isTodayAnyExpiry(timestamp, *portfolio):
        return True
    if timestamp == exit:
        return True
    return False


def get_lots_for_entry(ticker, timestamp, **kwargs):
    logging_information = {}
    legs = Helpers.get_legs_ironfly_WithFarOptionsOfPrice_ATMpriceXfactor(ticker, timestamp, 1, ticker.intention, price_factor)
    vega_ticker_ironfly = get_vega_legs(ticker, timestamp, *legs)
    if vega_ticker_ironfly is None:
        return None, None
    unweighted_vega = vega_ticker_ironfly * ticker.lot_size
    notional_vega = tell_notional_vega(ticker)
    idea_for_weight = ticker.weight
    logging_information = {
        'Lot Size': ticker.lot_size,
        'Weight': ticker.weight,
        'Vega per IronFly': vega_ticker_ironfly,
        'Vega per Lot IronFly': vega_ticker_ironfly * ticker.lot_size,
        'Target Notional Vega': notional_vega * ticker.weight
    }
    
    # if 'vega_neutral' in kwargs and kwargs['vega_neutral']:
    
    if 'ic_neutral' in kwargs and kwargs['ic_neutral'] == True:
        if ticker.is_component:
            idea_for_weight *= math.sqrt(ticker.get_ic_at(timestamp))

    elif 'theta_neutral' in kwargs and kwargs['theta_neutral'] == True:
        strike_ticker, _ = ticker.find_moneyness_strike(timestamp, 0, Option.Call)
        logging_information['ATM Strike'] = strike_ticker
        if ticker.is_component:
            import math
            idea_for_weight *= math.sqrt(ticker.get_ic_at(timestamp))

    lots_ticker = int(np.round(notional_vega * idea_for_weight/ unweighted_vega))
    logging_information['Lots to take Position'] = lots_ticker
    logging_information['Vega Satisfied'] = lots_ticker * unweighted_vega

    print(f">> {ticker.symbol}")
    for info_key, info_value in logging_information.items():
        print(f"  {info_key}: {info_value}")
    return lots_ticker


def take_dispersion_position(timestamp, remarks, ticker, lots):
    legs = []
    legs = Helpers.get_legs_ironfly_WithFarOptionsOfPrice_ATMpriceXfactor(ticker, timestamp, lots, ticker.intention, price_factor)
    for leg in legs:
        key = f'{ticker.symbol}_{leg.id()}'
        if key not in ticker.tokens.keys():
            ticker.tokens[key] = TL.Token(ticker, leg.Instrument, leg.Strike, leg.LegName)
        token = ticker.tokens[key]
        token.add_position(timestamp, lots, leg.Position)
    ticker.take_position(timestamp, remarks, *legs)


class DispersionPTSL(TL.PTSLHandling):
    def __init__(self, profit_target, stop_loss, *portfolio):
        super().__init__(profit_target, stop_loss, *portfolio)

    def is_valid(self, timestamp):
        # if PTSL is active, update to make sure it's correct
        if self.active_ptsl:
            self.update_validity(timestamp)
        return not self.active_ptsl   
    
    def update_validity(self, timestamp):
        if timestamp.date() != self.triggered_at.date():
            self.reset()
            print(f"New day, PTSL for Intraday Strategy is resetted")
        

def squareoff(timestamp, remarks, logging_token_update = True):
    print("************  SQUARE OFF BEGINS  *********** ")
    for ticker in portfolio:
        TL.squareoff_ticker(timestamp, remarks, ticker, logging_token_update)
    print("* SQUARE OFF TRADES SAVED, TOKENS IN TICKERS UPDATED")
    print("************  SQUARE OFF COMPLETE  *********** ")


def UpdateDispersionTickers(timestamp, remarks, hedge, logging_token_update, logging_hedging):
    hedge_using_synthetic_futures = True
    for ticker in portfolio:
        delta_threshold_per_lot = tell_delta_threshold_per_lot(ticker)
        TL.HandleUpdate(ticker, timestamp, remarks, hedge, delta_threshold_per_lot, hedge_using_synthetic_futures, logging_token_update, logging_hedging)
    

def zoom_dispersion_trade(start, end, file_name):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    performance = Helpers.zoom_tokens_performance_bar_by_bar(*portfolio, start=start, end=end)
    Plot.save_df_to_excel(performance, file_name)

ic = pd.read_csv(r'C:\Users\vinayak\Desktop\Backtesting\Dispersion\Optimizations\ICs\IC_NearIV.csv', index_col=0, parse_dates=True)


def visualise_dispersion_trade(start, end, file_name):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    result_df = Helpers.get_summary_portfolio(*portfolio, start=start, end=end)

    result_df['IC'] = ic.loc[start:end, 'ic']

    fig = Plot.plot_df(result_df, *(result_df.columns))
    Plot.save_plot(fig, file_name)


def save_dispersion_trades(start, end, file_name):
    trades_dict = {}
    trades_dict['All Trades'] = Helpers.get_trades_portfolio(*portfolio, start=start, end=end)
    for ticker in portfolio:
        trades_dict[ticker.symbol] = Helpers.get_trades_ticker(ticker, start=start, end=end)
    Plot.save_df_to_excel(trades_dict, file_name)


sys.stdout = file

for ticker in portfolio:
    ticker.reset_trades()

TrackPTSL = DispersionPTSL(profit_target, stop_loss, *portfolio)
timestamps = index.timestamps
current_position = None
trade_start_date, trade_end_date, trade_count = None, None, 0
for timestamp in timestamps:
    print()
    print(f"Timestamp: {timestamp}")
    #######################################################################################################################################################
    if TL.check_existing_position(index):
        
        print(f"Existing Position Check: TRUE | IC position: {current_position}")
        
        # 3:20 (Market Close) SQUAREOFF
        if exit_signal(timestamp):
            print("************  MARKET CLOSE  *********** ")
            UpdateDispersionTickers(timestamp, 'NoHedging', False, True, False)
            squareoff(timestamp, f'Exit Square Off', True)
            trade_end_date = timestamp
            trade_count+=1
            zoom_dispersion_trade(trade_start_date, trade_end_date, fr"{folder_path}\Zoom_Trade.xlsx")
            visualise_dispersion_trade(trade_start_date, trade_end_date, fr"{folder_path}\Visualise_Trade.html")
            continue

        # PROFIT TARGET AND STOP LOSS SQUAREOFF
        if TrackPTSL.is_valid(timestamp) and TrackPTSL.status(timestamp) != PTSL.Valid:
            print("************  TrackPTSL Trigger hit *********** ")
            print(f"{TrackPTSL.nature} Square off at NetPnl of {TrackPTSL.pnl_last_trade}")
            UpdateDispersionTickers(timestamp, 'NoHedging', False, True, False)
            squareoff(timestamp, f'{TrackPTSL.nature} Square Off', True)
            trade_end_date = timestamp
            trade_count+=1
            zoom_dispersion_trade(trade_start_date, trade_end_date, fr"{folder_path}\Zoom_Trade.xlsx")
            visualise_dispersion_trade(trade_start_date, trade_end_date, fr"{folder_path}\Visualise_Trade.html")
            continue
            
        # IF NO NEED TO SQUAREOFF, HEDGE IF NEEDED
        UpdateDispersionTickers(timestamp, 'Delta Hedging using Synthetic Futures', True, True, True)
        print("========================================================================================================================================================================")
        continue
    #######################################################################################################################################################
    
    
    #######################################################################################################################################################
    if entry_signal(timestamp):
        print("************  ENTRY TIME REACHED  ************  ")
        UpdateDispersionTickers(timestamp, "NoHedging", False, False, False)
        if TL.isTodayAnyExpiry(timestamp, *portfolio):
            print("Expiry Day today. Trading on Expiry Day is not Allowed")
            continue

        if not TrackPTSL.is_valid(timestamp):
            print(f"{TrackPTSL.nature} was Triggered at {TrackPTSL.triggered_at.strftime('%H:%M on %d/%b/%Y')} and Z-Score has not yet reverted back to mean (abs(z) is not yet <= {epsilon})")
            continue
        
        for ticker in portfolio:
            if ticker.is_index:
                ticker.set_intention(ICView)
            else:
                ticker.set_intention(ICView.opposite())
            lots_ticker = get_lots_for_entry(ticker, timestamp)
            take_dispersion_position(timestamp, f'{ICView.name} IC', ticker, lots_ticker)

        TrackPTSL.fresh_trade(timestamp)
        current_position = ICView
        trade_start_date = timestamp
        print(f"{ICView} IC Trade executed")

        continue
    #######################################################################################################################################################

    UpdateDispersionTickers(timestamp, "NoHedging", False, True, False)
    print("========================================================================================================================================================================")
    

file.close()
sys.stdout = original_stdout.value


# visualise_dispersion_trade(
#     start=index.df_futures.index[0], 
#     end=index.df_futures.index[-1], 
#     file_name='PLOT_Complete_Period_Summary.html'
# )


save_dispersion_trades(
    start=index.df_futures.index[0], 
    end=index.df_futures.index[-1], 
    file_name=fr'{folder_path}\Info_Trades_for_{strategy_desc}.xlsx'
)
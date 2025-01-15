import numpy as np
import pandas as pd
from Modules.enums import Option, FNO, LongShort, Leg, PTSL
from Modules import Helpers

class Token:
    def __init__(self, ticker, instrument, strike, legname = ''):
        self.ticker = ticker
        self.timestamps = ticker.timestamps
        self.lot_size = self.ticker.lot_size
        self.instrument = instrument
        self.expiry = None
        self.strike = strike
        self.legname = legname
        self.instrument = instrument
        self.lots = 0
        self.expenses = 0
        self.running_pnl = 0
        self.position = LongShort.Neutral
        self.ohlc = ticker.ohlc
        self.stats = pd.DataFrame(0.0, 
                                  index=ticker.df_futures.index,
                                  columns=['instrument', 
                                           'net_lots', 
                                           'position', 
                                           'net_delta', 
                                           'net_vega', 
                                           'underlying_iv', 
                                           'running_pnl_without_expenses', 
                                           'expenses', 
                                           'running_pnl'])
        self.stats['position'] = self.stats['position'].astype(object)
        self.stats['instrument'] = self.stats['instrument'].astype(object)
        self.stats['position'] = LongShort.Neutral
        self.slippage = 7.5
        if self.instrument != FNO.FUTURES:
            # OPTIONS
            self.transaction_costs = 11.5
            self.secDesc = f'{ticker.symbol}_{instrument.name}_{strike}'
            self.stats['instrument'] = self.instrument
            self.data = ticker.get_opts(instrument)[strike]
        else:
            # FUTURES
            self.transaction_costs = 2.5
            self.secDesc = f'{ticker.symbol}_{instrument.name}'
            self.stats['instrument'] = self.instrument
            self.data = ticker.df_futures[ticker.ohlc.name]


    def calc_expenses(self, transaction_value):
        return transaction_value * (self.slippage + self.transaction_costs) * 0.01 * 0.01

    def add_position(self, timestamp, lots, position):
        if self.expiry == None:
            self.expiry = self.ticker.get_expiry(timestamp)
        updated_net_lots = self.lots * self.position.value + lots * position.value
        self.lots = abs(updated_net_lots)
        self.position = LongShort(np.sign(updated_net_lots))
        transaction_value = abs(lots) * self.lot_size * self.data.loc[timestamp]
        self.expenses += self.calc_expenses(transaction_value)
        self.stats.loc[timestamp, 'expenses'] = self.expenses
        self.stats.loc[timestamp, 'running_pnl'] -= self.calc_expenses(transaction_value)
        self.stats.loc[timestamp, 'position'] = self.position
        self.stats.loc[timestamp, 'net_lots'] = self.lots

    def update_df(self, timestamp, logging=True):
        time_ix = self.timestamps.get_loc(timestamp)
        if pd.isna(self.data.loc[timestamp]):
            if self.instrument == FNO.FUTURES:
                print(f"{self.ohlc.name} Price of Futures is Null")
            else:
                print(f"{self.ohlc.name} Price of {self.instrument.name} Option of Strike {self.strike} is Null")
            return
        if pd.isna(self.data.iloc[time_ix-1]):
            if self.instrument == FNO.FUTURES:
                print(f"{self.ohlc.name} Price of Futures is Null")
            else:
                print(f"{self.ohlc.name} Price of {self.instrument.name} Option of Strike {self.strike} is Null")
            return

        pnl_this_moment = (self.data.loc[timestamp] - self.data.iloc[time_ix-1]) * (self.position.value) * self.lots * self.lot_size
        self.running_pnl += pnl_this_moment
        self.stats.loc[timestamp, 'position'] = self.position
        self.stats.loc[timestamp, 'net_lots'] = self.lots
        self.stats.loc[timestamp, 'expenses'] = self.expenses
        self.stats.loc[timestamp, 'running_pnl_without_expenses'] = self.running_pnl
        self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl - self.expenses
        if self.position == LongShort.Neutral or self.position == 0:
            self.stats.loc[timestamp, 'net_delta'] = 0
            self.stats.loc[timestamp, 'net_vega'] = 0
            self.stats.loc[timestamp, 'underlying_iv'] = 0
            return 
        if self.instrument != FNO.FUTURES:
            greeks = self.ticker.Greeks(timestamp, self.instrument, self.strike, False, logging)
            if greeks is None:
                if logging:
                    print(f"  Token did not receive Greeks")
                    print(f"  Delta and Vega Information is filled with previously available greeks (timestamp - 1)")
                    self.stats.loc[timestamp, 'net_delta'] = self.stats['net_delta'].iloc[time_ix-1]
                    self.stats.loc[timestamp, 'net_vega'] = self.stats['net_vega'].iloc[time_ix-1]
                return
            self.stats.loc[timestamp, 'net_delta'] = greeks['delta'] * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'net_vega'] = greeks['vega'] * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'underlying_iv'] = greeks['implied_volatility']
        else:
            self.stats.loc[timestamp, 'net_delta'] = 1 * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'net_vega'] = 0
            self.stats.loc[timestamp, 'underlying_iv'] = 0


class PTSLHandling:
    def __init__(self, profit_target, stop_loss, *portfolio):
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.portfolio = portfolio
        self.timestamps = portfolio[0].timestamps
        self.active_ptsl = False
        self.nature = PTSL.Valid
        self.entry_timestamp = None
        self.entry_timestamp_minus_1 = None
        self.triggered_at = None
        self.pnl_last_trade = None

    def trigger(self, timestamp):
        self.active_ptsl = True
        self.triggered_at = timestamp

    def reset(self):
        self.nature = PTSL.Valid
        self.active_ptsl = False
        self.triggered_at = None
    
    def fresh_trade(self, timestamp):
        self.active_ptsl = False
        self.entry_timestamp = timestamp
        ix = self.portfolio[0].df_futures.index.get_loc(timestamp) - 1
        self.entry_timestamp_minus_1 = self.portfolio[0].df_futures.index[ix]
        self.nature = PTSL.Valid
        self.pnl_last_trade = 0
    
    def helper_function(self, timestamp, column):
        value = 0
        for ticker in self.portfolio:
            for token in ticker.tokens.values():
                value += token.stats.loc[timestamp, column]
        return value

    def get_net_profit(self, timestamp):
        running_pnl_without_expenses_entry = self.helper_function(self.entry_timestamp, 'running_pnl_without_expenses')
        cumulative_expenses_entry = self.helper_function(self.entry_timestamp_minus_1, 'expenses')
        running_pnl_without_expenses_exit = self.helper_function(timestamp, 'running_pnl_without_expenses')
        cumulative_expenses_exit = self.helper_function(timestamp, 'expenses')
        
        net_profit = (running_pnl_without_expenses_exit - running_pnl_without_expenses_entry) - (cumulative_expenses_exit - cumulative_expenses_entry)
        return net_profit

    def status(self, timestamp):
        net_profit = self.get_net_profit(timestamp)
        if net_profit >= self.profit_target:
            self.trigger(timestamp)
            self.nature = PTSL.ProfitTarget
            self.pnl_last_trade = net_profit
            return self.nature
        if net_profit <= -self.stop_loss:
            self.trigger(timestamp)
            self.nature = PTSL.StopLoss
            self.pnl_last_trade = net_profit
            return self.nature
        return PTSL.Valid
    
    def handle_square_off(self, timestamp):
        self.active_ptsl = False
    


class Martingale:
    def __init__(self, min_interval, min_change, max_bet=8):
        self.lots = 0
        self.multiplier = 1
        self.min_change = min_change
        self.min_interval = min_interval
        self.data = []
        self.max_bet = max_bet

    def entry(self, timestamp, lots, position, value):
        self.last_entry_time = timestamp
        self.lots += lots
        self.direction = position.value
        self.last_entry_value = value
        self.prev_value = value
        self.prev_slope = 0
        self.data.append({'timestamp': timestamp, 'value': value})

    def is_valid(self, timestamp, value):
        from Modules import Data_Processing as dp
        if((self.direction) * (value) > self.direction * self.last_entry_value):
            return False
        value_shift_check = (self.direction) * (self.last_entry_value - value) >= self.min_change
        time_shift_check =  dp.trading_minutes_between(self.last_entry_time, timestamp) >= self.min_interval
        return (value_shift_check and time_shift_check)

    def bet(self, timestamp, value): 
        self.data.append({'timestamp': timestamp, 'value': value})
        if not self.is_valid(timestamp, value):
            return 0
        lots = 0    
        curr_slope = value - self.prev_value
        if (curr_slope * self.prev_slope < 0):
            self.multiplier *= 2
            if self.multiplier <= self.max_bet:
                lots = self.lots * self.multiplier
                self.last_entry_time = timestamp
                self.lots += lots
                self.last_entry_value = value
        self.prev_value = value
        self.prev_slope = curr_slope
        if lots > 0:
            self.data[-1]['Piled a Bet'] = lots
        return lots

    def show_history(self, plot=True):
        df = pd.DataFrame(self.data)
        df = df.set_index('timestamp')
        df = df.ffill()
        df = df.fillna(0)
        if plot:
            from Modules import Plot
            return Plot.plot_df(df, *df.columns)
        return df

'-v-v-v-v-v-v-v-v-v-v-v- CHECKS/ TRADE INFO/ FLAGS -v-v-v-v-v-v-v-v-v-v-v-'
def check_existing_position(ticker):
    pos = False
    for _, token in ticker.tokens.items():
        pos = pos | (token.position.value != 0) 
    return pos


def isTodayAnyExpiry(timestamp, *portfolio):
    is_expiry = False
    for ticker in portfolio:
        is_expiry = is_expiry | ticker.is_expiry(timestamp)
    return is_expiry

def isLastNdays(timestamp, n, *portfolio):
    isLastN = False
    for ticker in portfolio:
        expiry = ticker.get_expiry_at(timestamp)
        isLastN = isLastN | ((expiry - timestamp).days <= n)
    return isLastN

'-^-^-^-^-^-^-^-^-^-^-^- CHECKS/ TRADE INFO/ FLAGS -^-^-^-^-^-^-^-^-^-^-^-'






'-v-v-v-v-v-v-v-v-v-v-v- TRADE EXECUTIONS/ UPDATIONS -v-v-v-v-v-v-v-v-v-v-v-'

def squareoff_ticker(timestamp, remarks, ticker, logging_token_update):
    # time_ix = ticker.timestamps.get_loc(timestamp)
    legs = []
    if(not check_existing_position(ticker)):
        print(f">> {ticker.symbol} does not hold any position")
        print()
        return
    print(f">> Squaring off {ticker.symbol}")
    for key, token in ticker.tokens.items():
        # token.update_df(timestamp, logging_token_update)
        if token.position == LongShort.Neutral:
            continue
        legs.append(
            Leg(
                token.position.opposite(),
                token.lots,
                token.instrument,
                token.strike,
                token.data.loc[timestamp],
                token.legname
            )
        )
        token.add_position(timestamp, token.lots, token.position.opposite())
    if len(legs):
        ticker.take_position(timestamp, remarks, *legs)
        for leg in legs:
            if leg.Instrument == FNO.FUTURES:
                print(f"  SQUARED OFF {ticker.symbol, leg.Instrument.name}")
            else:
                print(f"  SQUARED OFF {ticker.symbol, leg.Instrument.name, int(leg.Strike)}")

        print()
    else:    
        print(f" Did not find any Legs for {ticker.symbol}")


def HandleUpdate(ticker, timestamp, remarks, hedge = True, delta_threshold_per_lot = 1e9, hedge_using_syn = True, logging_token_update=True, logging_hedging=True):
    # time_ix = ticker.timestamps.get_loc(timestamp)
    print()
    print(f">> {ticker.symbol}")
    active_tokens = {}
    inactive_tokens = {}
    for key, token in ticker.tokens.items():
        if token.position.value != 0:
            active_tokens[key] = token
        else:
            inactive_tokens[key] = token

    print(f"  ----- Active Tokens -----")
    for key, token in active_tokens.items():
        token.update_df(timestamp, logging_token_update)
        if not logging_token_update:
            continue
        if token.instrument != FNO.FUTURES:
            print(f"{token.ticker.symbol, token.instrument.name, float(token.strike)} | position = {token.position.name} | Delta (L) = {np.round(token.stats.loc[timestamp, 'net_delta']/ticker.lot_size, 2)} (position accounted) | Lots in position: {token.lots}")
        else:
            print(f"{token.ticker.symbol, token.instrument.name}  | position = {token.position.name} | Delta (L) = {np.round(token.stats.loc[timestamp, 'net_delta']/ ticker.lot_size, 2)} | Lots in position: {token.lots}")
    print(f"  ----- Inactive Tokens -----")
    for key, token in inactive_tokens.items():
        token.update_df(timestamp, False)
        if not logging_token_update:
            continue
        if token.instrument != FNO.FUTURES:
            print(f"{token.ticker.symbol, token.instrument.name, float(token.strike)} | position = {token.position.name}")
        else:
            print(f"{token.ticker.symbol, token.instrument.name} | position = {token.position.name}")
    print(f"  -- Hedging Sanity/Logs --")
    if hedge:
        if not check_existing_position(ticker):
            return
        if hedge_using_syn:
            hedge_delta_using_synthetic_futures(timestamp, remarks, ticker, delta_threshold_per_lot, logging_hedging)
        else:
            hedge_delta_using_futures(timestamp, remarks, ticker, delta_threshold_per_lot, logging_hedging)
    print()


'-^-^-^-^-^-^-^-^-^-^-^- TRADE EXECUTIONS/ UPDATIONS -^-^-^-^-^-^-^-^-^-^-^-'







def hedge_delta_using_futures(timestamp, remarks, ticker, delta_threshold_per_lot, logging = True):
    print(f"-------  Hedging Check for {ticker.symbol}  -----------")
    net_delta = ticker.get_net_delta(timestamp)
    delta_per_lot = net_delta/ticker.lot_size 
    # delta_threshold_per_lot = ticker.delta_threshold_per_lots()
    print(f"Delta (per Lot): {np.round(delta_per_lot, 2)} | Threshold Lots: {delta_threshold_per_lot}")
    if abs(delta_per_lot) >= delta_threshold_per_lot:
        position = LongShort(np.sign(delta_per_lot)).opposite()
        lots_futures = int(np.floor(abs(delta_per_lot)))
        legname = 'Hedging/Futures'
        leg = Leg(
            position, 
            lots_futures,
            FNO.FUTURES,
            None,
            legname
        )
        key = f'{ticker.symbol}_{leg.id()}'
        if key not in ticker.tokens.keys():
            ticker.tokens[key] = Token(ticker, FNO.FUTURES, None, legname)
        token = ticker.tokens[key]
        token.add_position(timestamp, lots_futures, position)
        ticker.take_position(timestamp, remarks, leg)
        print(f"Added {lots_futures * position.value} Lots of Delta by going {position.name} {lots_futures} lots of future. New Delta = {np.round(delta_per_lot, 2) + lots_futures * position.value}")
        print(f"-------  Hedged Delta {ticker.symbol}  -------")
    else:
        print(f"-------  Did not Hedge {ticker.symbol}  -------")


def hedge_delta_using_synthetic_futures(timestamp, remarks, ticker, delta_threshold_per_lot, logging = True):
    print(f"-------  Hedging Check for {ticker.symbol}  -----------")
    net_delta = ticker.get_net_delta(timestamp)
    delta_per_lot = net_delta/ticker.lot_size 
    # delta_threshold_per_lot = ticker.get_delta_threshold_per_lot()
    print(f"Delta (per Lot): {np.round(delta_per_lot, 2)} | Threshold Lots: {delta_threshold_per_lot}")
    if abs(delta_per_lot) >= delta_threshold_per_lot:
        position = LongShort(np.sign(delta_per_lot)).opposite()
        lots_futures = int(np.floor(abs(delta_per_lot)))
        legs = Helpers.get_legs_synthetic_future(ticker, timestamp, lots_futures, position)
        logging_info = []
        for leg in legs:
            key = f'{ticker.symbol}_{leg.id()}'
            if key not in ticker.tokens.keys():
                ticker.tokens[key] = Token(ticker, leg.Instrument, leg.Strike, leg.LegName)
            token = ticker.tokens[key]
            token.add_position(timestamp, leg.Lots, leg.Position)
            logging_info.append(f"{leg.Position.name} {lots_futures} of {leg.Instrument.name}")
        ticker.take_position(timestamp, remarks, *legs)
        print(f"Added {lots_futures * position.value} Lots of Delta by going {", and ".join(logging_info)}. New Delta = {np.round(delta_per_lot, 2) + lots_futures * position.value}")
        print(f"-------  Hedged Delta {ticker.symbol}  -------")
    else:
        print(f"-------  Did not Hedge {ticker.symbol}  -------")





def TEMPLATE_OneMonth_or_ONEDAY():
    def get_tickers_porfolio(start, end):
        constituents = {}
        ohlc = OHLC.close
        index = daf.ticker(index_symbol, index_lot_size, True, start, end, expiry_type, True, timeframe, True, 0.1)
        index.initializeDispersion(constituents, False, 1)
        index.set_ohlc(ohlc)
        index.set_intention(ICView)
        for stock in basket.Symbols():
            constituents[stock] = daf.ticker(stock, basket.LotSize(stock), True, start, end, expiry_type, True, timeframe, False, 0.1)
            constituents[stock].initializeDispersion({}, True, basket.Weight(stock))
            constituents[stock].set_ohlc(ohlc)
            constituents[stock].set_intention(ICView.opposite())
        portfolio = [index] + [component for component in index.components.values()]
        return portfolio
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='ME') # or B
    for month in date_range:
        start, end = month.replace(day=1).date(), month.date()
        portfolio = get_tickers_porfolio()

def TEMPLATE_GeneralFlow_TradingLoop():

    # Build stratAdjustedFunctionality Module
    # handle meta data there. like what to do if its a stock or index. 
    # delta threshold if its long or short

    def entry_signal(timestamp):
        #  Global Variable Portfolio
        if timestamp.hour == 9 and timestamp.minute == 20:
            return True
        pass
    def exit_signal(timestamp):
        #  Global Variable Portfolio
        if timestamp.hour == 15 and timestamp.minute == 20:
            return True
        pass
    def Update_Strategy_Tickers(timestamp, remarks, hedge, log_token_update, log_hedge):
        for ticker in portfolio:
            HandleUpdate(
                ticker, 
                timestamp, 
                remarks, 
                hedge, 
                logging_token_update, 
                logging_hedging
            )
            continue
        pass
    def SquareOff_Portfolio(timestamp, remarks, logging_sqoff_info):
        print("************  SQUARE OFF BEGINS  *********** ")
        for ticker in portfolio:
            TL.squareoff_ticker(timestamp, remarks, ticker, logging_sqoff_info)
        print("* SQUARE OFF TRADES SAVED, TOKENS IN TICKERS UPDATED")
        print("************  SQUARE OFF COMPLETE  *********** ")
    class StrategyPTSL(TL.PTSLHandling):
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
    def get_lots_for_entry(ticker, timestamp, **kwargs):
        legs = Helpers.get_legs_spread(ticker, timestamp)
        logging_information = {
            'Lot Size': ticker.lot_size,
            # Other parameters
        }
        # do computation and find lots
        print(f">> {ticker.symbol}")
        for info_key, info_value in logging_information.items():
            print(f"  {info_key}: {info_value}")
        return lots_ticker
    def take_strategy_position(timestamp, remarks, ticker, lots):
        legs = Helpers.get_legs()
        for leg in legs:
            key = f'{ticker.symbol}_{leg.id()}'
            if key not in ticker.tokens.keys():
                ticker.tokens[key] = TL.Token(ticker, leg.Instrument, leg.Strike, leg.LegName)
            token = ticker.tokens[key]
            token.add_position(timestamp, lots, leg.Position)
        ticker.take_position(timestamp, remarks, *legs)

    profit_target = None
    stop_loss = None
    portfolio = []
    TrackPTSL = StrategyPTSL(profit_target, stop_loss, *portfolio)

    timestamps = portfolio[0].timestamps
    current_position = None
    trade_start_date, trade_end_date, trade_count = None, None, 0
    for timestamp in timestamps:
        print()
        print(f"Timestamp: {timestamp}")
        #######################################################################################################################################################
        if TL.check_existing_position(portfolio[0]): #Custom logic for checking position in portfolio (|| of all tickers)
            
            print(f"Existing Position Check: TRUE | Dispersion position: {current_position}")
            
            # 3:20 (Market Close) SQUAREOFF
            if exit_signal(timestamp):
                print("************  MARKET CLOSE  *********** ")
                Update_Strategy_Tickers(timestamp, 'Remarks', False, True, False)
                SquareOff_Portfolio(timestamp, f'3:20/Market Close Square Off', True)
                trade_end_date = timestamp
                trade_count+=1
                zoom_dispersion_trade(trade_start_date, trade_end_date, f"Trade_{strategy_desc}.xlsx")
                visualise_dispersion_trade(trade_start_date, trade_end_date, f"Visualise_Trade_{strategy_desc}.html")
                continue

            # PROFIT TARGET AND STOP LOSS SQUAREOFF
            if TrackPTSL.is_valid(timestamp) and TrackPTSL.status(timestamp) != PTSL.Valid:
                print("************  TrackPTSL Trigger hit *********** ")
                print(f"{TrackPTSL.nature} Square off at NetPnl of {TrackPTSL.pnl_last_trade}")
                Update_Strategy_Ticker(timestamp, 'NoHedging', False, True, False)
                SquareOff_Portfolio(timestamp, f'{TrackPTSL.nature} Square Off', True)
                trade_end_date = timestamp
                trade_count+=1
                zoom_dispersion_trade(trade_start_date, trade_end_date, f"Trade_{strategy_desc}.xlsx")
                visualise_dispersion_trade(trade_start_date, trade_end_date, f"Visualise_Trade_{strategy_desc}.html")
                continue
                
            # IF NO NEED TO SQUAREOFF, HEDGE IF NEEDED
            Update_Strategy_Tickers(timestamp, 'Remarks for Delta Hedging (Syn/Fut)', True, True, True)
            print("========================================================================================================================================================================")
            continue
        #######################################################################################################################################################
        
        
        #######################################################################################################################################################
        if entry_signal(timestamp):
            print("************  ENTRY TIME REACHED  ************  ")
            Update_Strategy_Tickers(timestamp, "General Remarks", False, False, False)
            if TL.isTodayAnyExpiry(timestamp, portfolio):
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
                if lots_ticker == 0:
                    print("Cant take Dispersion Position, lots index for target notional vega can't be 0")
                    continue
                take_strategy_position(timestamp, f'Remarks on Taking Position', ticker, lots_ticker)
            
            TrackPTSL.fresh_trade(timestamp)
            current_position = ICView
            trade_start_date = timestamp
            print(f"{ICView} IC Trade executed")

            continue
        #######################################################################################################################################################

        Update_Strategy_Tickers(timestamp, "NoHedging", False, True, False)
        print("========================================================================================================================================================================")
        
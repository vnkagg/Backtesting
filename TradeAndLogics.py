import numpy as np
import pandas as pd
from enums import *

class Token:
    def __init__(self, ticker, opt_type, strike, fno = FNO.OPTIONS):
        self.ticker = ticker
        self.lot_size = self.ticker.lot_size
        self.opt_type = opt_type
        self.strike = strike
        self.fno = fno
        if self.fno == FNO.OPTIONS:
            self.data = ticker.get_opts(opt_type)[strike]
        else:
            self.data = ticker.df_futures[ticker.ohlc.name]
        self.lots = 0
        self.running_pnl = 0
        self.position = LongShort.Neutral
        self.stats = pd.DataFrame(0.0, index=ticker.df_futures.index,
                                  columns=['net_lots', 'position', 'running_pnl', 'net_delta', 'net_vega', 'underlying_iv'])
        self.stats['position'] = self.stats['position'].astype(object)
        self.called_for_timestamp = pd.DataFrame(False, index=ticker.df_futures.index, columns=['wasCalled'])
        self.slippage = 7.5
        if self.fno == FNO.OPTIONS:
            self.transaction_costs = 11.5
        elif self.fno == FNO.FUTURES:
            self.transaction_costs = 1.5


    def expenses(self, lots):
        return lots * (self.slippage + self.transaction_costs) * 0.01 * 0.01

    def add_position(self, timestamp, lots, position):
        lots = (self.lots * self.position.value + lots * position.value)
        self.lots = abs(lots)
        self.position = LongShort(np.sign(lots))
        transaction_value = lots * self.lot_size * self.data.loc[timestamp]
        self.running_pnl -= self.expenses(transaction_value)

    def update_df(self, time_ix, timestamp, logging=True):
        if pd.isna(self.data.loc[timestamp]):
            if self.fno == FNO.FUTURES:
                print(f"{self.fno.name} {self.ohlc.name} Price is Null")
            else:
                print(f"{self.opt_type.name} Option of Strike {self.strike} {self.ohlc.name} Price is Null")
            return
        if pd.isna(self.data.iloc[time_ix-1]):
            if self.fno == FNO.FUTURES:
                print(f"{self.fno.name} {self.ohlc.name} Price was Null at Previous Timestamp")
            else:
                print(f"Strike {self.strike} {self.opt_type.name} Option's {self.ohlc.name} Price was Null at Previous Timestamp")
            return
        # print(f"Not null at {timestamp}")
        pnl_this_moment = (self.data.loc[timestamp] - self.data.iloc[time_ix-1]) * (self.position.value) * self.lots * self.lot_size
        self.running_pnl += pnl_this_moment
        if self.fno == FNO.OPTIONS:
            self.stats.loc[timestamp, 'position'] = self.position
            self.stats.loc[timestamp, 'net_lots'] = self.lots
            self.stats.loc[timestamp, 'instrument'] = self.opt_type
            self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl
            greeks = self.ticker.greeks(timestamp, self.opt_type, self.strike, logging)
            if greeks is None:
                if logging:
                    print(f"{self.ticker.symbol}'s Token did not receive Greeks")
                    print(f"  Delta and Vega Information was not updated")
                    self.stats.loc[timestamp, 'net_delta'] = self.stats['net_delta'].iloc[time_ix-1]
                    self.stats.loc[timestamp, 'net_vega'] = self.stats['net_vega'].iloc[time_ix-1]
                    print(f"  Filled Current Timestamp with the previously latest available Greeks (t-1)")
                return
            self.stats.loc[timestamp, 'net_delta'] = greeks['delta'] * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'net_vega'] = greeks['vega'] * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'underlying_iv'] = greeks['implied_volatility']
        else:
            self.stats.loc[timestamp, 'position'] = self.position
            self.stats.loc[timestamp, 'net_lots'] = self.lots
            self.stats.loc[timestamp, 'instrument'] = self.fno
            self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl
            self.stats.loc[timestamp, 'net_delta'] = 1 * self.lots * self.position.value * self.lot_size
            self.stats.loc[timestamp, 'net_vega'] = 0
            self.stats.loc[timestamp, 'underlying_iv'] = 0
        # try:
        #     print("Printing running pnl from Token.update_df", self.stats.loc[timestamp, 'running_pnl'])
        # except Exception as e:
        #     print(e)

class PTSLHandling:
    def __init__(self, profit_target, stop_loss, *tickers):
        self.active_ptsl = False
        self.nature = "None"
        self.running_pnl_at_last_entry = 0
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.tickers = tickers
        self.triggered_at = None
        self.pnl_last_trade = None

    def trigger(self, timestamp):
        self.active_ptsl = True
        self.triggered_at = timestamp
    
    def fresh_trade(self, timestamp):
        self.active_ptsl = False
        self.running_pnl_at_last_entry = self.get_running_pnl(timestamp)
        self.nature = "None"
        self.pnl_last_trade = 0
    
    def get_running_pnl(self, timestamp):
        net_profit = 0
        for ticker in self.tickers:
            for token in ticker.tokens.values():
                net_profit += token.stats.loc[timestamp, 'running_pnl']
        return net_profit

    def status(self, timestamp):
        net_profit = self.get_running_pnl(timestamp) - self.running_pnl_at_last_entry
        if net_profit >= self.profit_target:
            self.trigger(timestamp)
            self.nature = "Profit Target"
            self.pnl_last_trade = net_profit
            return self.nature
        if net_profit <= -self.stop_loss:
            self.trigger(timestamp)
            self.nature = "Stop Loss"
            self.pnl_last_trade = net_profit
            return self.nature
        return "good_to_go"
    
    def handle_square_off(self, timestamp):
        self.active_ptsl = False
    
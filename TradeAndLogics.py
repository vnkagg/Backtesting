import numpy as np
import pandas as pd
from enums import *

class Token:
    def __init__(self, ticker, opt_type, strike, fno = FNO.OPTIONS):
        self.ticker = ticker
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



    def add_position(self, lots, position):
        lots = (self.lots * self.position.value + lots * position.value)
        self.lots = abs(lots)
        self.position = LongShort(np.sign(lots))

    def update_df(self, time_ix, timestamp):
        if pd.isna(self.data.iloc[time_ix-1]) or pd.isna(self.data.loc[timestamp]):
            return
        # print(f"Not null at {timestamp}")
        pnl_this_moment = (self.data.loc[timestamp] - self.data.iloc[time_ix-1]) * (self.position.value) * self.lots
        self.running_pnl += pnl_this_moment
        if self.fno == FNO.OPTIONS:
            self.stats.loc[timestamp, 'position'] = self.position
            self.stats.loc[timestamp, 'net_lots'] = self.lots
            self.stats.loc[timestamp, 'instrument'] = self.opt_type
            self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl
            greeks = self.ticker.greeks(timestamp, self.opt_type, self.strike)
            if greeks is None:
                print(f"Could not update {self.ticker.symbol} information for {timestamp} due to unavailability of greeks. TL.Token.update_df() -> self.ticker.greeks")
                return
            self.stats.loc[timestamp, 'net_delta'] = greeks['delta'] * self.lots * self.position.value
            self.stats.loc[timestamp, 'net_vega'] = greeks['vega'] * self.lots * self.position.value
            self.stats.loc[timestamp, 'underlying_iv'] = greeks['implied_volatility']
        else:
            self.stats.loc[timestamp, 'position'] = self.position
            self.stats.loc[timestamp, 'net_lots'] = self.lots
            self.stats.loc[timestamp, 'instrument'] = self.fno
            self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl
            self.stats.loc[timestamp, 'net_delta'] = 1 * self.lots * self.position.value
            self.stats.loc[timestamp, 'net_vega'] = 0
            self.stats.loc[timestamp, 'underlying_iv'] = 0
        # try:
        #     print("Printing running pnl from Token.update_df", self.stats.loc[timestamp, 'running_pnl'])
        # except Exception as e:
        #     print(e)


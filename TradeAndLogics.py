import numpy as np
import pandas as pd
from enums import *

class Token:
    def __init__(self, ticker, opt_type, strike):
        self.ticker = ticker
        self.opt_type = opt_type
        self.strike = strike
        self.opts = ticker.get_opts(opt_type)[strike]
        self.lots = 0
        self.running_pnl = 0
        self.position = LongShort.Neutral
        self.stats = pd.DataFrame(0.0, index=ticker.df_futures.index,
                                  columns=['net_lots', 'running_pnl', 'net_delta', 'net_vega', 'underlying_iv'])


    def add_position(self, lots, position):
        self.lots += lots * position.value
        self.position = LongShort(np.sign(self.lots))

    def update_df(self, time_ix, timestamp):
        if pd.isna(self.opts.iloc[time_ix-1]) or pd.isna(self.opts.loc[timestamp]):
            return
        # print(f"Not null at {timestamp}")
        pnl_this_moment = (self.opts.loc[timestamp] - self.opts.iloc[time_ix-1]) * (self.position.value) * self.lots
        self.running_pnl += pnl_this_moment
        greeks = self.ticker.greeks(timestamp, self.opt_type, self.strike)
        self.stats.loc[timestamp, 'net_lots'] = self.lots
        self.stats.loc[timestamp, 'running_pnl'] = self.running_pnl
        self.stats.loc[timestamp, 'net_delta'] = greeks['delta'] * self.lots
        self.stats.loc[timestamp, 'net_vega'] = greeks['vega'] * self.lots
        self.stats.loc[timestamp, 'underlying_iv'] = greeks['implied_volatility']
        # try:
        #     print("Printing running pnl from Token.update_df", self.stats.loc[timestamp, 'running_pnl'])
        # except Exception as e:
        #     print(e)


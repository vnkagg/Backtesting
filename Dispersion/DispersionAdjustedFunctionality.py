from Modules import Data_Processing as dp
from Modules import Data as data
from Modules.enums import LongShort
import pandas as pd
import numpy as np
from Modules.Utility import is_invalid_value

class ticker(dp.ticker):
    def __init__(self, symbol, lot_size, FetchData, start_date, end_date, expiry_type, toFill = True, t = 1, is_index = False, risk_free=0.1):
        super().__init__(symbol, lot_size,  FetchData, start_date, end_date, expiry_type, toFill, t, is_index, risk_free)

    def initializeDispersion(self, components, is_child, weight):
        # if self.is_index == is_component:
        #     if self.is_index:
        #         raise ValueError(f"CONFLICT: The ticker was initialized to be an Index. An index can't be a component")
        #     else:
        #         raise ValueError(f"CONFLICT: The ticker was initialized to be a Stock. A stock can't be an Index")
        self.components = components
        self.is_child = is_child
        self.is_parent = not is_child
        if self.is_parent:
            for component in self.components.values():
                component.index = self
        self.weight = weight

    def set_intention(self, intention):
        self.intention = intention

    def get_delta_threshold_per_lot(self):
        if self.intention == LongShort.Long:
            return 5
        else:
            return 1
        #  ALT LOGIC
        # if(self.is_index):
        #     return 5
        # return 1

    def get_bhav_data_futures(self, dates_to_fetch):
        fetched_df = data.get_bhav_data_futures_for_dates(self.symbol, self.expiry_type, dates_to_fetch)
        fetched_df = dp.drop_duplicates_from_df(fetched_df)
        fetched_df.set_index('date_timestamp', inplace=True)
        if hasattr(self, 'df_bhav'):
            self.df_bhav = pd.concat([self.df_bhav, fetched_df])
            self.df_bhav.sort_index(inplace=True)
        else:
            self.df_bhav = fetched_df
        return 

    # EOD data to be used for 50 days
    def get_bhav_data_in_window(self, timestamp, window=21):
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
        end_date = timestamp.date()
        start_date = dp.get_date_minus_n_days(end_date, window)
        expected_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        expected_date_range = dp.get_market_valid_days(expected_date_range)
        timestamps_demanded = timestamps_to_fetch = expected_date_range
        if hasattr(self, 'df_bhav'):
            data_present = timestamps_demanded.isin(self.df_bhav.index)
            timestamps_to_fetch = timestamps_demanded[~data_present]
        dates_to_fetch = pd.to_datetime(timestamps_to_fetch)
        if len(dates_to_fetch):
            self.get_bhav_data_futures(dates_to_fetch)
        return self.df_bhav.loc[timestamps_demanded, self.ohlc.name]

    def get_correlations_in_look_back_period(self, timestamp, window=21):
        bhav_data_look_back_period = []
        for stock, component in self.components.items():
            component_bhav_data_look_back_period = component.get_bhav_data_in_window(timestamp, window).rename(stock)
            if component_bhav_data_look_back_period is None:
                print(f"Can't get correlations in look back period without fetching bhav data for {stock}")
                return None
            if len(component_bhav_data_look_back_period) < window:
                return None
            bhav_data_look_back_period.append(component_bhav_data_look_back_period)
        df_merged = pd.concat(bhav_data_look_back_period, axis=1)
        # print(f"Correlations Calculation between {self.symbol} components completed")
        correlation_matrix = df_merged.corr()
        return correlation_matrix

    def get_ic_at(self, timestamp, log_sanity=False):
        if 'ic' not in self.df_futures.columns:
            self.generate_ic(timestamp, log_sanity)
        if is_invalid_value(self.df_futures.loc[timestamp, 'ic']):
            self.generate_ic(timestamp, log_sanity)
        return self.df_futures.loc[timestamp, 'ic']

    def generate_ic(self, timestamp, log_sanity=False, log_error=True):
        print(f"Timestamp: {timestamp} | Implied Correlation")
        self.df_futures.loc[timestamp, 'ic'] = pd.NA
        
        flag = False
        if pd.isna(self.get_iv_at(timestamp, log_sanity)):
            if log_error:
                print(f"IV is Nan for {self.symbol}")
            flag = True
        for s, c in self.components.items():
            if pd.isna(c.get_iv_at(timestamp, log_sanity)):
                if log_error:
                    print(f"IV is nan for {s}")
                flag = True
        if flag:
            if log_error:
                print(f"Skipping Timestamp due to NaN IV's")
                print('-------------------------------------------')
            return
        if log_sanity:
            print(f"IC sanity Parameters")
            print(f"  IV({self.symbol}): {self.get_iv_at(timestamp)}")
            for symbol, component in self.components.items():
                print(f"  IV({symbol}): {component.get_iv_at(timestamp)} | weight: {component.weight}")

        numerator, denominator = (self.get_iv_at(timestamp)*self.weight)**2, 0
        for s1, component1 in self.components.items():
            for s2, component2 in self.components.items():
                if s1 == s2:
                    numerator -= (component1.weight * component1.get_iv_at(timestamp))**2
                else:
                    denominator += component1.weight * component2.weight * component1.get_iv_at(timestamp) * component2.get_iv_at(timestamp)

        if denominator == 0:
            if log_sanity or log_error:
                print(f"IC Denominator was 0. Invalidating this timestamp")
        else:
            self.df_futures.loc[timestamp, 'ic'] = numerator / denominator
            
        print(f"Dirty Correlation(IC): {self.df_futures.loc[timestamp, 'ic']}")
        print('-------------------------------------------')
        return
    
    def get_historical_volatility_at(self, timestamp, window=21):
        futures_prices = self.get_bhav_data_in_window(timestamp, window)
        log_returns = np.log(futures_prices / futures_prices.shift(1))
        historical_volatility = log_returns.std() * np.sqrt(252)
        return float(np.round(historical_volatility, 2))

    def generate_hc(self, timestamp, window=21, log_sanity=False, log_error=True):
        print(timestamp)
        if log_sanity:
            print(f"HC sanity Parameters")
        hv = {}
        hv[self.symbol] = self.get_historical_volatility_at(timestamp)
        if log_sanity:
            print(f"  HV({self.symbol}): {hv[self.symbol]}")
        for symbol, component in self.components.items():
            hv[symbol] = component.get_historical_volatility_at(timestamp)
            if log_sanity:
                print(f"  HV({symbol}): {hv[symbol]} | weight: {component.weight}")
        numerator, denominator = (hv[self.symbol] *self.weight)**2, 0
        for s1, component1 in self.components.items():
            for s2, component2 in self.components.items():
                if s1 == s2:
                    numerator -= (component1.weight * hv[s1])**2
                else:
                    denominator += component1.weight * component2.weight * hv[s1] * hv[s2]

        if denominator == 0:
            if log_sanity or log_error:
                print(f"HC Denominator was 0. Invalidating this timestamp")
            return
        else:
            self.df_futures.loc[timestamp, 'hc'] = numerator / denominator
            
        print(f"Dirty Correlation(HC): {self.df_futures.loc[timestamp, 'hc']}")
        print('-------------------------------------------')
        return
        

    def get_hc_at(self, timestamp, window=21):

        if 'hc' not in self.df_futures.columns:
            self.generate_hc(timestamp, window)
        if is_invalid_value(self.df_futures.loc[timestamp, 'hc']):
            self.generate_hc(timestamp, window)
        return self.df_futures.loc[timestamp, 'hc']

class RawWeightedPortfolio:
    def __init__(self):
        self.constituents = {}
        return
    
    def insert(self, symbol, lot_size, weight):
        data = {
            'Weight': weight,
            'LotSize': lot_size
        }
        self.constituents[symbol] = data
    
    def normalize_constituents_weights(self):
        import copy
        total_weight = 0
        constituents = copy.deepcopy(self.constituents)
        for values in constituents.values():
            total_weight += values['Weight']
        if total_weight != 100:
            for values in constituents.values():
                values['Weight'] = (values['Weight'] / total_weight) * 100
        return constituents

    def LotSize(self, symbol):
        constituents = self.normalize_constituents_weights()
        return constituents[symbol]['LotSize']
    
    def Weight(self, symbol):
        constituents = self.normalize_constituents_weights()
        return constituents[symbol]['Weight']/100

    def Symbols(self):
        return self.constituents.keys()
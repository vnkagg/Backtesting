import numpy as np
import pandas as pd
from Modules.enums import FNO, Option

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
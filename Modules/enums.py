from enum import Enum
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np

class LongShort(Enum):
    Long = 1
    Neutral = 0
    Short = -1

    def opposite(self):
        if self == LongShort.Long:
            return LongShort.Short
        elif self == LongShort.Short:
            return LongShort.Long
        return LongShort.Neutral

class Option(Enum):
    Put = -1
    Call = 1

class Spread(Enum):
    Straddle = 2
    Strangle = 3
    Butterfly = 4
    IronFly = 5
    Condor = 6
    IronCondor = 7
    BullCallSpread = 8
    BearCallSpread = 9
    BullPutSpread = 10
    BearPutSpread = 11

class OHLC(Enum):
    open = 0
    high = 1
    low = 2
    close = 3

class PTSL(Enum):
    StopLoss = -1
    Valid = 0
    ProfitTarget = 1

class FNO(Enum):
    OPTIONS = 0
    FUTURES = 1

class DB(Enum):
    QDAP = 0
    LocalDB = 1
    GeneralOrQuantiPhi = 2

class Phase(Enum):
    Past = -1
    Present = 0
    Future = 1

class Period(Enum):
    Daily = "D"
    Monthly = "ME"
    Annual = "YE"
    Complete = ""

class Greeks(Enum):
    Delta = 0
    Gamma = 1
    Theta = 2
    Vega = 3
    Rho = 4
    Implied_Volatility = 5

@dataclass
class GreeksParameters:
    symbol: str
    timestamp: str
    expiry_date: str
    option_type: Option
    option_strike: float
    option_price: float
    underlying_price: float
    risk_free_rate: float
    dividend_yield: float

@dataclass
class Leg:
    Position: LongShort
    Lots: float
    Instrument: Union[Option, FNO]
    Strike: float
    Price: Optional[float] = None
    LegName: Optional[str] = ""
    LotSize: Optional[float] = None

    def payoff(self, x, lotsize_include=False):
        if self.Instrument == FNO.FUTURES:
            return (x - self.Price) * self.Lots * self.Position.value * (1*(1-int(lotsize_include)) + int(lotsize_include)*self.LotSize)
        else:
            intrinsic_value = np.maximum((x - self.Strike) * self.Instrument.value, 0)
            return (intrinsic_value - self.Price) * self.Lots * self.Position.value * (1*(1-int(lotsize_include)) + int(lotsize_include)*self.LotSize)

    def id(self):
        if self.Instrument == FNO.FUTURES:
            return 'Futures'
        else:
            return f'{self.Instrument.name}_{self.Strike}'

    def __str__(self):
        return (
            f"Leg(\n"
            f"  Position={self.Position.name},\n"
            f"  Lots={self.Lots},\n"
            f"  Instrument={self.Instrument},\n"
            f"  Strike={self.Strike},\n"
            f"  Price={self.Price},\n"
            f"  LegName='{self.LegName}'\n"
            f")"
        )
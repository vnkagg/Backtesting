from enum import Enum

class LongShort(Enum):
    Long = 1
    Neutral = 0
    Short = -1

class Option(Enum):
    Put = 0
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

class FNO(Enum):
    OPTIONS = 0
    FUTURES = 1


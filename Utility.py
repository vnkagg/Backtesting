def reload_modules():
    import importlib
    import TradeAndLogics as TL
    importlib.reload(TL)
    import Plot
    importlib.reload(Plot)
    import Data as data
    importlib.reload(data)
    import Data_Processing as dp
    importlib.reload(dp)

def fix_output(original_stdout, *file):
    import sys
    for f in file:
        f.close()
    sys.stdout = original_stdout.value

class const:
    def __init__(self, value):
        self._value = value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        return
    
def pwd():
    import os
    return os.getcwd()

def round(x, dig=2):
    import numpy as np
    return np.round(x, dig)
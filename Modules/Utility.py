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




def fix_output(original_stdout):
    import gc
    import io
    import sys
    sys.stdout = original_stdout
    for obj in gc.get_objects():
        try:
            # Check if it's a file-like object and not closed
            if isinstance(obj, io.TextIOWrapper) and not obj.closed:
                if hasattr(obj, 'name') and obj.name and not obj.name.startswith("<"):
                    print(f"Open file detected: {obj.name}")
                obj.close()
        except Exception as e:
            print(f"Error while processing object: {e}")




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

def ceil_of_division(a, b):
    return (a + b - 1)//b

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def is_invalid_value(value):
    # Check for NaN, None, or pd.NA using Pandas' robust `isna` method
    if pd.isna(value):
        return True
    if np.isnan(value):
        return True
    if isinstance(value, pd.DataFrame) and value.empty:
        return True
    # Check for empty strings or empty objects
    if isinstance(value, str) and value.strip() == "":
        return True
    # Check for zero if the value is numeric
    if isinstance(value, (int, float, np.number)) and value == 0:
        return True
    # Check for empty containers (like lists, dictionaries, etc.)
    if isinstance(value, (list, dict, set, tuple)) and len(value) == 0:
        return True
    # If none of the above conditions are met, the value is valid
    return False

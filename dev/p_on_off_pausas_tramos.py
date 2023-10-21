#%%
#from itertools import chain, combinations
#import math
#from src.simulador_v02 import *  
#from scipy.stats import gmean
from src.datos_utils import *
#import optuna
#import itertools
import pandas as pd
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, time
#import math
#import random 
#from src.forecast_utils import *

##----------------------Datos históricos de un día----------------
import pandas as pd
import numpy as np
from datetime import timedelta

def get_time_intervals(df, n, percentage:float=100):
    # Step 1: Find the minimum and maximum times from the FH_Emi column
    min_time = df['FH_Emi'].min()
    max_time = df['FH_Emi'].max()    
    # Step 2: Calculate the total time span
    total_span = max_time - min_time    
    # Step 3: Divide this span by n to get the length of each interval
    interval_length = total_span / n    
    # Step 4: Create the intervals
    intervals = [(min_time + i*interval_length, min_time + (i+1)*interval_length) for i in range(n)]    
    # New Step: Adjust the start time of each interval based on the percentage input
    adjusted_intervals = [(start_time + 0.01 * (100 - percentage) * (end_time - start_time), end_time) for start_time, end_time in intervals]
    # Step 5: Format the intervals as requested
    formatted_intervals = [(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')) for start_time, end_time in adjusted_intervals]
    
    return formatted_intervals

# Usage:
# intervals = get_time_intervals(df, 4, 80)

dataset     = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
get_time_intervals(el_dia_real, 4, 80)
#%%
""" 
From that function I am getting outputs like this:

[('08:40:11', '10:07:44'),
 ('10:07:44', '11:35:17'),
 ('11:35:17', '13:02:50'),
 ('13:02:50', '14:30:23')]
 
Now add a new argument which has to be a percentage (like 80). If the input is 80, it means that you
have to report only the  last 80% of each interval 
"""
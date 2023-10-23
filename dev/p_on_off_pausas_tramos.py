#%%
#from itertools import chain, combinations
#import math
#from src.simulador_v02 import *  
#from scipy.stats import gmean
import os
os.chdir('/DeepenData/Repos/Flux_v0')
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
get_time_intervals(el_dia_real, 4, 100)
#%%
""" 
I have this list of tuples with time intervals within; (start, end):

[('08:40:11', '10:07:44'),
 ('10:07:44', '11:35:17'),
 ('11:35:17', '13:02:50'),
 ('13:02:50', '14:30:23')]

Also, I have a dataframe with the column "FH_Emi" having dtype as satetime64[s], 
an entry in "FH_Emi" may look like 2023-05-15 08:42:12.

I need a function to partition my dataframe in to the number of tuples (or time intervals) in the list above,
So the number of output partitions (each one has to be dataframes) is equal to the number of
time intervals. Each partition must start and end  according the time interval and the values in FH_Emi. 


"""
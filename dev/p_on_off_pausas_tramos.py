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

def get_time_intervals(df, n):
    # Step 1: Find the minimum and maximum times from the FH_Emi column
    min_time = df['FH_Emi'].min()
    max_time = df['FH_Emi'].max()    
    # Step 2: Calculate the total time span
    total_span = max_time - min_time    
    # Step 3: Divide this span by n to get the length of each interval
    interval_length = total_span / n    
    # Step 4: Create the intervals
    intervals = [(min_time + i*interval_length, min_time + (i+1)*interval_length) for i in range(n)]    
    # Step 5: Format the intervals as requested
    formatted_intervals = [(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')) for start_time, end_time in intervals]
    
    return formatted_intervals

# Assume df is your DataFrame
# n is the number of intervals you want, e.g., 4
# intervals = get_time_intervals(df, 4)


dataset     = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
get_time_intervals(el_dia_real, 4)
#%%
""" 
I have this dataframe, this column is index (ignore it):

IdOficina	IdSerie	IdEsc	FH_Emi	FH_Llama	FH_AteIni	FH_AteFin	T_Esp	T_Ate
269906	2	12	7	2023-05-15 08:40:11	2023-05-15 08:40:36	2023-05-15 08:40:36	2023-05-15 08:42:12	25	96
269907	2	14	10	2023-05-15 08:40:54	2023-05-15 08:42:53	2023-05-15 08:42:55	2023-05-15 08:45:32	121	157
269908	2	14	12	2023-05-15 08:41:07	2023-05-15 08:43:14	2023-05-15 08:43:32	2023-05-15 08:51:10	145	458
270001	2	14	11	2023-05-15 08:41:18	2023-05-15 08:45:05	2023-05-15 08:46:13	2023-05-15 09:00:34	295	861
270002	2	14	10	2023-05-15 08:41:27	2023-05-15 08:45:33	2023-05-15 08:45:36	2023-05-15 08:51:57	249	381
...	...	...	...	...	...	...	...	...	...
271297	2	10	7	2023-05-15 14:12:59	2023-05-15 14:28:20	2023-05-15 14:28:20	2023-05-15 14:29:10	921	50
271298	2	10	9	2023-05-15 14:13:23	2023-05-15 14:37:58	2023-05-15 14:38:13	2023-05-15 16:00:46	1490	4953
271299	2	12	8	2023-05-15 14:14:26	2023-05-15 14:16:33	2023-05-15 14:16:52	2023-05-15 14:22:09	146	317
271300	2	10	6	2023-05-15 14:18:32	2023-05-15 14:35:41	2023-05-15 14:35:41	2023-05-15 14:36:18	1029	37
271301	2	17	13	2023-05-15 14:30:23	2023-05-15 14:30:40	2023-05-15 14:31:23	2023-05-15 16:07:00	60	5737

the dtypes are:
IdOficina            Int32
IdSerie               Int8
IdEsc                 Int8
FH_Emi       datetime64[s]
FH_Llama     datetime64[s]
FH_AteIni    datetime64[s]
FH_AteFin    datetime64[s]
T_Esp                int32
T_Ate                int32
dtype: object

Make a function which based on the column FH_Emi, extract n (user defined)
sequentialy equaly time spaced intervals of time considering the min and max values from FH_Emi,

and return the intervals in hour:min:sec like (("8:00","9:00"),("9:00","10:00"),("10:00","11:00"),("11:00","12:00"))
"""
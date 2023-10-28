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
#Implementacion
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import random


dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
modos    = ['Rebalse','Alternancia', 'Rebalse']
atributos_series = atributos_x_serie(ids_series=series, 
                                    sla_porcen_user=None, 
                                    sla_corte_user=None, 
                                    pasos_user=None, 
                                    prioridades_user=None)

niveles_servicio_x_serie = {atr_dict['serie']:
                           (atr_dict['sla_porcen']/100, atr_dict['sla_corte']/60) 
                           for atr_dict in atributos_series}

prioridades =       {atr_dict['serie']:
                    atr_dict['prioridad']
                    for atr_dict in atributos_series}

planificacion = {'0': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '1': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '12': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '33': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '34': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '35': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '49': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series), 
    'configuracion_atencion':random.sample(modos, 1)[0]}}],
 '50': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '51': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills':get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}]}
intervals  = get_time_intervals(un_dia, 4, 100) # Una funcion que recibe un dia, un intervalo, y un porcentaje de actividad para todos los intervalos
partitions = partition_dataframe_by_time_intervals(un_dia, intervals) # TODO: implementar como un static del simulador? 
#%%
registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, partitions[0], prioridades) # 

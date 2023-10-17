#%%

#%%
import warnings
warnings.filterwarnings("ignore")

from datetime import date, time
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datetime import date, time
import math
import random 
from src.forecast_utils import *
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
SERIES = [5, 10, 11, 12, 14, 17]#[10, 14, 12, 17, 5, 11]

forecaster_ttp = ForecasterTTP(dataset=dataset, series=SERIES, atenciones_diarias_minimas=1)

#%%
# Genera una proyeccion de demanda, 
demandas_proyectadas_q = forecaster_ttp.proyectar_demanda_SARIMAX('2023-05-15').demandas_proyectada_cuantizadas()
demandas_proyectadas_q = ForecasterTTP.un_dia('2023-05-15', demandas_proyectadas_q) # Retorna solo un dia 2023-05-15
#%%
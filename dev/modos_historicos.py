#%%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import random
import optuna
from src.datos_utils import DatasetTTP, obtener_skills
from src.optuna_utils import non_empty_subsets
import itertools
import numpy as np
from src.optuna_utils import plan_unico
from dev.atributos_de_series import atributos_x_serie
import optuna
import numpy as np
import time
from src.optuna_utils import (
    sla_x_serie, 
    extract_skills_length, 
    non_empty_subsets
    )
from   src.utils_Escritoriosv05_Simv05 import (
                                            get_permutations_array,
                                           plan_unico,
                                           generar_planificacion,
                                           extract_min_value_keys,
                                            DatasetTTP,
                                            get_time_intervals,
                                            partition_dataframe_by_time_intervals)
from dev.Escritoriosv05_Simv05 import simv05
from dev.atributos_de_series import atributos_x_serie
import math

dataset                                 = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz") # IdOficina=2)
un_dia                                  = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills       = obtener_skills(un_dia)

#%%
######################################
#-----Modo parámetros históricos------
#######################################

def plan_desde_skills(skills, porcentaje_actividad, inicio):    
    return  {id: [
                    {'inicio':inicio,
                    'termino':None,
                    'prioridades': {
                        'skills': sks,
                        'configuracion_atencion': None,
                        'porcentaje_actividad'  :  porcentaje_actividad,
                        'atributos_series':atributos_x_serie(
                            ids_series=sorted(list({val for sublist in skills.values() for val in sublist})), 
                            sla_porcen_user=None, 
                            sla_corte_user=None, 
                            pasos_user=None, 
                            prioridades_user=None),
                    }}
                        ] for id, sks in skills.items()}


plan_desde_skills(skills, porcentaje_actividad = 80, inicio = '08:00:00')



#%%
from typing import Dict, List, Tuple
import random
from copy import deepcopy
from itertools import count, islice
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import pandas as pd

def actualizar_conexiones(original_dict, update_dict):  
    for key, value in update_dict.items():
        if key in original_dict:
            original_dict[key]['conexion'] = value            
    return deepcopy(original_dict)
  
class MisEscritorios:
    
    def __init__(self,
                 inicio_tramo:  pd.Timestamp, 
                 fin_tramo:     pd.Timestamp,
                 planificacion: dict, 
                 conexiones:    dict = None):
      
        self.planificacion = planificacion
        self.escritorios   = {k:{
                                "skills": v[0]['propiedades']['skills'],
                                'modo_atencion' : v[0]['propiedades']['modo_atencion'],
                                'contador_tiempo_disponible': iter(count(start=0, step=1)),
                                'numero_de_atenciones':0,
                                'porcentaje_actividad': v[0]['propiedades']['porcentaje_actividad'],
                                'duracion_inactividad':int(
                                (1- v[0]['propiedades']['porcentaje_actividad'])*(fin_tramo - inicio_tramo).total_seconds()/60)
                                } 
                              for k,v in self.planificacion.items()}
        if not conexiones:
        #     #si no se provee el estado de los conexiones se asumen todas como True (todos conectados):
             conexiones                         = {f"{key}": random.choices([True, False], [1, 0])[0] for key in self.escritorios}
        self.escritorios                        = actualizar_conexiones(self.escritorios, conexiones)       
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.nuevos_escritorios_programados     = []
        self.registros_escritorios              = []
        
        #assert 0 < porcentaje_actividad <= 1
        #tiempo_total         = (fin_tramo - inicio_tramo).total_seconds()/60
        #self.duracion_inactividad = (1-porcentaje_actividad)*(fin_tramo - inicio_tramo).total_seconds()/60
        
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



""" 
Modificar clase `MisEscritorios` para que se instancie con `planificacion`, `niveles_servicio_x_serie` y `prioridades`
"""

planificacion = {'0': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills' : get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,          
    }}],
 '1': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '12': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '33': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '34': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '35': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '49': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series), 
    'modo_atencion':random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '50': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}],
 '51': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills':get_random_non_empty_subset(series),
    'modo_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(50, 90)/100,
    }}]}

supervisor = MisEscritorios(inicio_tramo  = un_dia['FH_Emi'].min(),
                                     fin_tramo     = un_dia['FH_Emi'].max(),
                                     planificacion = planificacion)

supervisor.escritorios
#%%

# %%

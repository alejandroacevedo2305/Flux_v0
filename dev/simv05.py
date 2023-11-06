#%%
import numpy as np
from datetime import datetime
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import DatasetTTP, obtener_skills
import random

from src.optuna_utils import (
    sla_x_serie, 
    calculate_geometric_mean, 
    extract_skills_length, 
    extract_min_value_keys, 
    extract_max_value_keys, 
    non_empty_subsets, 
    get_random_non_empty_subset, 
    get_time_intervals,
    partition_dataframe_by_time_intervals,  
    plan_unico
    )

from src.simulador_v02 import (
    reset_escritorios_OFF,
    one_cycle_iterator,
    create_multiindex_df,
    generate_integer, 
    actualizar_conexiones,
    generador_emisiones,
    timestamp_iterator,
    terminar_un_tramo,
    iniciar_un_tramo,
    update_escritorio,
    separar_por_conexion,
    poner_pasos_alternancia,
    pasos_alternancia,
    mismo_minuto,
    balancear_carga_escritorios,
    extract_highest_priority_and_earliest_time_row,
    remove_selected_row,
    FIFO
    ) 
import pandas as pd

from datetime import timedelta
from dev.atributos_de_series import atributos_x_serie
from dev.pasos_alternancia_y_prioridades_x_escri import (
    generar_pasos_para_alternancia_v02, 
    pasos_alternancia_v02,
    poner_pasos_alternancia_v02,
    MisEscritorios_v03,        
)

from dev.simv04 import (
    reloj_rango_horario, 
    match_emisiones_reloj)
def update_conexion_key(original_dict, updates):    
    for key, value in updates.items():  # Loop through the keys and values in the updates dictionary.
        if key in original_dict:  # Check if the key from updates exists in the original dictionary.
            original_dict[key]['conexion'] = value['conexion']
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
modos    = ['FIFO']#['Rebalse','Alternancia', 'Rebalse']
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
planificacion = {
        '0': [{'inicio': '08:00:11',
        'termino': "10:30:00",
        'propiedades': {'skills' : get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,
                
            }},
              {'inicio': '11:33:00',
        'termino': "12:40:00",
        'propiedades': {'skills' : get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,
                
            }}
              ],
        
        '1': [{'inicio': '09:00:11',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '2': [{'inicio': '10:00:11',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '3': [{'inicio': '12:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '4': [{'inicio': '08:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '5': [{'inicio': '08:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '6': [{'inicio': '08:00:56',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series), 
            'configuracion_atencion':random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '7': [{'inicio': '08:00:56',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '8': [{'inicio': '10:00:56',
        'termino': '11:00:00',
        'propiedades': {'skills':get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
            'atributos_series':atributos_series,
            }},
               {'inicio': '12:00:00',
        'termino': '16:00:00',
        'propiedades': {'skills':get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
            'atributos_series':atributos_series,
            }}]
        }

hora_cierre               = '20:00:00'
        

reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
registros_atenciones  = pd.DataFrame()
matcher_emision_reloj = match_emisiones_reloj(un_dia)

supervisor            = MisEscritorios_v03(inicio_tramo            = un_dia['FH_Emi'].min(),
                                    fin_tramo                = un_dia['FH_Emi'].max(),
                                    planificacion            = planificacion,
                                    niveles_servicio_x_serie = niveles_servicio_x_serie)
fecha                = un_dia.FH_Emi.iloc[0].date()
registros_atenciones = pd.DataFrame()
fila                 = pd.DataFrame()
#for hora_actual in reloj:     
hora_actual = next(reloj)
#supervisor.aplicar_agenda(hora_actual=  pd.Timestamp(f"{fecha} {hora_actual}"), agenda = planificacion)

[d['conexion'] for _, d in supervisor.escritorios.items()]

#%%

hora_actual = "11:30:00"
conexiones = dict()
for idEsc, un_escritorio in planificacion.items():
    for un_tramo in un_escritorio:
        on = hora_actual >= un_tramo['inicio'] and (lambda: 
            hora_actual <= un_tramo['termino'] if un_tramo['termino'] is not None else True)()
        #print(f"{idEsc}: {on} {hora_actual} >= {un_tramo['inicio']} and {hora_actual} <= {un_tramo['termino']}")
        #if on:            
        conexiones = conexiones | {idEsc: {'conexion': on}}
update_conexion_key(supervisor.escritorios, conexiones)  
 
escritorios_ON, escritorios_OFF = separar_por_conexion(supervisor.escritorios)
escritorios_OFF                 = reset_escritorios_OFF(escritorios_OFF)



[d['conexion'] for _, d in escritorios_ON.items()], [d['conexion'] for _, d in escritorios_OFF.items()]


#%%
def iniciar_un_tramo(hora_actual, tramos_un_escritorio):
    for tramo in tramos_un_escritorio:
        inicio  = pd.Timestamp(f"{hora_actual.date()} {tramo['inicio']}")
        #termino = pd.Timestamp(f"{hora_actual.date()} {tramo['termino']}")
        if (hora_actual >= inicio):
            return tramo
    return False

def terminar_un_tramo(hora_actual, tramos_un_escritorio):

    for idx_tramo, tramo in enumerate(tramos_un_escritorio):
        if tramo['termino'] is None:
            return False
        ##inicio  = pd.Timestamp(f"{hora_actual.date()} {tramo['inicio']}")
        termino = pd.Timestamp(f"{hora_actual.date()} {tramo['termino']}")
        if hora_actual > termino:
            return [tramo, idx_tramo]
    return False 

def aplicar_agenda(self, hora_actual, agenda):
    
    for idEsc, tramos_un_escritorio in agenda.items():
        
        if tramo_idx_tramo := terminar_un_tramo(hora_actual, tramos_un_escritorio):
            tramo     = tramo_idx_tramo[0]
            idx_tramo = tramo_idx_tramo[1]
            #print(f"{idEsc} termina tramo (eliminado de agenda): {tramo}")
            self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'terminar')
            del agenda[idEsc][idx_tramo]   
        
        if tramo:=  iniciar_un_tramo(hora_actual, tramos_un_escritorio):
            #se va seguir ejecutando mientras el tramo sea válido
            #poner alguna flag para q no se vuelva a ejecutar
            #print(f"{idEsc} inicia tramo: {tramo}")
            self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'iniciar')
  
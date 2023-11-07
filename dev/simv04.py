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

def reloj_rango_horario(start: str, end: str):
    start_time = datetime.strptime(start, '%H:%M:%S').time()
    end_time = datetime.strptime(end, '%H:%M:%S').time()
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    for minute in range(start_minutes, end_minutes + 1):
        hours, remainder = divmod(minute, 60)
        yield "{:02d}:{:02d}:{:02d}".format(hours, remainder, start_time.second)

class match_emisiones_reloj():
    def __init__(self, bloque_atenciones) -> bool:        
        self.bloque_atenciones = bloque_atenciones[['FH_Emi', 'IdSerie', 'T_Ate']]       
    def match(self, tiempo_actual):        
        # Convert the given time string to a timedelta object
        h, m, s = map(int, tiempo_actual.split(':'))
        given_time = timedelta(hours=h, minutes=m, seconds=s)
        # Filter rows based on the given condition
        try:
            mask = self.bloque_atenciones['FH_Emi'].apply(
                    lambda x: abs(timedelta(
                    hours=x.hour, minutes=x.minute, seconds=x.second) - given_time) <= timedelta(seconds=60)) 
                
            # Rows that satisfy the condition
            self.match_emisiones   = self.bloque_atenciones[mask].copy()
            self.match_emisiones['espera'] = 0        
            self.bloque_atenciones = self.bloque_atenciones[~mask]#.copy()            
            return self
        except KeyError:
            #self.bloque_atenciones 
            pass

def simulador_v04(un_dia, planificacion, niveles_servicio_x_serie, hora_cierre):
    
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
    for hora_actual in reloj:
        supervisor.aplicar_agenda(hora_actual=  pd.Timestamp(f"{fecha} {hora_actual}"), agenda = planificacion)

        if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
            en_atencion            = supervisor.filtrar_x_estado('atención') or []
            en_pausa               = supervisor.filtrar_x_estado('pausa') or []
            escritorios_bloqueados = set(en_atencion + en_pausa)            
            escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
            #print("iterar_escritorios_bloqueados")        
            supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

        if disponibles:= supervisor.filtrar_x_estado('disponible'):
            conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
            #print('iterar_escritorios_disponibles')
            supervisor.iterar_escritorios_disponibles(conectados_disponibles)
        matcher_emision_reloj.match(hora_actual)
        
        if not matcher_emision_reloj.match_emisiones.empty:
            emisiones      = matcher_emision_reloj.match_emisiones
            fila           = pd.concat([fila, emisiones])        
    
        if not fila.empty:
            if disponibles:= supervisor.filtrar_x_estado('disponible'):
                conectados_disponibles       = balancear_carga_escritorios(
                                                                            {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                                'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                            for k,v in supervisor.escritorios_ON.items() if k in disponibles}
                                                                            )            
                #print(conectados_disponibles)
                for un_escritorio in conectados_disponibles:
                    configuracion_atencion = supervisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                    fila_filtrada          = fila[fila['IdSerie'].isin(supervisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])
                    if  fila_filtrada.empty:
                            continue #
                    elif configuracion_atencion == "FIFO":
                        cliente_seleccionado = FIFO(fila_filtrada)
                        fila = remove_selected_row(fila, cliente_seleccionado)
                        supervisor.iniciar_atencion(un_escritorio, cliente_seleccionado)            
                    registros_atenciones = pd.concat([registros_atenciones, pd.DataFrame(cliente_seleccionado).T ])
                    
        fila['espera'] += 60    
    return registros_atenciones, fila
if __name__ == "__main__":
    #ejemplo:

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


    planificacion = {'0': [{'inicio': '08:00:11',
        'termino': None,
        'propiedades': {'skills' : get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,
                
            }}],
        '1': [{'inicio': '08:00:11',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '12': [{'inicio': '08:00:11',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '33': [{'inicio': '08:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '34': [{'inicio': '08:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '35': [{'inicio': '08:00:03',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '49': [{'inicio': '08:00:56',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series), 
            'configuracion_atencion':random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '50': [{'inicio': '08:00:56',
        'termino': None,
        'propiedades': {'skills': get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,

            }}],
        '51': [{'inicio': '08:00:56',
        'termino': None,
        'propiedades': {'skills':get_random_non_empty_subset(series),
            'configuracion_atencion': random.sample(modos, 1)[0],
            'porcentaje_actividad'  : np.random.randint(85, 90)/100,
            'atributos_series':atributos_series,
            }}]}

    hora_cierre               = '20:00:00'
    registros_atenciones, fila = simulador_v04(un_dia, planificacion, niveles_servicio_x_serie, hora_cierre)

    len(registros_atenciones) , len(fila)
#%%



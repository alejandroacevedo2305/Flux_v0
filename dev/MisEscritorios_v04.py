#%%
import numpy as np
from datetime import datetime
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import DatasetTTP, obtener_skills
import random
from itertools import count, islice
from typing import List
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
    #MisEscritorios_v03,        
)

from dev.simv04 import (
    reloj_rango_horario, 
    match_emisiones_reloj)
def actualizar_keys_tramo(original_dict, updates):    
    for key, value in updates.items():  # Loop through the keys and values in the updates dictionary.
        if key in original_dict:  # Check if the key from updates exists in the original dictionary.
            original_dict[key]['conexion']               = value['conexion']
            original_dict[key]['skills']                 = value['skills']
            original_dict[key]['configuracion_atencion'] = value['configuracion_atencion']
            original_dict[key]['atributos_series']       = value['atributos_series']            
            #original_dict[key]['duracion_pausas']        = value['duracion_pausas']
            #original_dict[key]['probabilidad_pausas']    = value['probabilidad_pausas']
            original_dict[key]['porcentaje_actividad']   = value['porcentaje_actividad']

class MisEscritorios_v04:
    
    def __init__(self,
                 inicio_tramo:  pd.Timestamp, 
                 fin_tramo:     pd.Timestamp,
                 planificacion: dict, 
                 conexiones:    dict = None,
                 niveles_servicio_x_serie=None,
                 ):
      
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        self.planificacion = planificacion

        self.escritorios = {k: {  # Dictionary comprehension starts; k is the key, and the value is another nested dictionary.
                                    'estado': 'disponible',  # Assigns the string 'disponible' to the key 'estado'.
                                    'tiempo_actual_disponible': 0,  # Initializes 'tiempo_actual_disponible' to 0.
                                    'skills': v[0]['propiedades'].get('skills'),  # Uses .get() to safely extract 'skills' from 'propiedades'.                                    
                                    'configuracion_atencion': v[0]['propiedades'].get('configuracion_atencion'),  # Similar to 'skills', safely extracts 'configuracion_atencion'.
                                    'contador_tiempo_disponible': iter(count(start=0, step=1)),  # Creates an iterator using Python's itertools.count, starting from 0 and incrementing by 1.
                                    'numero_de_atenciones': 0,  # Initializes 'numero_de_atenciones' to 0.                                    
                                    # Tries to safely extract 'porcentaje_actividad' from 'propiedades' using .get().
                                    'porcentaje_actividad': v[0]['propiedades'].get('porcentaje_actividad'),
                                    # Checks if 'porcentaje_actividad' exists, and if not, sets 'duracion_inactividad' to None.
                                    'duracion_inactividad': int(
                                        (1 - v[0]['propiedades'].get('porcentaje_actividad', 0)) * (fin_tramo - inicio_tramo).total_seconds() / 60
                                    ) if v[0]['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                    # Checks if 'porcentaje_actividad' exists, and if not, sets 'contador_inactividad' to None.
                                    'contador_inactividad': iter(islice(
                                        count(start=0, step=1),
                                        int((1 - v[0]['propiedades'].get('porcentaje_actividad', 0)) * (fin_tramo - inicio_tramo).total_seconds() / 60)
                                    )) if v[0]['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                    'duracion_pausas': (1, 4, 30),  # Tuple containing min, avg, and max pause durations based on historical data.
                                    'probabilidad_pausas': .5,  # Probability that a pause will occur, again based on historical data.
                                    'numero_pausas': None,  # Initializes 'numero_pausas' to None.
                                    'atributos_series': v[0]['propiedades'].get('atributos_series')
                                        if v[0]['propiedades'].get('atributos_series') is not None else None,
                                    'prioridades': {dict_series['serie']: dict_series['prioridad'] for dict_series in v[0]['propiedades'].get('atributos_series')}
                                        if v[0]['propiedades'].get('atributos_series') is not None else None,
                                    'conexion': True,                                       
                                }
                                for k, v in planificacion.items()}  # The loop iterates over each key-value pair in self.planificacion.
        
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}        
        
    def aplicar_planificacion(self, hora_actual, planificacion):
        
        propiedades_tramo = dict()
        for idEsc, un_escritorio in planificacion.items():
            for un_tramo in un_escritorio:
                on_off = hora_actual >= un_tramo['inicio'] and (lambda: 
                    hora_actual <= un_tramo['termino'] if un_tramo['termino'] is not None else True)()
                #print(f"{idEsc}: {on} {hora_actual} >= {un_tramo['inicio']} and {hora_actual} <= {un_tramo['termino']}")           
                propiedades_tramo = propiedades_tramo | {idEsc: {
                                                'conexion': on_off, #se setean conexiones
                                                'skills':                 un_tramo['propiedades']['skills'],
                                                'configuracion_atencion': un_tramo['propiedades']['configuracion_atencion'],
                                                'atributos_series':       un_tramo['propiedades']['atributos_series'],
                                                #'duracion_pausas':        un_tramo['propiedades']['duracion_pausas'],
                                                #'probabilidad_pausas':    un_tramo['propiedades']['probabilidad_pausas'],
                                                'porcentaje_actividad':   un_tramo['propiedades']['porcentaje_actividad'],                                                 
                                                }}       
                
        actualizar_keys_tramo(self.escritorios_ON, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.
        actualizar_keys_tramo(self.escritorios_OFF, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.

        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion({**self.escritorios_ON, **self.escritorios_OFF})
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)  #En los escritorios desconectados, resetear los iteradores que cuentan el tiempo de bloqueo y  poner los escritorios en estados disponible, así quedan listo para volver a conectarse
        self.escritorios_ON                       = poner_pasos_alternancia_v02(self.escritorios_ON, pasos_alternancia_v02)

    def iniciar_atencion(self, escritorio, cliente_seleccionado):
        # iterar los escritorios y emisiones
        #for escr_bloq, emi in zip(escritorios_a_bloqueo, emision):
            
            #extraer los minutos que dura la atención asociada a la emision
        minutos_atencion = round(cliente_seleccionado.T_Ate/60)#round((cliente_seleccionado.T_Ate - cliente_seleccionado.FH_AteIni).total_seconds()/60)
            #reescribir campos:
            
        self.escritorios_ON[escritorio]['contador_tiempo_atencion'] = iter(islice(count(start=0, step=1), minutos_atencion))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']           = 'atención'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_atencion']  = minutos_atencion#tiempo de atención     
        self.escritorios[escritorio]['numero_de_atenciones'] += 1 #se guarda en self.escritorios para que no se resetee.
        self.escritorios_ON[escritorio]['numero_de_atenciones'] = self.escritorios[escritorio]['numero_de_atenciones'] 


    def filtrar_x_estado(self, state: str):     
        #obtener estados
        self.estados = {escr_i: {'estado': propiedades['estado'], 'configuracion_atencion': 
                        propiedades['configuracion_atencion']} for escr_i, propiedades in self.escritorios_ON.items()} 
        #extraer por disponibilidad    
        if disponibilidad := [
            key for key, value in self.estados.items() if value['estado'] == state
        ]:
            return disponibilidad
        else:
            #print(f"No hay escritorio {state}")
            return False
    def iniciar_pausa(self, escritorio, tipo_inactividad:str = "Porcentaje", generador_pausa = generate_integer):
        # sourcery skip: extract-method, move-assign
      
        if tipo_inactividad == "Porcentaje":            
            self.escritorios_ON[escritorio]['estado'] = 'pausa'            
        else:
            min_val, avg_val, max_val = self.escritorios_ON[escritorio]['duracion_pausas']
            probabilidad_pausas       = self.escritorios_ON[escritorio]['probabilidad_pausas']
            minutos_pausa             = generador_pausa(min_val, avg_val, max_val, probabilidad_pausas)

            self.escritorios_ON[escritorio]['contador_tiempo_pausa'] = iter(islice(count(start=0, step=1), minutos_pausa))#nuevo contador de minutos limitado por n_minutos
            self.escritorios_ON[escritorio]['estado']                = 'pausa'#estado
            self.escritorios_ON[escritorio]['minutos_pausa']         = minutos_pausa#tiempo 
    def iniciar_tiempo_disponible(self,escritorio):
        self.escritorios_ON[escritorio]['contador_tiempo_disponible'] = iter(count(start=0, step=1))
        self.escritorios_ON[escritorio]['estado']                     = 'disponible'#     
    def iterar_escritorios_bloqueados(self, escritorios_bloqueados: List[str], tipo_inactividad:str = "Porcentaje"):

        for escri_bloq in escritorios_bloqueados:
            #ver si está en atención:
            if self.escritorios_ON[escri_bloq]['estado'] == 'atención':                
                #avanzamos en un minuto el tiempo de atención
                tiempo_atencion = next(self.escritorios_ON[escri_bloq]['contador_tiempo_atencion'], None)
                #si terminó la atención
                if tiempo_atencion is None: 
                    #iniciar pausa 
                    self.iniciar_pausa(escri_bloq, tipo_inactividad)
            #si el escritorio está en pausa:            
            elif self.escritorios_ON[escri_bloq]['estado'] == 'pausa':
                  #chequeamos si la inactividad es por pocentaje o pausas históricas
                 if tipo_inactividad == "Porcentaje":
                   #Avanzamos el contador de inactividad en un minuto
                   tiempo_inactividad = next(self.escritorios_ON[escri_bloq]['contador_inactividad'],None)
                   #si termina el tiempo de inactividad
                   if tiempo_inactividad is None:
                     #pasa a estado disponible
                     self.iniciar_tiempo_disponible(escri_bloq)
                 else: #pausas históricas                 
                    #iteramos contador_tiempo_pausa:
                    tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
                    if tiempo_pausa is None: 
                        #si termina tiempo en pausa pasa a estado disponible
                        self.iniciar_tiempo_disponible(escri_bloq)
    def iterar_escritorios_disponibles(self, escritorios_disponibles: List[str]):
        
        for escri_dispon in escritorios_disponibles:               
            #avanzamos en un minuto el tiempo que lleva disponible.
            tiempo_disponible = next(self.escritorios_ON[escri_dispon]['contador_tiempo_disponible'], None)
            if tiempo_disponible is not None:
            #guardar el tiempo que lleva disponible
                self.escritorios_ON[escri_dispon]['tiempo_actual_disponible'] = tiempo_disponible



                
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
hora_cierre               = '17:00:00'    
reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
registros_atenciones  = pd.DataFrame()
matcher_emision_reloj = match_emisiones_reloj(un_dia)

supervisor            = MisEscritorios_v04(inicio_tramo      = un_dia['FH_Emi'].min(),
                                    fin_tramo                = un_dia['FH_Emi'].max(),
                                    planificacion            = planificacion,
                                    niveles_servicio_x_serie = niveles_servicio_x_serie)
fecha                = un_dia.FH_Emi.iloc[0].date()
registros_atenciones = pd.DataFrame()
fila                 = pd.DataFrame()
# pd.Timestamp(f"{fecha} {hora_actual}")
for hora_actual in reloj:
    print(hora_actual)
    supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion)
    print(supervisor.escritorios_ON.keys(), supervisor.escritorios_OFF.keys())
    print(f"%%%%%%%%%%% disponible{supervisor.filtrar_x_estado('disponible')}")
    print(f"%%%%%%%%%%% atención {supervisor.filtrar_x_estado('atención')} pausa {supervisor.filtrar_x_estado('pausa')}")
    
    if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
        en_atencion            = supervisor.filtrar_x_estado('atención') or []
        en_pausa               = supervisor.filtrar_x_estado('pausa') or []
        escritorios_bloqueados = set(en_atencion + en_pausa)            
        escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
        print("iterar_escritorios_bloqueados")        
        supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

    if disponibles:= supervisor.filtrar_x_estado('disponible'):
        conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
        print('iterar_escritorios_disponibles')
        supervisor.iterar_escritorios_disponibles(conectados_disponibles)

    matcher_emision_reloj.match(hora_actual)
    
    if not matcher_emision_reloj.match_emisiones.empty:
        emisiones      = matcher_emision_reloj.match_emisiones
        fila           = pd.concat([fila, emisiones])  
            
    #fila['espera'] += 60
    if not fila.empty:
        if disponibles:= supervisor.filtrar_x_estado('disponible'):
            conectados_disponibles       = balancear_carga_escritorios(
                                                                        {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                            'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                        for k,v in supervisor.escritorios_ON.items() if k in disponibles}
                                                                        )
            print(f"conectados_disponibles: {conectados_disponibles}")
            for un_escritorio in conectados_disponibles:
                configuracion_atencion = supervisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                #print(f"configuracion_atencion {configuracion_atencion}")
                fila_filtrada          = fila[fila['IdSerie'].isin(supervisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])
                #print(f"fila_filtrada: {fila_filtrada}")
                if  fila_filtrada.empty:
                        continue
                elif configuracion_atencion == "FIFO":
                     cliente_seleccionado = FIFO(fila_filtrada)
                     #print(f"cliente_seleccionado: {cliente_seleccionado}")
                     fila = remove_selected_row(fila, cliente_seleccionado)
                     supervisor.iniciar_atencion(un_escritorio, cliente_seleccionado)            
                     registros_atenciones = pd.concat([registros_atenciones, pd.DataFrame(cliente_seleccionado).T ])
    print(f"-----------------disponible{supervisor.filtrar_x_estado('disponible')}")
    print(f"----------------atención {supervisor.filtrar_x_estado('atención')} pausa {supervisor.filtrar_x_estado('pausa')}") 
    fila['espera'] += 60
    #print(fila)  
    
        
len(registros_atenciones), len(fila)
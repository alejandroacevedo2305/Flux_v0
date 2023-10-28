#%%

from typing import Dict, List, Tuple
import random
from copy import deepcopy
from itertools import count, islice
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  

def unir_values_en_tupla(data_dict: Dict[int, List[int]], label_dict: Dict[int, str]) -> Dict[int, Tuple[List[int], str]]:

    return {key: (value, label_dict.get(key, '')) for key, value in data_dict.items()}

def actualizar_conexiones(original_dict, update_dict):  

    # Iterate through each key-value pair in the update_dict
    for key, value in update_dict.items():
        # Check if the key exists in the original_dict

        if key in original_dict:
            # Update the 'conexion' field with the new boolean value
            original_dict[key]['conexion'] = value            

    return deepcopy(original_dict)
class MisEscritorios:
    
    def __init__(self, skills:Dict[str, List[int]], configuraciones: Dict[str, str], conexiones: Dict[str, bool] = None, niveles_servicio_x_serie:dict =None):
        """_summary_
        Args:
            skills (Dict[str, List[int]]): Series cargadas para cada escritorio.
            configuraciones (Dict[str, str]): Configuraciones de atencion para cada escritorio. 
            conexiones (Dict[str, bool], optional): estado de la conexion de los escritorios. Defaults to None.
        """        
        self.skills                 = skills
        self.configuraciones        = configuraciones
        #anexar las configuraciones de atenciones al dicionario de las skills usando unir_values_en_tupla. 
        #los key-values por ejemplo quedan así 'escritorio_2': ([2, 5, 6, 7, 13, 16], 'RR')    
        
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        
        self.skills_configuraciones = unir_values_en_tupla(self.skills, self.configuraciones)      
        #iteramos self.skills_configuraciones para armar el dicionario con los escritorios
        self.escritorios            = {key: #escritorio i
                                    {
                                    "skills":series,#series, #series cargadas en el escritorio i
                                    'contador_bloqueo':None, #campo vacío donde se asignará un iterador que cuenta los minutos que el escritorio estará ocupado en servicio 
                                    'minutos_bloqueo':None, #campo vacío donde se asignarán los minutos que se demora la atención 
                                    'estado':'disponible', #si el escritorio está o no disponible para atender   
                                    'configuracion_atencion':config, #configuración del tipo de antención, FIFO, RR, etc.
                                    'pasos_alternancia': None, #objeto que itera en la tabla con pasos de alternancia.
                                    'conexion':None, #campo vacío donde se asignará el estado de la conexión del escritorio (ON/OFF)
                                    'numero_de_atenciones':0, #min 
                                    'numero_pausas':       None,
                                    'numero_desconexiones': None,
                                    'tiempo_actual_disponible':   0, #max
                                    'tiempo_actual_en_atención':  None,
                                    'tiempo_actual_pausa':        None,
                                    'tiempo_actual_desconectado': None,
                                    'contador_tiempo_disponible': iter(count(start=0, step=1)),
                                    'duracion_pausas': (1, 4, 47), #min, avg, max
                                    'probabilidad_pausas':.5, #probabilidad que la pausa ocurra
                                    } for key,(series, config) in self.skills_configuraciones.items()}        
        if not conexiones:
        #     #si no se provee el estado de los conexiones se asumen todas como True (todos conectados):
             conexiones                         = {f"{key}": random.choices([True, False], [1, 0])[0] for key in self.escritorios}
        self.escritorios                        = actualizar_conexiones(self.escritorios, conexiones)       
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.nuevos_escritorios_programados     = []
        self.registros_escritorios              = []
        
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
#%%
configuraciones = {k:np.random.choice(["Alternancia", "FIFO", "Rebalse"], p=[.5,.25,.25]) for k in skills}
svisor          = MisEscritorios(skills= skills, configuraciones = configuraciones, niveles_servicio_x_serie = niveles_servicio_x_serie)
svisor.escritorios

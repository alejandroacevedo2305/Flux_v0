#%%
import itertools
import random
import pandas as pd
from typing import Dict
def one_cycle_iterator(series, start_pos):
    part_one = series[start_pos+1:]
    part_two = series[:start_pos]
    complete_cycle = pd.concat([part_one, part_two])
    return iter(complete_cycle)

def map_priority_to_steps(input_data: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:

    unique_priorities = sorted(set(val['prioridad'] for val in input_data.values()))
    
    priority_to_steps = {priority: step for priority, step in zip(unique_priorities, reversed(unique_priorities))}
    
    for key, inner_dict in input_data.items():
        inner_dict['pasos'] = priority_to_steps[inner_dict['prioridad']]
    
    return input_data

def create_multiindex_df(data_dict):

    multi_index_list = []
    sorted_data = sorted(data_dict.items(), key=lambda x: x[1]['prioridad'])
    
    priority_groups = {}
    for k, v in sorted_data:
        priority = v['prioridad']
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append((k, v))
    
    position = 0  # To enumerate rows
    for priority, items in priority_groups.items():
        
        random.shuffle(items)
        
        for k, v in items:
            pasos = v['pasos']
            
            # Add each entry 'pasos' number of times
            for _ in range(pasos):
                multi_index_list.append((position, priority, k))
                position += 1
                
    multi_index = pd.MultiIndex.from_tuples(
        multi_index_list, 
        names=["posicion", "prioridad", "serie"]
    )
    
    # Initialize the DataFrame
    df = pd.DataFrame(index=multi_index)
    
    return df

def rank_by_magnitude(arr):
    sorted_indices = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    rank_arr = [0] * len(arr)
    rank = 1  # Initialize rank
    for i in range(len(sorted_indices)):
        index, value = sorted_indices[i]
        if i > 0 and value == sorted_indices[i - 1][1]:
            pass  # Don't increment the rank
        else:
            rank = i + 1  # Set new rank
        rank_arr[index] = rank  # Store the rank at the original position
    return rank_arr

def calcular_prioridad(porcentaje, espera, alpha:float=2, beta:float=1):
    
    return ((porcentaje**alpha)/(espera**beta))

def generar_pasos_para_alternancia(niveles_servicio_x_serie, alpha:float=2, beta:float=1):

    priority_levels     = rank_by_magnitude([calcular_prioridad(porcentaje, espera, alpha, beta)
                                             for (porcentaje, espera) in niveles_servicio_x_serie.values()])
    niveles_de_servicio = [{'porcentaje':porcen, "espera":espera, "prioridad" : priori} 
                           for (porcen, espera), priori in zip(niveles_servicio_x_serie.values(),priority_levels)]

    return create_multiindex_df(
                    map_priority_to_steps({s: niveles_de_servicio[i] for i, s in enumerate(list(niveles_servicio_x_serie.keys()))})
                    ).reset_index(drop=False)
    

from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
#from src.optuna_utils import *
#from src.simulador_v02 import *  
import random


#dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
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
#generar_pasos_para_alternancia(niveles_servicio_x_serie)


rank_by_magnitude([calcular_prioridad(porcentaje, espera)
                                             for (porcentaje, espera) in niveles_servicio_x_serie.values()])

#%% 
class pasos_alternancia():
    
    """ 
    generar pasos de alternancia
    """
    
    def __init__(self, niveles_servicio_x_serie, skills, alpha:float=2, beta:float=1):
        
        self.pasos             = generar_pasos_para_alternancia(niveles_servicio_x_serie, alpha, beta)
        self.pasos             = self.pasos[self.pasos['serie'].isin(skills)].reset_index(drop=True)
        self.pasos['posicion'] = self.pasos.index
        self.iterador_posicion = itertools.cycle(self.pasos.posicion)                   
    def buscar_cliente(self, fila_filtrada):        
        self.posicion_actual        = self.pasos.iloc[next(self.iterador_posicion)]
        serie_en_la_posicion_actual = self.posicion_actual.serie
        if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].empty:
            #print(f"serie_en_la_posicion_actual  {self.posicion_actual.serie} coincidió con cliente(s)")
            #de los clientes q coincidieron retornar el que llegó primero
            return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
        else:
            #print(f"serie_en_la_posicion_actual no conincidió, buscando en otros pasos")
            start_position                = self.posicion_actual.posicion
            single_cycle_iterator_x_pasos = one_cycle_iterator(self.pasos.serie, start_position)            
            for serie_en_un_paso in single_cycle_iterator_x_pasos:                
                if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].empty:
                    #print(f"serie {serie_en_un_paso} en otro paso coincidió con cliente")
                    return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
            else:
                raise ValueError(
                "Las series del escritorio no coinciden con la serie del cliente. No se puede atender. ESTO NO DE DEBERIA PASAR, EL FILTRO TIENE QUE ESTAR FUERA DEL OBJETO.")
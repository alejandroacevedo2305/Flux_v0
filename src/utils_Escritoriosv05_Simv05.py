#%%
import itertools
from copy import deepcopy
import pandas as pd
import random

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
            
def separar_por_conexion(original_dict):  

    # Create deep copies of the original dictionary
    true_dict = deepcopy(original_dict)
    false_dict = deepcopy(original_dict)    

    # Create lists of keys to remove
    keys_to_remove_true = [key for key, value in true_dict.items() if value.get('conexion') is not True]
    keys_to_remove_false = [key for key, value in false_dict.items() if value.get('conexion') is not False]    

    # Remove keys from the deep-copied dictionaries
    for key in keys_to_remove_true:
        del true_dict[key]
    for key in keys_to_remove_false:
        del false_dict[key]
    return true_dict, false_dict

def reset_escritorios_OFF(desk_dict):
    # Initialize the update dictionary
    update_dict = {'contador_bloqueo': None,
                   'minutos_bloqueo': None,
                   'estado': 'disponible',
                   }    
    # Loop through each key-value pair in the input dictionary
    for desk, info in desk_dict.items():
        # Update the inner dictionary with the update_dict values
        info.update(update_dict)        
    return desk_dict

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

def one_cycle_iterator(series, start_pos):
    part_one = series[start_pos+1:]
    part_two = series[:start_pos]
    complete_cycle = pd.concat([part_one, part_two])
    return iter(complete_cycle)

def generar_pasos_para_alternancia_v02(atributos_series):

    return create_multiindex_df({atr_dict['serie']:
                        {'porcentaje' :atr_dict['sla_porcen'], 
                        'espera'      :atr_dict['sla_corte']/60, 
                        'prioridad'   :atr_dict['prioridad'],
                        'pasos'       :atr_dict['pasos']}
                        for atr_dict in atributos_series}).reset_index(drop=False)   

class pasos_alternancia_v02():
    def __init__(self, atributos_series, skills):
        
        self.pasos             = generar_pasos_para_alternancia_v02(atributos_series)
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


def poner_pasos_alternancia_v02(escritorios: dict, class_to_instantiate):
    """ 
    ejemplo:
        self.escritorios_ON  = poner_pasos_alternancia_v02(self.escritorios_ON, pasos_alternancia_v02)
    """
    for key, value in escritorios.items():
        # Check if 'configuracion_atencion' is 'Alternancia'
        if value.get('configuracion_atencion') == 'Alternancia':
            # Instantiate the class and assign it to 'pasos_alternancia'
            value['pasos_alternancia'] = class_to_instantiate(atributos_series=escritorios[key]['atributos_series'] , skills = escritorios[key]['skills'])

    return escritorios
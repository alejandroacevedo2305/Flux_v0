#%%
from datetime import timedelta

from dataclasses import dataclass
import pandas as pd
from datetime import date
from collections import defaultdict
import numpy as np
from datetime import datetime
from typing import List
import math
from   itertools                       import (count, islice)
import itertools
import logging
import random
from copy import deepcopy

import random


def get_permutations_array(length):
    # Generate the list based on the input length
    items = list(range(1, length + 1))
    
    # Get all permutations of the list
    all_permutations = list(itertools.permutations(items))
    return np.array(all_permutations)
def plan_unico(lst_of_dicts):
    new_list = []    
    # Initialize a counter for the new globally unique keys
    global_counter = 0    
    # Loop through each dictionary in the original list
    for dct in lst_of_dicts:        
        # Initialize an empty dictionary to hold the key-value pairs of the original dictionary but with new keys
        new_dct = {}        
        # Loop through each key-value pair in the original dictionary
        for key, value in dct.items():            
            # Assign the value to a new key in the new dictionary
            new_dct[global_counter] = value            
            # Increment the global counter for the next key
            global_counter += 1        
        # Append the newly created dictionary to the list
        new_list.append(new_dct)
        
    return {str(k): v for d in new_list for k, v in d.items()}

def extract_min_value_keys(input_dict):
    output_dict = {}  # Initialize an empty dictionary to store the result
    # Loop through each item in the input dictionary
    for workforce, values_dict in input_dict.items():
        max_key = min(values_dict, key=values_dict.get)  # Find the key with the maximum value in values_dict
        max_value = values_dict[max_key]  # Get the maximum value
        output_dict[workforce] = (max_key, max_value)  # Add the key and value to the output dictionary
    return output_dict  # Return the output dictionary

def get_time_intervals(df, n, porcentaje_actividad:float=1):
    assert porcentaje_actividad <= 1 
    assert porcentaje_actividad > 0 

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
    adjusted_intervals = [(start_time + 1 * (1 - porcentaje_actividad) * (end_time - start_time), end_time) for start_time, end_time in intervals]
    # Step 5: Format the intervals as requested
    formatted_intervals = [(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')) for start_time, end_time in adjusted_intervals]
    
    return formatted_intervals
def partition_dataframe_by_time_intervals(df, intervals):
    partitions = []    
    # Loop over each time interval to create a partition
    for start, end in intervals:
        # Convert the time strings to Pandas time objects
        start_time = pd.to_datetime(start).time()
        end_time = pd.to_datetime(end).time()        
        # Create a mask for filtering the DataFrame based on the time interval
        mask = (df['FH_Emi'].dt.time >= start_time) & (df['FH_Emi'].dt.time <= end_time)        
        # Apply the mask to the DataFrame to get the partition and append to the list
        partitions.append(df[mask])        
    return partitions

def extract_highest_priority_and_earliest_time_row(df, priorities):
    # Create a copy of the DataFrame to avoid modifying the original unintentionally
    df_copy = df.copy()

    # Set the 'Priority' column
    df_copy['Priority'] = df_copy['IdSerie'].map(priorities)
    
    # Sort the DataFrame first based on the Priority and then on the 'FH_Emi' column
    df_sorted = df_copy.sort_values(by=['Priority', 'FH_Emi'], ascending=[True, True])

    return df_sorted.iloc[0]


def remove_selected_row(df, selected_row):
    """
    Removes the specified row from the DataFrame.
    """
    # If DataFrame or selected_row is None, return original DataFrame
    if df is None or selected_row is None:
        return df
    
    # Get the index of the row to be removed
    row_index = selected_row.name
    
    # Remove the row using the 'drop' method
    updated_df = df.drop(index=row_index)
    
    return updated_df
def FIFO(df):
    if df is None or df.empty:
        return None   
    min_time = df['FH_Emi'].min()
    earliest_rows = df[df['FH_Emi'] == min_time]
    if len(earliest_rows) > 1:
        selected_row = earliest_rows.sample(n=1)
    else:
        selected_row = earliest_rows   
    return selected_row.iloc[0]

def balancear_carga_escritorios(desk_dict):
    sorted_desks = sorted(desk_dict.keys(), key=lambda x: (desk_dict[x]['numero_de_atenciones'], -desk_dict[x]['tiempo_actual_disponible']))
    
    return sorted_desks
def generate_integer(min_val:int=1, avg_val:int=4, max_val:int=47, probabilidad_pausas:float=.5):
    # Validate probabilidad_pausas
    if probabilidad_pausas < 0 or probabilidad_pausas > 1:
        return -1
        # Validate min_val and max_val
    if min_val > max_val:
        return -1
        # Validate avg_val
    if not min_val <= avg_val <= max_val:
        return -1
        # Initialize weights with 1s
    weights = [1] * (max_val - min_val + 1)
        # Calculate the distance of each possible value from avg_val
    distance_from_avg = [abs(avg_val - i) for i in range(min_val, max_val + 1)]
        # Calculate the total distance for normalization
    total_distance = sum(distance_from_avg)
        # Update weights based on distance from avg_val
    if total_distance == 0:
        weights = [1] * len(weights)
    else:
        weights = [(total_distance - d) / total_distance for d in distance_from_avg]
        # Generate a random integer based on weighted probabilities
    generated_integer = random.choices(range(min_val, max_val + 1), weights=weights, k=1)[0]
        # Determine whether to return zero based on probabilidad_pausas
    if random.random() > probabilidad_pausas:
        return 0
    
    return generated_integer


def actualizar_keys_tramo(original_dict, updates):    
    for key, value in updates.items():  # Loop through the keys and values in the updates dictionary.
        if key in original_dict:  # Check if the key from updates exists in the original dictionary.
            original_dict[key]['conexion']               = value['conexion']
            original_dict[key]['skills']                 = value['skills']
            original_dict[key]['configuracion_atencion'] = value['configuracion_atencion']
            original_dict[key]['pasos_alternancia']       = value['pasos_alternancia']            
            original_dict[key]['prioridades']           = value['prioridades']
            original_dict[key]['pasos']                 = value['pasos']
            original_dict[key]['porcentaje_actividad']   = value['porcentaje_actividad']
            original_dict[key]['inicio']                 = value['inicio']
            original_dict[key]['termino']                = value['termino']
            original_dict[key]['duracion_pausas']                = value['duracion_pausas']
            original_dict[key]['numero_de_atenciones']                = value['numero_de_atenciones']
            original_dict[key]['porcentaje_actividad']                = value['porcentaje_actividad']
            original_dict[key]['duracion_inactividad']                = value['duracion_inactividad']
            original_dict[key]['contador_inactividad']                = value['contador_inactividad']
            original_dict[key]['probabilidad_pausas']                = value['probabilidad_pausas']



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
class pasos_alternancia_v03():
    def __init__(self, prioridades, pasos):
        
        assert prioridades.keys() == pasos.keys()

        dict1       = {k: {'prioridad': v} for k,v in prioridades.items()}
        dict2       = {k: {'pasos': v} for k,v in pasos.items()}
        merged_dict = {}
        for key in dict1:
            if key in dict2:
                merged_dict[key] = {**dict1[key], **dict2[key]}
        self.pasos             = create_multiindex_df(merged_dict).reset_index(drop=False)
        self.pasos['posicion'] = self.pasos.index
        self.iterador_posicion = itertools.cycle(self.pasos.posicion)
        
                           
    def buscar_cliente(self, fila_filtrada):        
        self.posicion_actual        = self.pasos.iloc[next(self.iterador_posicion)]
        serie_en_la_posicion_actual = self.posicion_actual.serie
        if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].empty:
            logging.info(f"modo alternancia: serie_en_la_posicion_actual  {self.posicion_actual.serie} coincidió con cliente(s)")
            #de los clientes q coincidieron retornar el que llegó primero
            return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
        else:
            logging.info(f"modo alternancia: serie_en_la_posicion_actual ({serie_en_la_posicion_actual}) no conincidió con la fila filtrada {tuple(fila_filtrada.IdSerie)}, buscando en otros pasos")
            start_position                = self.posicion_actual.posicion
            logging.info(f"modo alternancia: instanciando one_cycle_iterator con pasos de series = {tuple(self.pasos.serie)} start_position = {start_position}")
            single_cycle_iterator_x_pasos = one_cycle_iterator(self.pasos.serie, start_position)            
            for serie_en_un_paso in single_cycle_iterator_x_pasos: 
                logging.info(f"modo alternancia: iterando serie_en_un_paso -> {serie_en_un_paso}")
                if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].empty:
                    logging.info(f"modo alternancia: serie {serie_en_un_paso} iterando en otro paso coincidió con cliente")
                    return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
            else:
                logging.info(f"modo alternancia: fila_filtrada.IdSerie= {fila_filtrada.IdSerie} serie_en_un_paso: {serie_en_un_paso} en otro paso coincidió con cliente")
                raise ValueError(
                "Las series del escritorio no coinciden con la serie del cliente. No se puede atender. ESTO NO DE DEBERIA PASAR, EL FILTRO TIENE QUE ESTAR FUERA DEL OBJETO.")

from collections import defaultdict
import numpy as np
 
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

class atributos_de_series():
    def __init__(self, ids_series:list):      

        self.ids_series         = np.array(ids_series, dtype=int)
        self.atributos_x_series = [
                                    {'serie': s} for s in ids_series
                                    ]
    def atributo(self, sla_input_user, atributo, low, high):
        self.atributos_x_series = [atr_dict | 
                {atributo: np.random.randint(low,high) if sla_input_user is None else sla_p, 
                } for
                atr_dict, sla_p in 
                zip(self.atributos_x_series, itertools.repeat( None)  if sla_input_user is None else sla_input_user)]
        
        return  self
    def prioridades(self, prioridad_input_user):
        if prioridad_input_user is None:
            prioridades_autom    = rank_by_magnitude([calcular_prioridad(s['sla_porcen'], s['sla_corte']) for s in self.atributos_x_series])
            self.atributos_x_series = [s|{'prioridad':p} for s,p in zip(self.atributos_x_series, prioridades_autom)]
            
        else:
            self.atributos_x_series = [s|{'prioridad':p} for s,p in zip(self.atributos_x_series, prioridad_input_user)]
        return  self
            

def atributos_x_serie(ids_series, sla_porcen_user=None, sla_corte_user=None, pasos_user=None, prioridades_user=None):    

    att_x_s = atributos_de_series(ids_series = ids_series)

    return att_x_s.atributo(sla_porcen_user, 'sla_porcen', 70, 85).atributo(
            sla_corte_user, 'sla_corte', 10*60, 30*60).atributo(
                pasos_user, 'pasos',1,5).prioridades(prioridades_user).atributos_x_series

def obtener_skills(un_dia):   

    skills_defaultdict  = defaultdict(list)
    for index, row in un_dia.iterrows():

        skills_defaultdict[row['IdEsc']].append(row['IdSerie'])
    for key in skills_defaultdict:
        skills_defaultdict[key] = list(set(skills_defaultdict[key]))
        
    skills = dict(skills_defaultdict)   
    return {f"{k}": v for k, v in skills.items()}#

from dataclasses import dataclass
from datetime import date, time, datetime

@dataclass
class DatasetTTP:
    atenciones : pd.DataFrame
    atenciones_agg : pd.DataFrame
    atenciones_agg_dia : pd.DataFrame

    @staticmethod
    def _reshape_df_atenciones(df : pd.DataFrame, resample = "60t") -> 'pd.DataFrame':

        resampler = ( df
            .set_index("FH_Emi")
            .groupby(by=["IdOficina","IdSerie"])
            .resample(resample)
        )

        medianas = resampler.median()[["T_Esp","T_Ate"]]

        medianas["Demanda"] = ( resampler
            .count().IdSerie
            .rename("Demanda") # Esto es una serie, asi que solo tiene un nombre
            # .reset_index(level="IdSerie") #
        )

        return medianas[["Demanda","T_Esp","T_Ate"]]

    @staticmethod
    def _atenciones_validas( df ) -> 'pd.DataFrame':
        """Un helper para limpiar la data en base a ciertas condiciones logicas"""

        # TODO: implementar loggin de cosas (n) que se eliminan en estas atenciones raras
        # TODO: Posiblemente limpiar esto en un unico . . ., o usar Polars
        df = df.dropna( how = 'any' ) # Inmediatamente elimina los NaN
        df = df[~(df["FH_Emi"] == df["FH_AteFin"])]
        df = df[~(df["FH_AteIni"] == df["FH_Emi"])]
        df = df[~(df["FH_AteIni"] == df["FH_AteFin"])]
        df = df[~(df["FH_Emi"] > df["FH_Llama"])]
        df = df[~((df["FH_Llama"] - df["FH_Emi"]) > pd.Timedelta(hours=12))]
        df = df[~((df["FH_AteIni"] - df["FH_Llama"]) > pd.Timedelta(hours=1))]
        df = df[~((df["FH_AteFin"] - df["FH_AteIni"]) < pd.Timedelta(seconds=5))]

        return df

    @staticmethod
    def desde_csv_atenciones( csv_path : str ) -> 'DatasetTTP':
        df = pd.read_csv( csv_path, 
            usecols = ["IdOficina","IdSerie","IdEsc","FH_Emi","FH_Llama","FH_AteIni","FH_AteFin"], 
            parse_dates = [3, 4, 5, 6]
            ).astype({
                "IdOficina" : "Int32",
                "IdSerie" : "Int8",
                "IdEsc" : "Int8",
                "FH_Emi" : "datetime64[s]",
                "FH_Llama" : "datetime64[s]",
                "FH_AteIni" : "datetime64[s]",
                "FH_AteFin" : "datetime64[s]",
            })
        
        df = DatasetTTP._atenciones_validas(df) # Corre la funcion de limpieza

        df["T_Esp"] = ( df["FH_AteIni"] - df["FH_Emi"] ).dt.seconds 
        df["T_Ate"] = ( df["FH_AteFin"] - df["FH_AteIni"] ).dt.seconds 

        return DatasetTTP( 
            atenciones = df, 
            atenciones_agg = DatasetTTP._reshape_df_atenciones( df ),
            atenciones_agg_dia = DatasetTTP._reshape_df_atenciones( df, resample = '1D' )
         )


    @staticmethod
    def desde_sql_server() -> 'DatasetTTP':

        sql_query = """-- La verdad esto podria ser un View, pero no sé si eso es mas rapido en MS SQL --
        SELECT
            -- TOP 5000 -- Por motivos de rendimiento
            IdOficina, -- Sucursal fisica
            IdSerie,   -- Motivo de atencion
            IdEsc,     -- Escritorio de atencion

            FH_Emi,    -- Emision del tiquet, con...
            -- TIEMPO DE ESPERA A REDUCIR --
            FH_Llama,  -- ... hora de llamada a resolverlo,  ...
            FH_AteIni, -- ... inicio de atencion, y
            FH_AteFin  -- ... termino de atencion

        FROM Atenciones -- Unica tabla que estamos usando por ahora -- MOCK
        -- FROM dbo.Atenciones -- Unica tabla que estamos usando por ahora -- REAL

        WHERE
            (IdSerie IN (10, 11, 12, 14, 5)) AND 
            (FH_Emi > '2023-01-01 00:00:00') AND     -- De este año
            (FH_Llama IS NOT NULL) AND (Perdido = 0) -- CON atenciones
            
        ORDER BY FH_Emi DESC; -- Ordenado de mas reciente hacia atras (posiblemente innecesario)"""
        raise NotImplementedError("Este metódo aún no está implementado.")


    @staticmethod
    def desde_csv_batch() -> 'DatasetTTP':
        raise NotImplementedError("Este metódo aún no está implementado.")

    def un_dia(self, dia : date):
        """Retorna las atenciones de un dia historico"""

        inicio_dia = pd.Timestamp( dia )
        fin_dia = pd.Timestamp( dia ) + pd.Timedelta(days=1)

        return self.atenciones[ (inicio_dia <= self.atenciones["FH_Emi"]) & (self.atenciones["FH_Emi"] <= fin_dia) ]
    
def get_random_non_empty_subset(lst):
    # Step 1: Check that the input list is not empty
    if not lst:
        return "Input list is empty, cannot generate a subset."
    
    # Step 2: Generate a random integer for the size of the subset.
    # Since we want a non-empty subset, we select a size between 1 and len(lst).
    subset_size = random.randint(1, len(lst))
    
    # Step 3: Randomly select 'subset_size' unique elements from the list
    return random.sample(lst, subset_size)

def generar_planificacion(un_dia,  modos: list    = ['FIFO','Alternancia', 'Rebalse']):
    skills   = obtener_skills(un_dia)
    series   = sorted(list({val for sublist in skills.values() for val in sublist}))
    #modos    = ['FIFO','Alternancia', 'Rebalse']

    atributos_series = atributos_x_serie(ids_series=series, 
                                        sla_porcen_user=None, 
                                        sla_corte_user=None, 
                                        pasos_user=None, 
                                        prioridades_user=None)
    niveles_servicio_x_serie = {atr_dict['serie']:
                            (atr_dict['sla_porcen']/100, atr_dict['sla_corte']/60) 
                            for atr_dict in atributos_series}
    planificacion = {
        
            '0': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            '1': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            
            '2': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            '3': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            
            '4': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            '5': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            
            
            '6': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            '7': [{'inicio': '08:00:00',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            
            '8': [{'inicio': '08:08:08',
                  'termino': "09:38:08",
              'propiedades': {'skills' : get_random_non_empty_subset(series),
   'configuracion_atencion': random.sample(modos, 1)[0],
   'porcentaje_actividad'  : int(88)/100,
         'atributos_series': atributos_series,
                    
                }},
                {'inicio': '10:08:08',
                'termino': None,
            'propiedades': {'skills' : get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : int(88)/100,
                    'atributos_series':atributos_series,
                    
                }}
                ],
            
            '9': [{'inicio': '09:09:09',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  :  int(99)/100,
                    'atributos_series':atributos_series,

                }}],
            '10': [{'inicio': '08:10:10',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '11': [{'inicio': '08:11:11',
            'termino': '13:11:11',
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '12': [{'inicio': '08:12:12',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '13': [{'inicio': '08:13:13',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '14': [{'inicio': '08:14:14',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series), 
                'configuracion_atencion':random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '15': [{'inicio': '08:15:15',
            'termino': None,
            'propiedades': {'skills': get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                    'atributos_series':atributos_series,

                }}],
            '14': [{'inicio': '08:14:14',
            'termino': '11:14:14',
            'propiedades': {'skills':get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,
                }},
               {
            'inicio':  '18:08:08',
            'termino': '19:14:14',
            'propiedades': {'skills':get_random_non_empty_subset(series),
                'configuracion_atencion': random.sample(modos, 1)[0],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                'atributos_series':atributos_series,
                }}]
            }
    return planificacion, niveles_servicio_x_serie

from datetime import timedelta

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
        self.bloque_atenciones = bloque_atenciones[['FH_Emi','FH_AteIni','IdSerie','T_Ate','IdEsc']]       
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
        
class match_emisiones_reloj_historico():
    def __init__(self, bloque_atenciones) -> bool:        
        self.bloque_atenciones = bloque_atenciones[['FH_AteIni' ,'IdSerie', 'T_Ate', 'IdEsc']]       
    def match(self, tiempo_actual):        
        # Convert the given time string to a timedelta object
        h, m, s = map(int, tiempo_actual.split(':'))
        given_time = timedelta(hours=h, minutes=m, seconds=s)
        # Filter rows based on the given condition
        try:
            mask = self.bloque_atenciones['FH_AteIni'].apply(
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
class Escritoriosv07:    
    def __init__(self,
                 planificacion: dict,
                 #params_pausas:tuple = (1, 2, 4), 
                 factor_pausas: float = 1,
                 probabilidad_pausas:float= 0.55,
                 ):
        #self.duracion_pausas          = params_pausas 
        self.factor_pausas            = factor_pausas
        self.probabilidad_pausas      = probabilidad_pausas      
        self.planificacion            = planificacion
        self.escritorios              = {k: {
                                            'inicio' : None, 
                                            'termino' : None, 
                                            'estado':                     'disponible',  #
                                            'tiempo_actual_disponible':   0,  # 
                                            'skills':                     None, #v[0]['propiedades'].get('skills'),  #                                    
                                            'configuracion_atencion':     None, #v[0]['propiedades'].get('configuracion_atencion'),  # 
                                            'contador_tiempo_disponible': iter(count(start=0, step=1)),  # 
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':      None,
                                            'duracion_inactividad':      None,                                    
                                            'contador_inactividad':      None,                                    
                                            'duracion_pausas':            None, #(1, 5, 15),  # --- pausas ---
                                            'probabilidad_pausas':        None,          # --- pausas ---
                                            'numero_pausas':              None,        # --- pausas ---
                                            'prioridades':                None,                                                                        
                                            'pasos':                      None,               
                                            'conexion':                   False,
                                            'pasos_alternancia': None,                                       
                                            }
                                            for k, v in planificacion.items()}  #        
        
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.propiedades_tramos                 = []        
        
    def aplicar_planificacion(self, hora_actual, planificacion, tiempo_total):
        
        propiedades_tramo = dict()
        for idEsc, un_escritorio in planificacion.items():
            for un_tramo in un_escritorio:
                on_off = hora_actual >= un_tramo['inicio'] and (lambda: 
                         hora_actual <= un_tramo['termino'] if un_tramo['termino'] is not None else True)()
                propiedades_tramo = propiedades_tramo | {idEsc: {
                                                'inicio'  : un_tramo['inicio'], 
                                                'termino' : un_tramo['termino'], 
                                                'conexion': on_off, #se setean conexiones
                                                'skills':                 un_tramo['propiedades']['skills'],
                                                'configuracion_atencion': un_tramo['propiedades']['configuracion_atencion'],
                                                'porcentaje_actividad':   un_tramo['propiedades']['porcentaje_actividad'],
                                                'prioridades':            {dict_series['serie']: dict_series['prioridad'] for dict_series in 
                                                                           un_tramo['propiedades'].get('atributos_series')},
                                                'pasos':                  {dict_series['serie']: dict_series['pasos'] for dict_series in 
                                                                           un_tramo['propiedades'].get('atributos_series')},                                                
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':       un_tramo['propiedades'].get('porcentaje_actividad'),
                                            'duracion_inactividad':       int(
                                                                            (1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60)
                                                                            ) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                            'contador_inactividad':  # None,    
                                                                        iter(islice(
                                                                        count(start=0, step=1),
                                                                        int((1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60))
                                                                        )) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,
                                                                        
                                            'duracion_pausas': 
                                                tuple(np.array((lambda x: 
                                                        (int(x / 10), int(x)/2, int(1 * x)))(
                                                            ((1-un_tramo['propiedades'].get('porcentaje_actividad')
                                                            )*self.factor_pausas*tiempo_total))).astype(int)),
                                                
                                                # (lambda x: (int(x / 10), int(x)/2, int(1 * x)))(
                                                #                ((1-un_tramo['propiedades'].get('porcentaje_actividad')
                                                #                 )*tiempo_total)/ (tiempo_total/30)),
                                                            # tuple(
                                                            #     self.factor_pausas*(
                                                            #     (
                                                            #     1-un_tramo['propiedades'].get('porcentaje_actividad')
                                                            #     )*
                                                            #     tiempo_total)*np.array(self.duracion_pausas)),                                          
                                            
                                            
                                            'probabilidad_pausas': 0 if un_tramo['propiedades'].get('porcentaje_actividad')==1 else self.probabilidad_pausas,
                                            #(1, 5, 15),
                        #                     'contador_tiempo_disponible':  
                        # self.escritorios_ON[idEsc]['contador_tiempo_disponible'] if {**self.escritorios_ON, **self.escritorios_OFF}[idEsc]['conexion'] == on_off == True else iter(count(start=0, step=1)), 
                        
                                            'pasos_alternancia': pasos_alternancia_v03(prioridades = 
                                                                                                   {dict_series['serie']: dict_series['prioridad'] for dict_series in 
                                                                                                   un_tramo['propiedades'].get('atributos_series')},
                                pasos = 
                                            {dict_series['serie']: dict_series['pasos'] for dict_series in 
                                                un_tramo['propiedades'].get('atributos_series')})
                                                if un_tramo['propiedades']['configuracion_atencion'] == 'Alternancia' else None, 
                                            }}
                if on_off:
                    break   
                
                
        self.propiedades_tramos.append(propiedades_tramo)        
        actualizar_keys_tramo(self.escritorios_ON, propiedades_tramo)  #
        actualizar_keys_tramo(self.escritorios_OFF, propiedades_tramo)  #        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion({**self.escritorios_ON, **self.escritorios_OFF})
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)  #
        
    def iniciar_atencion(self, escritorio, cliente_seleccionado):

        minutos_atencion =  int(math.floor(cliente_seleccionado.T_Ate/60))#round((cliente_seleccionado.T_Ate - cliente_seleccionado.FH_AteIni).total_seconds()/60)
            
        self.escritorios_ON[escritorio]['contador_tiempo_atencion'] = iter(islice(count(start=0, step=1), minutos_atencion))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']           = 'atención'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_atencion']  = minutos_atencion#tiempo de atención     
        self.escritorios[escritorio]['numero_de_atenciones'] += 1 #se guarda en self.escritorios para que no se resetee.
        self.escritorios_ON[escritorio]['numero_de_atenciones'] = self.escritorios[escritorio]['numero_de_atenciones'] 
        
        logging.info(f"el escritorio {escritorio} inició atención x {minutos_atencion} minutos ({cliente_seleccionado.T_Ate} segundos)")
        


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
            logging.info(f"No hay escritorio {state}")
            return False
    def iniciar_pausa(self, escritorio, tipo_inactividad:str = "Pausas", generador_pausa = generate_integer):
        # sourcery skip: extract-method, move-assign
      
        if tipo_inactividad == "Porcentaje":            
            self.escritorios_ON[escritorio]['estado'] = 'pausa'            
        else:
            min_val, avg_val, max_val = self.escritorios_ON[escritorio]['duracion_pausas']
            probabilidad_pausas       = self.escritorios_ON[escritorio]['probabilidad_pausas']
            minutos_pausa             = generador_pausa(min_val, avg_val, max_val, probabilidad_pausas)

            self.escritorios_ON[escritorio]['contador_tiempo_pausa'] = iter(islice(count(start=0, step=1), minutos_pausa)) #if minutos_pausa > 0 else None
            self.escritorios_ON[escritorio]['estado']                = 'pausa'#estado
            self.escritorios_ON[escritorio]['minutos_pausa']         = minutos_pausa#tiempo 
            
            
        logging.info(f"el escritorio {escritorio} inició pausa x {minutos_pausa} minutos")

            
    def iniciar_tiempo_disponible(self,escritorio):
        self.escritorios_ON[escritorio]['contador_tiempo_disponible'] = iter(count(start=0, step=1))
        self.escritorios_ON[escritorio]['estado']                     = 'disponible'#     

        logging.info(f"**el escritorio {escritorio} quedó **disponible**")

        
    def iterar_escritorios_bloqueados(self, escritorios_bloqueados: List[str], tipo_inactividad:str = "Pausas"):

        for escri_bloq in escritorios_bloqueados:
            #ver si está en atención:
            
            
            if self.escritorios_ON[escri_bloq]['estado'] == 'pausa': 
                        tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
                        if tiempo_pausa is None: 
                            #si termina tiempo en pausa pasa a estado disponible
                            self.iniciar_tiempo_disponible(escri_bloq)
                            
                        else:
                            pass
                            logging.info(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_pausa'] - tiempo_pausa} min de pausa")      
                                     
            if self.escritorios_ON[escri_bloq]['estado'] == 'atención':                
                #avanzamos en un minuto el tiempo de atención
                tiempo_atencion = next(self.escritorios_ON[escri_bloq]['contador_tiempo_atencion'], None)
                 
                #si terminó la atención
                if tiempo_atencion is None: 
                    #iniciar pausa 
                    self.iniciar_pausa(escri_bloq, tipo_inactividad)
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
                            
                        else:
                            
                            logging.info(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_pausa'] - tiempo_pausa} min de pausa")
                            pass    
                                        
                    
                else:
                    tiempo_atencion += 1
                    
                    logging.info(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_atencion'] - tiempo_atencion} min de atención") 
                    pass
                        
    def iterar_escritorios_disponibles(self, escritorios_disponibles: List[str]):
        
        for escri_dispon in escritorios_disponibles:               
            #avanzamos en un minuto el tiempo que lleva disponible.
            tiempo_disponible = next(self.escritorios_ON[escri_dispon]['contador_tiempo_disponible'])
            self.escritorios_ON[escri_dispon]['tiempo_actual_disponible'] = tiempo_disponible +1
        


# hora_cierre           = "16:30:00"
import logging


def simv7(un_dia, hora_cierre, planificacion, log_path: str = "dev/simulacion.log", probabilidad_pausas:float=0.55, factor_pausas:float=1):
    un_dia["FH_AteIni"] = None
    un_dia["FH_AteFin"] = None
    un_dia["IdEsc"] = None

    reloj = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
    registros_atenciones = pd.DataFrame()
    matcher_emision_reloj = match_emisiones_reloj(un_dia)
    supervisor = Escritoriosv07(planificacion=planificacion, probabilidad_pausas=probabilidad_pausas, factor_pausas=factor_pausas)
    registros_atenciones = pd.DataFrame()
    fila = pd.DataFrame()

    tiempo_total = (
        datetime.strptime(hora_cierre, "%H:%M:%S")
        - datetime.strptime(str(un_dia.FH_Emi.min().time()), "%H:%M:%S")
    ).total_seconds() / 60

    logging.basicConfig(filename=log_path, level=logging.INFO, filemode="w")

    for i, hora_actual in enumerate(reloj):
        total_mins_sim = i
        logging.info(
            f"--------------------------------NUEVA hora_actual {hora_actual}---------------------"
        )
        supervisor.aplicar_planificacion(
            hora_actual=hora_actual,
            planificacion=planificacion,
            tiempo_total=tiempo_total,
        )
        matcher_emision_reloj.match(hora_actual)

        if not matcher_emision_reloj.match_emisiones.empty:
            logging.info(f"nuevas emisiones")

            emisiones = matcher_emision_reloj.match_emisiones

            logging.info(
                f"hora_actual: {hora_actual} - series en emisiones: {list(emisiones['IdSerie'])}"
            )
            fila = pd.concat([fila, emisiones])
        else:
            logging.info(f"no hay nuevas emisiones hora_actual {hora_actual}")

        if supervisor.filtrar_x_estado("atención") or supervisor.filtrar_x_estado(
            "pausa"
        ):
            en_atencion = supervisor.filtrar_x_estado("atención") or []
            en_pausa = supervisor.filtrar_x_estado("pausa") or []
            escritorios_bloqueados = set(en_atencion + en_pausa)
            escritorios_bloqueados_conectados = [
                k
                for k, v in supervisor.escritorios_ON.items()
                if k in escritorios_bloqueados
            ]
            logging.info(
                f"iterar_escritorios_bloqueados: {escritorios_bloqueados_conectados}"
            )
            supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

        if supervisor.filtrar_x_estado("disponible"):
            conectados_disponibles = [
                k
                for k, v in supervisor.escritorios_ON.items()
                if k in supervisor.filtrar_x_estado("disponible")
            ]
            logging.info(f"iterar_escritorios_disponibles: {conectados_disponibles}")
            logging.info(
                "tiempo_actual_disponible",
                {
                    k: v["tiempo_actual_disponible"]
                    for k, v in supervisor.escritorios_ON.items()
                    if k in conectados_disponibles
                },
            )
            supervisor.iterar_escritorios_disponibles(conectados_disponibles)

            conectados_disponibles = balancear_carga_escritorios(
                {
                    k: {
                        "numero_de_atenciones": v["numero_de_atenciones"],
                        "tiempo_actual_disponible": v["tiempo_actual_disponible"],
                    }
                    for k, v in supervisor.escritorios_ON.items()
                    if k in conectados_disponibles
                }
            )

            for un_escritorio in conectados_disponibles:
                logging.info(f"iterando en escritorio {un_escritorio}")

                configuracion_atencion = supervisor.escritorios_ON[un_escritorio][
                    "configuracion_atencion"
                ]
                fila_filtrada = fila[
                    fila["IdSerie"].isin(
                        supervisor.escritorios_ON[un_escritorio].get("skills", [])
                    )
                ]  # filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])

                if fila_filtrada.empty:
                    continue
                elif configuracion_atencion == "FIFO":
                    un_cliente = FIFO(fila_filtrada)
                    logging.info(f"Cliente seleccionado x FIFO {tuple(un_cliente)}")
                elif configuracion_atencion == "Rebalse":
                    un_cliente = extract_highest_priority_and_earliest_time_row(
                        fila_filtrada,
                        supervisor.escritorios_ON[un_escritorio].get("prioridades"),
                    )
                    logging.info(f"Cliente seleccionado x Rebalse {tuple(un_cliente)}")
                elif configuracion_atencion == "Alternancia":
                    un_cliente = supervisor.escritorios_ON[un_escritorio][
                        "pasos_alternancia"
                    ].buscar_cliente(fila_filtrada)
                    logging.info(
                        f"Cliente seleccionado x Alternancia {tuple(un_cliente)}"
                    )

                fila = remove_selected_row(fila, un_cliente)
                logging.info(f"INICIANDO ATENCION de {tuple(un_cliente)}")
                supervisor.iniciar_atencion(un_escritorio, un_cliente)
                logging.info(
                    f"numero_de_atenciones de escritorio {un_escritorio}: {supervisor.escritorios_ON[un_escritorio]['numero_de_atenciones']}"
                )
                logging.info(
                    f"---escritorios disponibles: { supervisor.filtrar_x_estado('disponible')}"
                )
                logging.info(
                    f"---escritorios en atención: { supervisor.filtrar_x_estado('atención')}"
                )
                logging.info(
                    f"---escritorios en pausa: { supervisor.filtrar_x_estado('pausa')}"
                )

                un_cliente.IdEsc = int(un_escritorio)
                un_cliente.FH_AteIni = hora_actual
                registros_atenciones = pd.concat(
                    [registros_atenciones, pd.DataFrame(un_cliente).T]
                )

        if i == 0:
            fila["espera"] = 0
        else:
            fila["espera"] += 1 * 60
    logging.info(f"minutos simulados {total_mins_sim} minutos reales {tiempo_total}")
    return registros_atenciones, fila


def plan_desde_skills(skills, inicio, porcentaje_actividad=None):
    return {
        id: [
            {
                "inicio": inicio,
                "termino": None,
                "propiedades": {
                    "skills": sks,
                    "configuracion_atencion": random.choice(
                        ["FIFO", "Rebalse", "Alternancia"]
                    ),  # "Rebalse", # "Alternancia", #"Rebalse", #random.choice(["FIFO", "Rebalse", "Alternancia"]) "FIFO",
                    "porcentaje_actividad": random.uniform(0.80, 0.95) if porcentaje_actividad is None else porcentaje_actividad,
                    "atributos_series": atributos_x_serie(
                        ids_series=sorted(
                            list(
                                {val for sublist in skills.values() for val in sublist}
                            )
                        ),
                        sla_porcen_user=None,
                        sla_corte_user=None,
                        pasos_user=None,
                        prioridades_user=None,
                    ),
                },
            }
        ]
        for id, sks in skills.items()
    }
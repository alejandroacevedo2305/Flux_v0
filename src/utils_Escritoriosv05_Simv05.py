#%%
import itertools
from copy import deepcopy
import pandas as pd
import random
from datetime import datetime
def actualizar_keys_tramo(original_dict, updates):    
    for key, value in updates.items():  # Loop through the keys and values in the updates dictionary.
        if key in original_dict:  # Check if the key from updates exists in the original dictionary.
            original_dict[key]['conexion']               = value['conexion']
            original_dict[key]['skills']                 = value['skills']
            original_dict[key]['configuracion_atencion'] = value['configuracion_atencion']
            #original_dict[key]['atributos_series']       = value['atributos_series']            
            original_dict[key]['prioridades']           = value['prioridades']
            original_dict[key]['pasos']                 = value['pasos']
            original_dict[key]['porcentaje_actividad']   = value['porcentaje_actividad']
            original_dict[key]['inicio']                 = value['inicio']
            original_dict[key]['termino']                = value['termino']
            original_dict[key]['contador_tiempo_disponible']                = value['contador_tiempo_disponible']
            original_dict[key]['numero_de_atenciones']                = value['numero_de_atenciones']
            original_dict[key]['porcentaje_actividad']                = value['porcentaje_actividad']
            original_dict[key]['duracion_inactividad']                = value['duracion_inactividad']
            original_dict[key]['contador_inactividad']                = value['contador_inactividad']





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

def generar_planificacion(un_dia):
    skills   = obtener_skills(un_dia)
    series   = sorted(list({val for sublist in skills.values() for val in sublist}))
    modos    = ['FIFO','Alternancia', 'Rebalse']

    atributos_series = atributos_x_serie(ids_series=series, 
                                        sla_porcen_user=None, 
                                        sla_corte_user=None, 
                                        pasos_user=None, 
                                        prioridades_user=None)
    niveles_servicio_x_serie = {atr_dict['serie']:
                            (atr_dict['sla_porcen']/100, atr_dict['sla_corte']/60) 
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
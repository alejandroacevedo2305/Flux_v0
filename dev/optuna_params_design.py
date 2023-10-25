#%%
import optuna
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
from src.viz_utils import *
from src.optuna_utils import *
import optuna
import itertools
import pandas as pd
from scipy.stats.mstats import gmean
########################################################################
#-------------Cargar parámetros desde datos históricos------------------
########################################################################

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}

########################################################################
#--------Reconstruir planificación desde trials guardados en sqlite------------------
########################################################################

recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_objs.db") # TODO: MSSQL server someday
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "tramo_" in s.study_name]

# Seccion que deberia extraer un oprimo. 
# Guarda todos los estudios en un diccionario, con todos los trials, 
scores_studios = {} # >> { 'tramo_0' : {0 : 0.09123} } 
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: calcular_optimo_max_min(trial.values)
            # trial.values es una lista de SLA, tiempos de espera, etc. 
            # calcular_optimo_max_min hace la division entre SLA / largo_fila_espera
            # Cada key es el numero del trial. Kinda { 1: 0.089... } 
                for
                    trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
                    }    
 
trials_optimos          = extract_max_value_keys(scores_studios) # Para cada tramo, extrae el maximo, 
# Que es una tupla de forma {'tramo_0' : (514, 0.459)} (osea el tramo : { n_trial, valor_trial })

planificaciones_optimas = {}   
for k,v in trials_optimos.items():
    un_estudio               = optuna.multi_objective.load_study(study_name=k, storage=recomendaciones_db)
    trials_de_un_estudio     = un_estudio.get_trials(deepcopy=False)
    planificaciones_optimas  = planificaciones_optimas | {f"{k}":
        trial.user_attrs.get('planificacion')#calcular_optimo(trial.values)
                for
                    trial in trials_de_un_estudio if trial.number == v[0]
                    # trial.number == v[0] es el numero del trial, que se usa para hacer las planificaciones optimas
                    }   

# Unifica las planificaciones en una unica planificacion
# queda en el formato de planificacion, que deberia ser validada 
planificacion                =  plan_unico([plan for tramo,plan in planificaciones_optimas.items()]) 

# {'0': [{'inicio': '08:40:11',
#    'termino': '10:07:40',
#    'propiedades': {'skills': [5, 10, 11, 12, 14],
#     'configuracion_atencion': 'Rebalse'}}],
#  '1': [{'inicio': '08:40:11',
#    'termino': '10:07:40',
#    'propiedades': {'skills': [5, 10, 17],
#     'configuracion_atencion': 'Alternancia'}}],
#  '2': [{'inicio': '08:40:11',
#    'termino': '10:07:40',
#    'propiedades': {'skills': [5, 10, 11],
#     'configuracion_atencion': 'Alternancia'}}],
#  '3': [{'inicio': '08:40:11',
#    'termino': '10:07:40',
# # ... # VALIDAR ESTE OBJETO! 

prioridades                  =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) # Reemplazado por un input/historico desde la DB
registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) # 

# registros_atenciones -> tiene el mismo largo de las atenciones
# l_fila -> Cantidad de no atendidos al final del dia 

mins_de_corte_SLA               = [20, 20, 20, 20, 20, 20] # TODO: obtener de arriba
pocentajes_SLA                  = [40, 50, 60, 70, 80, 90] # Deberian salir del diccionario de SLAs de arriba 

registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
# Una lista de dataframes, donde cada dataframe contiene solo una serie -- why?

df_pairs                        = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) 
                                    for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]
# Recorre el registro por serie, lo zipea con los minimos de corte, la serie, y se los pasa a SLA por serie..
# que devuelve los niveles de servicio por hora
# df_pairs es una tupla con la ... No es mejor solo un dataframe, con indices? O con df[IdSerie] = int(serie) # Y usar proyeccion

# Este tiene el porcentaje de cada serie, dentro de tramos
porcentajes_reales  = {f"serie: {serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 

# This, SLA zipeado con los deseados, y luego la diferencia al cuadrado
# TODO: implementar como umbral para si SLA obtenido > SLA real, then 0
# v : value porcentaje real, p : porcentaje deseado
dif_cuadratica      = {k:(v-p)**2 for ((k,v),p) in zip(porcentajes_reales.items(),pocentajes_SLA)}

tuple(dif_cuadratica.values()) # Output del objetivo de Optuna -- Minimizar todas
# >>> 
# (1776.0204081632662,
#  78.75602031446186,
#  1290.129112869638,
#  73.469387755102,
#  108.88468809073731,
#  100.0)

#%%
# plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA histórico", color_sla="darkgoldenrod",n_rows=3,n_cols=2, heigth=10, width=10)
#%%
def objective(trial, 
    un_dia : pd.DataFrame,  # IdOficina  IdSerie  IdEsc, FH_Emi, FH_Llama  -- Deberia llamarse 'un_tramo'
    skills,  # {'escritorio_7': [10, 12], 'escritorio_10': [17, 14], 'escritorio_12': [17, 14], 'escritorio_11': <...> io_5': [10, 5], 'escritorio_8': [10, 12], 'escritorio_1': [10, 11, 12], 'escritorio_3': [10, 11]}
    subsets, # [(5,), (10,), (11,), (12,), (14,), (17,), (5, 10), (5, 11), (5, 12), (5, 14), (5, 17), (10, 11),  <...> 14, 17), (5, 10, 12, 14, 17), (5, 11, 12, 14, 17), (10, 11, 12, 14, 17), (5, 10, 11, 12, 14, 17)]
    niveles_servicio_x_serie,  # {5: (0.34, 35), 10: (0.34, 35), 11: (0.7, 45), 12: (0.34, 35), 14: (0.34, 35), 17: (0.6, 30)}
    modos_atenciones : list = ["Alternancia", "FIFO", "Rebalse"]
    ):
    
    try:
        
        # Un vector parametrico del largo de los escritorios. Entonces es un [True, False, True, True, etc.]
        # Queremos minimizar la suma de esta cosa 
        # NOTE: deberia ser 'escritorio_{i}_activo' 
        bool_vector  = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]

        # Misma cosa, sugiere el modo de atencion de cada escritorio, segun `modos_atenciones` arriba
        str_dict     = {i: trial.suggest_categorical(f'{i}',         modos_atenciones) for i in range(len(skills.keys()))} 
        
        # Aqui empieza el horror
        subset_idx = {i: trial.suggest_int(f'ids_{i}', 0, len(subsets) - 1) for i in range(len(skills.keys()))}
        # Skills puede ser la combinatoria no ordenada de todas las series, como Series 1, 2, 3 --> Subsets = (1,2,3), (1,2), (1,3), (2,3), (1), (2), (3), expungando (VACIO)
        # ya esta sanitizado para (1,3) == (3,1) (Combinatoria no-ordenada)
        # Optuna sugiere el indice del mejor subset -- de ~300 subsets posibles
        # esto hace ver bonitos a los Western Blot y micro-cortes de ... asfkln... dies. 
        # (en este momento "perro_demonio_1" se puso a gritar, y concuerdo con ella)
        # oh my sanity
        
        prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        planificacion            =  {} # Arma una planificacion con espacios parametricos. 
        inicio                   =  str(un_dia.FH_Emi.min().time())#'08:33:00'
        termino                  =  str(un_dia.FH_Emi.max().time())#'14:33:00'
        
        # Puede ser mas razonable usando un modelo_datos.simulador.planificacion 
        # TODO: convertir en una funcion y sacar de aqui. 
        # Loop through the keys
        for key in str_dict.keys():
            # Apply boolean mask
            if bool_vector[key]:
                # Create the inner dictionary
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills':list(subsets[subset_idx[key]]), # Set -> Lista, para el subset 'subset_idx', para el escritorio 'key'
                        #'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key], # FI FAI FO FU
                        #'zz': 0,
                    }
                }
                # Convert integer key to string and add the inner dictionary to a list, then add it to the output dictionary
                planificacion[str(key)] = [inner_dict] # NOTE: Es una lista why -- Config por trial por tramo del escritorio 

        #print(f"---------------------------{planificacion}")
        trial.set_user_attr('planificacion', planificacion) # This' actually cool 
        registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) 
                                    #   FUNCION SIMULADOR A EMPACAR, IDEALMENTE PARA VENDER AYER --- 
                                    #   no saca tiempos de SLA y otras cosas triviales -- Son triviales de calcular
                                    #   modulo Simulador.atenciones_a_SLAs() 

        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
        
        
        mins_de_corte_SLA               = [20, 20, 20, 20, 20, 20]
        pocentajes_SLA                  = [40, 50, 60, 70, 80, 90]
        
        df_pairs                        = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) 
                                    for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]
        #plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA histórico", color_sla="darkgoldenrod",n_rows=3,n_cols=2, heigth=10, width=10)
        # SLAs_x_Serie = {f"serie: {serie}": calculate_geometric_mean(esperas.espera, weights=len(esperas.espera)*[1]) 
        #                 for ((demandas, esperas), serie) in df_pairs} 
        # print('l_fila',l_fila)
        # print(SLAs_x_Serie)
        # obj1 = np.mean(list(SLAs_x_Serie.values()))
        # obj2 = (l_fila**2)/100
        # print(f"obj1: {obj1} - obj2: {obj2}")
        porcentajes_reales              = {f"serie: {serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 
        dif_cuadratica                  = {k:(v-p)**2 for ((k,v),p) in zip(porcentajes_reales.items(),pocentajes_SLA)}

        print(f"dif_cuadratica {dif_cuadratica}")
        return  tuple(dif_cuadratica.values()) #obj1, obj2
        

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

intervals  = get_time_intervals(un_dia, 4, 100) # Una funcion que recibe un dia, un intervalo, y un porcentaje de actividad para todos los intervalos
partitions = partition_dataframe_by_time_intervals(un_dia, intervals) # TODO: implementar como un static del simulador? 
# partitions es una lista de dataframes, que deberia tomar algo como un n-horas para particionar


import logging
from tqdm import tqdm # DIE! -- barra de progreso no funcional

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Initialize tqdm progress bar
n_trials     = int(20)
progress_bar = tqdm(total=n_trials, desc=f"Optimizing tramo", dynamic_ncols=True)

# Callback function to update progress bar
def update_progress_bar(study, trial):
    progress_bar.update(1)

storage = optuna.storages.get_storage("sqlite:///alejandro_objs_v2.db")

# Loop to optimize each partition
# Tramo == Estudio == Particion #> True
for idx, part in enumerate(partitions):

    # Update progress bar description for each partition
    progress_bar.set_description(f"Optimizing tramo_{idx}")

    # Reset the progress bar
    progress_bar.n = 0
    progress_bar.last_print_n = 0
    progress_bar.refresh()

    # Create a multi-objective study object and specify the storage
    study_name = f"tramo_{idx}"
    study = optuna.multi_objective.create_study(directions=len(series)*['minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)

    # TODO: sacar fuera
    subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))

    # Optimize with a timeout (in seconds)
    study.optimize(lambda trial: objective(trial,
                                           un_dia                   = part,
                                           skills                   = skills,
                                           subsets                  = subsets,
                                           niveles_servicio_x_serie = niveles_servicio_x_serie),
                   n_trials  = n_trials, #int(1e4),  # Make sure this is an integer
                   timeout   = 2*3600,    #  hours
                   callbacks = [update_progress_bar])  # Callback for progress bar

# Close the progress bar
progress_bar.close()

# TODO PARA APIFICAR:
# - Meter todo esto en una demo, mentalmente comprensible, en un Jupyter
# - Meter los modelos de datos de Pydantic, porque necesitamos que todo sea serializable
# - Cambiar el storage por Postgres? SQLITE funciona pero deberia ser optimo en algun momento

#%%

import optuna

recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_objs_v2.db") # Objetivos de 6-salidas
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "tramo_" in s.study_name]

scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: trial.values
                for
                    trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
                    } 
    
# score_studios >>> 'tramo' : (trial, (tupla de minimos cuadrados))
# {'tramo_0': {0: (695.0413223140498,
#    23.662716309404832,
#    19.753086419753135,
#    37.34567901234572,
#    685.2076124567476,
#    100.0),
#   1: (695.0413223140498,
#    34.51015937375372,
#    19.753086419753135,
#    296.60493827160514,
#    870.8285274894271,
#    544.4444444444449),
#   2: (695.0413223140498, 
#   ...

# GRANCIOSISIMA SELECCION DEL MEJOR TRIAL   

# vect_pesos = (1000, 500, 100, 10, 10, 10) # Pesos asignados por serie
# luego promedio ponderado con este vect_pesos -- harmonic mean puede ser

# alt, usar la distancia geometrica de esto, con ((vect_tuplas * vect_pesos)^2).sum().sqare_root()

# hierarchical = ( 1, 5, 3, 2, 4, 6 )
# for idx in hierarchical:
#     mejor_valor = max( tupla[idx], mejor_valor )
#     # la idea es quedarse con el que obtiene el maximo para la lista de forma ordenada. Luego optimizar los otros son meh
#     # en funcion, es casi como un 
#     #     df_tuplas.sort_values(by=[serie_4, serie_2, serie_3, serie_1, ...]).iloc[0]


#%% 
trials_optimos          = extract_max_value_keys(scores_studios)
planificaciones_optimas = {}   
for k,v in trials_optimos.items():
    un_estudio               = optuna.multi_objective.load_study(study_name=k, storage=recomendaciones_db)
    trials_de_un_estudio     = un_estudio.get_trials(deepcopy=False)
    planificaciones_optimas  = planificaciones_optimas | {f"{k}":
        trial.user_attrs.get('planificacion')#calcular_optimo(trial.values)
                for
                    trial in trials_de_un_estudio if trial.number == v[0]
                    }   

planificacion                =  plan_unico([plan for tramo,plan in planificaciones_optimas.items()])
prioridades                  =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 


registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)
#%%
registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
df_pairs = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                                                         [20, 20, 20, 20, 20, 20])]
                                                                                                         #[5, 25, 4, 30, 60, 70])]
plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA IA", color_sla="purple",n_rows=3,n_cols=2, heigth=10, width=12)
# SLAs_x_Serie = {f"serie: {serie}": calculate_geometric_mean(esperas.espera, weights=len(esperas.espera)*[1]) 
#                 for ((demandas, esperas), serie) in df_pairs} 




#%%
##################################################
# --------------PLOTLY-----------------------
###################################################3
import optuna
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
from src.optuna_utils import *

import optuna
import itertools
import pandas as pd
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
rows = len(un_dia)
chunk_size = rows // 4
partitions = [un_dia.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
# Create a SQLite storage to save all studies
#storage = optuna.storages.get_storage("sqlite:///multiple_studies.db")
tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]

import json
def format_dict_for_hovertext(dictionary):
    formatted_str = ""
    for key, value in dictionary.items():
        formatted_str += f"'{key}': {json.dumps(value)},<br>"
    return formatted_str[:-4]  # Remove the trailing HTML line break

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Connect to the SQLite database
storage = optuna.storages.get_storage("sqlite:///multiple_studies.db")
# Retrieve all study names
all_study_names = optuna.study.get_all_study_summaries(storage)
relevant_study_names = [s.study_name for s in all_study_names if "workforce_" in s.study_name]
# Infer the number of partitions
num_partitions = len(relevant_study_names)
# Initialize the plot
fig = make_subplots(rows=1, cols=num_partitions, shared_yaxes=True, subplot_titles=tramos)
# Loop to load each partition study and extract all trials
for idx, tramo in enumerate(tramos):  # Assuming `tramos` is defined elsewhere
    study_name = f"workforce_{idx}"
    # Load the study
    study = optuna.multi_objective.load_study(study_name=study_name, storage=storage)
    # Extract all trials
    all_trials = study.get_trials(deepcopy=False)
    # Initialize lists for the current partition
    current_partition_values_0 = []
    current_partition_values_1 = []
    current_partition_values_2 = []
    hover_texts = []
    for trial in all_trials:
        # Skip if the trial is not complete
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Append the values to the lists for the current partition
        current_partition_values_0.append(trial.values[0])
        current_partition_values_1.append(trial.values[1])
        current_partition_values_2.append(trial.values[2])
        hover_texts.append(f"{format_dict_for_hovertext(trial.user_attrs.get('planificacion'))}")  # Creating hover text from trial parameters
    # Plotting
    fig.add_trace(
        go.Scatter(
            x=current_partition_values_1,
            y=current_partition_values_0,
            mode='markers',
            marker=dict(opacity=0.5,
                        size=current_partition_values_2,
                        line=dict(
                width=2,  # width of border line
                color='black'  # color of border line
            )),
            hovertext=hover_texts,
            name=f"{tramo}"
        ),
        row=1,
        col=idx + 1  # Plotly subplots are 1-indexed
    )
    # Customizing axis
    fig.update_xaxes(title_text="n Escritorios")#, tickvals=list(range(min(current_partition_values_1), max(current_partition_values_1)+1)), col=idx+1)
    fig.update_yaxes(title_text="SLA global", row=1, col=idx+1)
# Show plot
fig.update_layout(title="Subplots with Hover Information")
fig.show()
#%% -------------------------------------------

def regenerate_global_keys(lst_of_dicts):

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
estudios = {}
for idx, tramo in enumerate(tramos):  # Assuming `tramos` is defined elsewhere
    study_name = f"workforce_{idx}"
    # Load the study
    study = optuna.multi_objective.load_study(study_name=study_name, storage=storage)
    # Extract all trials
    all_trials = study.get_trials(deepcopy=False)    
    estudios = estudios | {f"{study_name}":
        { i: ( trial.values[0]/(trial.values[1]+trial.values[2]), trial.user_attrs.get('planificacion')) for i, trial in enumerate(all_trials) if trial.state == optuna.trial.TrialState.COMPLETE}
        }
mejores_configs = []

for k,v in estudios.items():

   tuple_list = [plan for trial, plan in v.items() ]
   selected_tuple = max(tuple_list, key=lambda x: x[0])
   
   mejores_configs.append(selected_tuple[1])



# %%
""" 
"""
planificacion = regenerate_global_keys(mejores_configs)#mejores_configs[0]
prioridades   =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
registros_SLA = simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)

#%%

registros_sla               = pd.DataFrame()
tabla_atenciones         = un_dia[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
SLA_index                = 0
SLA_df                   = pd.DataFrame()
for cliente_seleccionado in tabla_atenciones.iterrows():
    

    un_cliente   = pd.DataFrame(cliente_seleccionado[1][['FH_Emi', 'IdSerie', 'espera']]).T
    registros_sla   =  pd.concat([registros_sla, un_cliente])#.reset_index(drop=True)
    SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(registros_sla, niveles_servicio_x_serie).items()), columns=['keys', 'values'])

    SLA_index+=1                        
    SLA_una_emision['index']            = SLA_una_emision.shape[0]*[SLA_index]
    SLA_una_emision['hora'] = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#SLA_una_emision.time().strftime('%H:%M:%S')
    SLA_df                              = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)


trajectoria_sla = SLA_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)

trajectoria_sla.iloc[0:700]
#.drop_duplicates(subset=['emisión'])
# pd.DataFrame(list(nivel_atencion_x_serie(tabla_atenciones, niveles_servicio_x_serie).items()), columns=['keys', 'values'])
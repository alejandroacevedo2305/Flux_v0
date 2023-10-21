#%%
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


########################################################################
#-------------Cargar parámetros desde datos históricos------------------
########################################################################
# dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
# un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
# skills   = obtener_skills(un_dia)
# series   = sorted(list({val for sublist in skills.values() for val in sublist}))
# SLAs     = [(0.6, 30), (0.34, 35), (0.7, 45)]
# niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
########################################################################
#--------Reconstruir planificación desde trials guardados en sqlite------------------
########################################################################

def calcular_optimo(multi_obj):
    return multi_obj[0]/(multi_obj[1]+multi_obj[2]) # max_SLA/(min_Esc + min_Skills)
def extract_max_value_keys(input_dict):
    output_dict = {}  # Initialize an empty dictionary to store the result
    # Loop through each item in the input dictionary
    for workforce, values_dict in input_dict.items():
        max_key = max(values_dict, key=values_dict.get)  # Find the key with the maximum value in values_dict
        max_value = values_dict[max_key]  # Get the maximum value
        output_dict[workforce] = (max_key, max_value)  # Add the key and value to the output dictionary
    return output_dict  # Return the output dictionary

recomendaciones_db   = optuna.storages.get_storage("sqlite:///multiple_studies.db")
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "workforce_" in s.study_name]

scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: calcular_optimo(trial.values)
                for
                    trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
                    }    
 
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

[plan for tramo,plan in planificaciones_optimas.items()]
#%%
""" 
"""

#%%
def objective(trial, un_dia,skills, subsets, niveles_servicio_x_serie,  modos_atenciones:list = ["Alternancia", "FIFO", "Rebalse"]):
    try:
        bool_vector  = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]
        str_dict     = {i: trial.suggest_categorical(f'{i}',         modos_atenciones) for i in range(len(skills.keys()))} 
        subset_idx = {i: trial.suggest_int(f'ids_{i}', 0, len(subsets) - 1) for i in range(len(skills.keys()))}
        prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        planificacion            =  {}
        inicio                   =  str(un_dia.FH_Emi.min().time())#'08:33:00'
        termino                  =  str(un_dia.FH_Emi.max().time())#'14:33:00'
        # Loop through the keys
        for key in str_dict.keys():
            # Apply boolean mask
            if bool_vector[key]:
                # Create the inner dictionary
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills':list(subsets[subset_idx[key]]),
                        #'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key],
                        #'zz': 0,
                    }
                }
                # Convert integer key to string and add the inner dictionary to a list, then add it to the output dictionary
                planificacion[str(key)] = [inner_dict]
        #print(f"---------------------------{planificacion}")
        trial.set_user_attr('planificacion', planificacion)
        registros_atenciones = optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) 
        
        
        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        series                          = sorted(list({val for sublist in skills.values() for val in sublist}))
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]      
        

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

rows = len(un_dia)
chunk_size = rows // 4
partitions = [un_dia.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
# Create a SQLite storage to save all studies
storage = optuna.storages.get_storage("sqlite:///multiple_studies.db")
tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]

# Loop to optimize each partition
for idx, part in enumerate(partitions):

    # Create a multi-objective study object and specify the storage
    study_name = f"workforce_{idx}"  # Unique name for each partition
    study = optuna.multi_objective.create_study(directions=['maximize', 'minimize', 'minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)
    # Optimize the study, the objective function is passed in as the first argument
    subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))
    study.optimize(lambda trial: objective(trial, 
                                        un_dia                   = part,
                                        skills                   = skills,
                                        subsets                  = subsets,
                                        niveles_servicio_x_serie = niveles_servicio_x_serie,
                                        ), 
                                        n_trials                 = 50)
    

#%%
# --------------PLOTLY-----------------------
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
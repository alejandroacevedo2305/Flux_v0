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

def calcular_optimo_max_min(multi_obj):
    return multi_obj[0]/(multi_obj[1]) # max_SLA/(min_n_espera)
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
def calculate_geometric_mean(series, weights=None):
    series = series.dropna()
    return np.nan if series.empty else gmean(series, weights=weights)
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


recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_objs.db")
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "tramo_" in s.study_name]

scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: calcular_optimo_max_min(trial.values)
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

planificacion                =  plan_unico([plan for tramo,plan in planificaciones_optimas.items()])
prioridades                  =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)

registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
df_pairs = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                          [20, 20, 20, 20, 20, 20])]
plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA histórico", color_sla="darkgoldenrod",n_rows=3,n_cols=2, heigth=10, width=10)
# SLAs_x_Serie = {f"serie: {serie}": calculate_geometric_mean(esperas.espera, weights=len(esperas.espera)*[1]) 
#                 for ((demandas, esperas), serie) in df_pairs} 

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
        registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) 
        
        
        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
        df_pairs = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                                [20, 20, 20, 20, 20, 20])]
        #plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA histórico", color_sla="darkgoldenrod",n_rows=3,n_cols=2, heigth=10, width=10)
        SLAs_x_Serie = {f"serie: {serie}": calculate_geometric_mean(esperas.espera, weights=len(esperas.espera)*[1]) 
                        for ((demandas, esperas), serie) in df_pairs} 
        print('l_fila',l_fila)
        print(SLAs_x_Serie)
        obj1 = np.mean(list(SLAs_x_Serie.values()))
        obj2 = (l_fila**2)/100
        print(f"obj1: {obj1} - obj2: {obj2}")
        
        
        return  obj1, obj2
        

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

# rows = len(un_dia)
# chunk_size = rows // 4
# partitions = [un_dia.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
# # Create a SQLite storage to save all studies
#
# tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]
def get_time_intervals(df, n, percentage:float=100):
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
    adjusted_intervals = [(start_time + 0.01 * (100 - percentage) * (end_time - start_time), end_time) for start_time, end_time in intervals]
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

intervals = get_time_intervals(un_dia, 4, 100)


partitions = partition_dataframe_by_time_intervals(un_dia, intervals)
import optuna
import logging
from tqdm import tqdm

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Initialize tqdm progress bar
progress_bar = tqdm(total=int(1e6), desc=f"Optimizing tramo", dynamic_ncols=True)

# Callback function to update progress bar
def update_progress_bar(study, trial):
    progress_bar.update(1)
#%%
# Get Optuna storage
storage = optuna.storages.get_storage("sqlite:///alejandro_objs.db")

# Loop to optimize each partition
for idx, part in enumerate(partitions):

    # Update progress bar description for each partition
    progress_bar.set_description(f"Optimizing tramo_{idx}")

    # Reset the progress bar
    progress_bar.n = 0
    progress_bar.last_print_n = 0
    progress_bar.refresh()

    # Create a multi-objective study object and specify the storage
    study_name = f"tramo_{idx}"
    study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)

    subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))

    # Optimize with a timeout (in seconds)
    study.optimize(lambda trial: objective(trial,
                                           un_dia                   = part,
                                           skills                   = skills,
                                           subsets                  = subsets,
                                           niveles_servicio_x_serie = niveles_servicio_x_serie),
                   n_trials  = int(1e4),  # Make sure this is an integer
                   timeout   = 2*3600,    # 8 hours
                   callbacks = [update_progress_bar])  # Callback for progress bar

# Close the progress bar
progress_bar.close()


#%%
# storage = optuna.storages.get_storage("sqlite:///alejandro_objs.db")
# # Loop to optimize each partition
# for idx, part in enumerate(partitions):

#     # Create a multi-objective study object and specify the storage
#     study_name = f"tramo_{idx}"  # Unique name for each partition
#     study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'],
#                                                 study_name=study_name,
#                                                 storage=storage, load_if_exists=True)
#     # Optimize the study, the objective function is passed in as the first argument
#     subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))
#     study.optimize(lambda trial: objective(trial, 
#                                         un_dia                   = part,
#                                         skills                   = skills,
#                                         subsets                  = subsets,
#                                         niveles_servicio_x_serie = niveles_servicio_x_serie,
#                                         ), 
#                                         n_trials                 = 1e6,
#                                           timeout   = 8*3600
#                                           )  # 3600 seconds = 1 hour
 
    
#%%

import optuna

recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_objs.db")
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "tramo_" in s.study_name]

scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: calcular_optimo_max_min(trial.values)
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
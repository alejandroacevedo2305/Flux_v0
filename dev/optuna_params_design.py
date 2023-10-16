#%%
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd

def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.84, 10), (0.34, 25), (0.8, 33)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
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
        print(f"---------------------------{planificacion}")
        trial.set_user_attr('planificacion', planificacion)
        registros_SLA = simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)       
        return gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna()), sum(bool_vector)

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
    study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'],
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
                                        n_trials                 = 10)
    

#%%
# --------------PLOTLY-----------------------
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
    hover_texts = []

    for trial in all_trials:
        # Skip if the trial is not complete
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Append the values to the lists for the current partition
        current_partition_values_0.append(trial.values[0])
        current_partition_values_1.append(trial.values[1])
        hover_texts.append(f"{format_dict_for_hovertext(trial.user_attrs.get('planificacion'))}")  # Creating hover text from trial parameters

    # Plotting
    fig.add_trace(
        go.Scatter(
            x=current_partition_values_1,
            y=current_partition_values_0,
            mode='markers',
            marker=dict(opacity=0.7),
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
#%% ---------------------matplotlib----------------------

import optuna
import matplotlib.pyplot as plt

# Connect to the SQLite database
storage = optuna.storages.get_storage("sqlite:///multiple_studies.db")

# Retrieve all study names
all_study_names = optuna.study.get_all_study_summaries(storage)
relevant_study_names = [s.study_name for s in all_study_names if "partition_" in s.study_name]

# Infer the number of partitions
num_partitions = len(relevant_study_names)

# Initialize the plot
fig, axs = plt.subplots(1, num_partitions, figsize=(12, 3), sharey=True, sharex=True)
if num_partitions == 1:
    axs = [axs]  # Make sure axs is a list even if there's only one subplot

# Loop to load each partition study and extract all trials
for idx, ax, tramo in zip(range(num_partitions), axs, tramos):
    study_name = f"partition_{idx}"
    
    # Load the study
    study = optuna.multi_objective.load_study(study_name=study_name, storage=storage)
    
    # Extract all trials
    all_trials = study.get_trials(deepcopy=False)
    
    # Initialize lists for the current partition
    current_partition_values_0 = []
    current_partition_values_1 = []

    for trial in all_trials:
        # Skip if the trial is not complete
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Append the values to the lists for the current partition
        current_partition_values_0.append(trial.values[0])
        current_partition_values_1.append(trial.values[1])

    # Plotting
    ax.scatter(current_partition_values_1,current_partition_values_0, alpha= 0.5)
    ax.set_title(f"{tramo}")
    ax.set_xlabel('n Escritorios')
    ax.set_ylabel( 'SLA global')

# Show plot
plt.tight_layout()
plt.show()










# %%

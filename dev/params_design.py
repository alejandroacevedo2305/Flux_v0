#%%
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *

def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))


dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.84, 10), (0.34, 25), (0.8, 33)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}


def objective(trial, un_dia,skills, niveles_servicio_x_serie,  modos_atenciones:list = ["Alternancia", "FIFO", "Rebalse"]):
    try:
        bool_vector = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]  
        #modos_atenciones = ["Alternancia", "FIFO", "Rebalse"]
        series = sorted(list({val for sublist in skills.values() for val in sublist}))
        str_dict = {i: trial.suggest_categorical('modo atención', modos_atenciones) for i in range(len(skills.keys()))}
        subset_dict = {i : trial.suggest_categorical('series', non_empty_subsets(series)) for i in range(len(skills.keys()))}
        # Storing user attributes
        trial.set_user_attr('bool_vector', bool_vector)
        trial.set_user_attr('str_dict', str_dict)
        trial.set_user_attr('subset_dict', subset_dict)
        prioridades              = prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        p                        = {}
        inicio                   = str(un_dia.FH_Emi.min().time())#'08:33:00'
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
                        'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key]
                    }
                }
                # Convert integer key to string and add the inner dictionary to a list, then add it to the output dictionary
                p[str(key)] = [inner_dict]
        
        registros_SLA = simular(p, niveles_servicio_x_serie, un_dia, prioridades)       
        return gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna()), sum(bool_vector)

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

import optuna
import pandas as pd
from scipy.stats import gmean

# Assuming the dataframe un_dia and other variables are already loaded.
# un_dia, skills, niveles_servicio_x_serie are assumed to be pre-defined
# Partition the data into 4 parts
rows = len(un_dia)
chunk_size = rows // 4
partitions = [un_dia.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
# Create a SQLite storage to save all studies
storage = optuna.storages.get_storage("sqlite:///multiple_studies.db")


tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]



# Loop to optimize each partition
for idx, part in enumerate(partitions):

    # Create a multi-objective study object and specify the storage
    study_name = f"partition_{idx}"  # Unique name for each partition
    study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)

    # Optimize the study, the objective function is passed in as the first argument
    study.optimize(lambda trial: objective(trial, 
                                        un_dia                   = part,
                                        skills                   = skills,
                                        niveles_servicio_x_serie = niveles_servicio_x_serie,
                                        ), 
                                        n_trials                 = 20)

#%%
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
fig, axs = plt.subplots(1, num_partitions, figsize=(15, 2), sharey=True, sharex=True)
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








#%%
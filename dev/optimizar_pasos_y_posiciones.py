#%%
import warnings
warnings.filterwarnings('ignore')

import random
import pandas as pd
import optuna
from src.datos_utils import DatasetTTP, obtener_skills
from src.optuna_utils import non_empty_subsets
import itertools
import numpy as np
def get_permutations_array(length):
    # Generate the list based on the input length
    items = list(range(1, length + 1))
    
    # Get all permutations of the list
    all_permutations = list(itertools.permutations(items))
    return np.array(all_permutations)

def check_priorities(dicts):
    # This set will store all the priority values we encounter.
    seen_priorities = set()    
    # Iterate over each key in the dictionary to access each list of dictionaries.
    for key in dicts:
        # Each key points to a list, we'll iterate through it.
        for entry in dicts[key]:
            # We are only interested in 'propiedades' and within it, 'prioridades'.
            priorities = entry['propiedades']['prioridades']            
            # Now we'll go through each value in 'prioridades'.
            for value in priorities.values():
                # If a value is already in the set, it's a duplicate and we return False.
                if value in seen_priorities:
                    return False
                # Otherwise, we add the value to the set.
                seen_priorities.add(value)                
    # If we've gone through all the data and found no duplicates, return True.
    return True

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
#%%
def objective(trial, 
    un_dia : pd.DataFrame,  # IdOficina  IdSerie  IdEsc, FH_Emi, FH_Llama  -- Deberia llamarse 'un_tramo'
    subsets, # [(5,), (10,), (11,), (12,), (14,), (17,), (5, 10), (5, 11), (5, 12), (5, 14), (5, 17), (10, 11),  <...> 14, 17), (5, 10, 12, 14, 17), (5, 11, 12, 14, 17), (10, 11, 12, 14, 17), (5, 10, 11, 12, 14, 17)]
    modos_atenciones : list = ["Alternancia", "FIFO", "Rebalse"],
    minimo_escritorios: int = 2,
    maximo_escritorios: int = 5,
    ):    
    try:

        bool_vector              = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(maximo_escritorios)]
        #Restricción de minimo de escritorios
        assert sum(bool_vector) >= minimo_escritorios, f"No cumple con minimo_escritorios: {minimo_escritorios}."
        
        str_dict                 = {i: trial.suggest_categorical(f'{i}',         modos_atenciones) for i in range(maximo_escritorios)} 
        subset_idx               = {i: trial.suggest_int(f'ids_{i}', 0, len(subsets) - 1) for i in range(maximo_escritorios)}   
        #prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        planificacion            =  {} # Arma una planificacion con espacios parametricos. 
        inicio                   =  str(un_dia.FH_Emi.min().time())#'08:33:00'
        termino                  =  str(un_dia.FH_Emi.max().time())#'14:33:00'
        
        #skills_len = len(list(subsets[subset_idx[key]]))
        
        for key in str_dict.keys():
            if bool_vector[key]:
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills':list(subsets[subset_idx[key]]), # Set -> Lista, para el subset 'subset_idx', para el escritorio 'key'
                        'configuracion_atencion': str_dict[key], # 
                        'prioridades': {
                                        i: 100*trial.suggest_int(f'prioridades_{i}', 1, len(list(subsets[subset_idx[key]]))) for i in list(subsets[subset_idx[key]])
                                        },
                        'pasos': {i: trial.suggest_int(f'pasos_{i}', 1, 4) for i in list(subsets[subset_idx[key]])},
                    }
                }
                planificacion[str(key)] = [inner_dict] # NOTE: Es una lista why -- Config por trial por tramo del escritorio 
                
                
                
        assert check_priorities(planificacion), f"prioridad duplicada."
        
        trial.set_user_attr('planificacion', planificacion) # This' actually cool 

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()
    
    return random.randint(0,10), random.randint(0,10)


skills   = obtener_skills(un_dia)
subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))
IA      = optuna.multi_objective.create_study(directions= 2*['minimize'])
IA.optimize(lambda trial: objective(trial,
                                           un_dia                   = un_dia,
                                           subsets                  = subsets,
                                           minimo_escritorios       = 2,
                                           maximo_escritorios       = 5
                                           ),
                   n_trials  = 1000, #int(1e4),  # Make sure this is an integer
                   #timeout   = 2*3600,   #  hours
                   )  
IA.get_trials()[0].user_attrs.get('planificacion')
#%%
def check_priorities(dicts):
    # This set will store all the priority values we encounter.
    seen_priorities = set()    
    # Iterate over each key in the dictionary to access each list of dictionaries.
    for key in dicts:
        # Each key points to a list, we'll iterate through it.
        for entry in dicts[key]:
            # We are only interested in 'propiedades' and within it, 'prioridades'.
            priorities = entry['propiedades']['prioridades']            
            # Now we'll go through each value in 'prioridades'.
            for value in priorities.values():
                # If a value is already in the set, it's a duplicate and we return False.
                if value in seen_priorities:
                    return False
                # Otherwise, we add the value to the set.
                seen_priorities.add(value)                
    # If we've gone through all the data and found no duplicates, return True.
    return True

# Example usage:
dictionaries = {
    '1': [{'inicio': '08:40:11',
        'termino': '14:30:23',
        'propiedades': {'skills': [17],
        'configuracion_atencion': 'FIFO',
        'prioridades': {17: 100},
        'pasos': {17: 4}}}],
    '2': [{'inicio': '08:40:11',
        'termino': '14:30:23',
        'propiedades': {'skills': [14, 17],
        'configuracion_atencion': 'FIFO',
        'prioridades': {14: 2, 17: 1},
        'pasos': {14: 4, 17: 4}}}]}
    
result = check_priorities(dictionaries)
print(result)  # This should print False because there is a duplicate priority value of 1.

#%%
""" 
I have dictionaties like this:

{'1': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {17: 100},
    'pasos': {17: 4}}}],
 '2': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {14: 200, 17: 100},
    'pasos': {14: 4, 17: 4}}}],
 '3': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [5, 17],
    'configuracion_atencion': 'Rebalse',
    'prioridades': {5: 200, 17: 100},
    'pasos': {5: 2, 17: 4}}}],
 '4': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [5, 10, 11, 17],
    'configuracion_atencion': 'Rebalse',
    'prioridades': {5: 200, 10: 200, 11: 200, 17: 100},
    'pasos': {5: 2, 10: 4, 11: 3, 17: 4}}}]}

I need a function which returns false if at least one values in 'prioridades'
is duplicated in one of the the outet keys,
for example when
{'1': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {17: 100},
    'pasos': {17: 4}}}],
 '2': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {14: 200, 17: 1},
    'pasos': {14: 4, 17: 4}}}]}
the function should return True. 
While if 
{'1': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {17: 100},
    'pasos': {17: 4}}}],
 '2': [{'inicio': '08:40:11',
   'termino': '14:30:23',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO',
    'prioridades': {14: 1, 17: 1},
    'pasos': {14: 4, 17: 4}}}]}
should return false.
"""
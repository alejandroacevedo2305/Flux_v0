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

from src.optuna_utils import plan_unico
from dev.atributos_de_series import atributos_x_serie
import optuna
import numpy as np
import time


from dev.MisEscritorios_v04 import simv04

def get_permutations_array(length):
    # Generate the list based on the input length
    items = list(range(1, length + 1))
    
    # Get all permutations of the list
    all_permutations = list(itertools.permutations(items))
    return np.array(all_permutations)

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
atributos_series = atributos_x_serie(ids_series=series, 
                                    sla_porcen_user=None, 
                                    sla_corte_user=None, 
                                    pasos_user=None, 
                                    prioridades_user=None)
niveles_servicio_x_serie = {atr_dict['serie']:
                            (atr_dict['sla_porcen']/100, atr_dict['sla_corte']) 
                            for atr_dict in atributos_series}




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
                #skills                    = list(subsets[subset_idx[key]])
                #permutaciones_prioridades = get_permutations_array(list(subsets[subset_idx[key]]).__len__())
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills': list(subsets[subset_idx[key]]),# list(subsets[subset_idx[key]]), # Set -> Lista, para el subset 'subset_idx', para el escritorio 'key'
                        'configuracion_atencion': str_dict[key], # 
                        'prioridades': {k:v for k,v in zip(list(subsets[subset_idx[key]]),
                                                           list(
                                        get_permutations_array(
                                            list(subsets[subset_idx[key]]).__len__())[
                                       trial.suggest_int(
                                           f'prioridades', 0, get_permutations_array(list(subsets[subset_idx[key]]).__len__()).__len__()
                                           )
                                       ]))}
                                     ,
                        'pasos': {i: trial.suggest_int(f'pasos_{i}', 1, 4) for i in list(subsets[subset_idx[key]])},
                    }
                }
                planificacion[str(key)] = [inner_dict] # NOTE: Es una lista why -- Config por trial por tramo del escritorio 
                
                
                        
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
                                           minimo_escritorios       = 3,
                                           maximo_escritorios       = 15
                                           ),
                   n_trials  = 10, #int(1e4),  # Make sure this is an integer
                   #timeout   = 2*3600,   #  hours
                   )  
planificacion_optuna = [trial for trial in IA.trials if trial.state == optuna.trial.TrialState.COMPLETE][1].user_attrs.get('planificacion')
planificacion_optuna
#%%


start_time = time.time()




planificacion = {un_escritorio[0]:
                [
                    {
                    'inicio': un_escritorio[1][0]['inicio'],
                'termino':un_escritorio[1][0]['termino'],
                'propiedades':{
                    'skills': un_escritorio[1][0]['propiedades']['skills'],
                'configuracion_atencion': un_escritorio[1][0]['propiedades']['configuracion_atencion'],
                'porcentaje_actividad'  : np.random.randint(85, 90)/100,
                
                'atributos_series':
                    atributos_x_serie(ids_series=un_escritorio[1][0]['propiedades']['skills'], 
                                                sla_porcen_user=[niveles_servicio_x_serie[s][0] for s in un_escritorio[1][0]['propiedades']['skills']]
            , 
                                                sla_corte_user=[niveles_servicio_x_serie[s][1] for s in un_escritorio[1][0]['propiedades']['skills']]
            , 
                                                pasos_user=list(un_escritorio[1][0]['propiedades']['pasos'].values()), 
                                                prioridades_user=list(un_escritorio[1][0]['propiedades']['prioridades'].values())),
                },}]
                for un_escritorio in planificacion_optuna.items()}



hora_cierre           = '23:00:00'  

registros_atenciones, fila, n_minutos = simv04(un_dia, hora_cierre, planificacion, niveles_servicio_x_serie)   
print(f"atendidos {len(registros_atenciones) }, en espera { len(fila) }")        
end_time = time.time()
elapsed_time = end_time - start_time
print(f"el simulador demoró {elapsed_time} segundos. Simulación desde las {str(un_dia.FH_Emi.min().time())} hasta las {hora_cierre} ({n_minutos/60} horas simuladas).")

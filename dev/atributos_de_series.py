#%%
import numpy as np
ids_series           = [6,  7,  11]
sla_porcen_user      = [20, 60, 80]
sla_corte_user       = [10*60, 15*60, 25*60]
prioridades_user     = [1,2,3]
pasos_user           = [2,4,5] 


ids_series         = np.array(ids_series, dtype=int)
sla_porcen_user    = np.array(sla_porcen_user, dtype=float)
sla_corte_user     = np.array(sla_corte_user, dtype=int)
prioridades_user   = np.array(prioridades_user, dtype=int)
pasos_user         = np.array(pasos_user, dtype=int)
atributos_x_series = [
                     {'serie': s} for s in ids_series
                     ]

import itertools


def update_atributos_x_series(atributos_x_series, sla_input_user, atributo):
    

    return  [atr_dict | 
                        {atributo: np.random.randint(20,90) if sla_input_user is None else sla_p, 
                        } for
                        atr_dict, sla_p in 
                        zip(atributos_x_series, itertools.repeat( None)  if sla_input_user is None else sla_input_user)]
    
    
atributos_x_series = update_atributos_x_series(atributos_x_series, sla_porcen_user, 'sla_porcen')
update_atributos_x_series(atributos_x_series, sla_corte_user, 'sla_corte')

#%%
if sla_porcen_user is None:
    atributos_x_series = [atr_dict | 
                            {'sla_porcen': np.random.randint(10,90) if sla_porcen_user is None else sla_p, 
                            } for
                            atr_dict, sla_p in zip(atributos_x_series, itertools.repeat( None)  if sla_porcen_user is None else sla_porcen_user)]
else:
    atributos_x_series = [{'serie': s, 
                            'sla_porcen': p, 
                            } for
                            s,p in zip(ids_series, sla_porcen_user)]
if sla_corte_user is None:
    atributos_x_series = [{'serie': s, 
                            'sla_corte':  np.random.randint(1,20*60), 
                            } for
                            s in ids_series]
else:
    atributos_x_series = [{'serie': s, 
                            'sla_corte': p, 
                            } for
                            s,p in zip(ids_series, sla_corte_user)]


atributos_x_series = [{'serie': s, 'sla_porcen': np.random.randint(10,90), 
                        'sla_corte': np.random.randint(1,20*60),
                        'prioridad':None, 
                        'pasos':None } for
                        s in ids_series]

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


prioridades_user     = [1,2,3]


prioridades_autom    = rank_by_magnitude([calcular_prioridad(s['sla_porcen'], s['sla_corte']) for s in atributos_x_series])
#atributos_x_series  = [s|{'prioridad':p} for s,p in zip(atributos_x_series, prioridades_autom)]

pasos_user           = [2,4,5] 
atributos_x_series   = [s|{'pasos':p} for s,p in zip(atributos_x_series, pasos_user)]





atributos_x_series
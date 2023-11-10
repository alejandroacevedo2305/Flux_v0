#%%
import numpy as np
import itertools

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
            
            
#%% Ejemplo de uso
if __name__ == "__main__":
    atributos_x_serie(ids_series=[6,  7,  11], sla_porcen_user=[20, 60, 80], 
                  sla_corte_user=[10*60, 15*60, 25*60], pasos_user=None, prioridades_user=[1,2,3])
#%%
# atributos_x_serie(ids_series=[6,  7,  11, 10], sla_porcen_user=None, 
#                   sla_corte_user=None, pasos_user=[6,4,4,4], prioridades_user=[4,3,2,1])
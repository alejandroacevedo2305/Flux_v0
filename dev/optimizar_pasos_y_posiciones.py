#%%
from src.datos_utils import DatasetTTP
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def shift_series(series, n, direction="right", plot=True, title = 'title'):

    mean = series.mean()
    std_dev = series.std()

    shift_value = n * std_dev

    if direction == "right":
        shifted_series = series + shift_value
    elif direction == "left":
        shifted_series = series - shift_value
    else:
        raise ValueError("Invalid direction. Choose either 'right' or 'left'.")

    noise = np.random.normal(10, 60, len(series))
    shifted_series += noise

    shifted_series = np.abs(shifted_series).astype(int)
    
    # Plot the histograms if plot is True
    if plot:
        plt.hist(series, alpha=0.5, label='Original Series', color='blue', edgecolor='black')
        plt.hist(shifted_series, alpha=0.5, label='Shifted Series', color='orange', edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    return pd.Series(shifted_series)
import copy
tiempos_de_espera                        = un_dia.T_Esp
tiempos_de_espera_workforce_proyectado   = shift_series(tiempos_de_espera, 1, direction='left', title = 'teórico proyectado')
tiempos_de_espera_workforce_historico    = shift_series(tiempos_de_espera, 2, direction='left', title = 'real histórico')
workforce_proyectado = copy.deepcopy(un_dia)
workforce_historico  = copy.deepcopy(un_dia)
workforce_proyectado.T_Esp = tiempos_de_espera_workforce_proyectado
workforce_historico.T_Esp = tiempos_de_espera_workforce_historico


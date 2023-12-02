#%%


import os
import pandas as pd
os.chdir("/DeepenData/Repos/Flux_v0")
import warnings

warnings.filterwarnings("ignore")
import time
from data.mocks import planificacion_simulador

import releases.simv6_1 as sim
from src.viz_utils import sla_x_serie, plot_all_reports_two_lines

dataset = sim.DatasetTTP.desde_csv_atenciones(
    "data/fonasa_monjitas.csv.gz"
)  # 

dataset = sim.DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills = sim.obtener_skills(el_dia_real)



#%%
series = sorted(list({val for sublist in skills.values() for val in sublist}))
registros_atenciones = pd.DataFrame()
tabla_atenciones = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
registros_atenciones = tabla_atenciones.copy()
registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int)
registros_x_serie = [registros_atenciones[registros_atenciones.IdSerie == s] for s in series]
df_pairs_his = [(sla_x_serie(r_x_s, '1H', corte=20), s) for r_x_s, s in zip(registros_x_serie, series)]


plot_all_reports_two_lines(df_pairs_1 = [df_pairs_his[idx] for idx in [0,1,2,3,4,5, 0]], 
                        df_pairs_2 = [df_pairs_his[idx] for idx in    [5,4,3,2,1,0, 1]], 
                        label_1="SLA histórico", label_2="SLA con IA", 
                        color_1="navy", color_2="purple", 
                        n_rows=4, n_cols=2, height=10, width=12, main_title="FONASA, Monjitas, 2023-05-15.")
#%%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,4))
axs = axs.ravel() 
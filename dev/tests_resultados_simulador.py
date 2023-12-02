#%%


import os
import pandas as pd
os.chdir("/DeepenData/Repos/Flux_v0")
import warnings

warnings.filterwarnings("ignore")
import time
from data.mocks import planificacion_simulador

import releases.simv6_1 as sim
from src.viz_utils import sla_x_serie, plot_all_reports_two_lines, plot_count_and_avg_two_lines

dataset = sim.DatasetTTP.desde_csv_atenciones(
    "data/fonasa_monjitas.csv.gz"
)  # 

dataset = sim.DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills = sim.obtener_skills(el_dia_real)


series = sorted(list({val for sublist in skills.values() for val in sublist}))
registros_atenciones = pd.DataFrame()
tabla_atenciones = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
registros_atenciones = tabla_atenciones.copy()
registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int)
registros_x_serie = [registros_atenciones[registros_atenciones.IdSerie == s] for s in series]
df_pairs_his = [(sla_x_serie(r_x_s, '1H', corte=20), s) for r_x_s, s in zip(registros_x_serie, series)]


# plot_all_reports_two_lines(df_pairs_1 = [df_pairs_his[idx] for idx in [0,1,2,3,4,5, 0]], 
#                         df_pairs_2 = [df_pairs_his[idx] for idx in    [5,4,3,2,1,0, 1]], 
#                         label_1="SLA histórico", label_2="SLA con IA", 
#                         color_1="navy", color_2="purple", 
#                         n_rows=4, n_cols=2, height=10, width=12, main_title="FONASA, Monjitas, 2023-05-15.")

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10,12))
axs      = axs.ravel() 



def plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, ax1, label_1, label_2, color_1, color_2, serie=''):
    # Debugging statement:
    print(f'Type of df_count_1: {type(df_count_1)}')
    # Ensure df_count_1 is a dataframe
    if not isinstance(df_count_1, pd.DataFrame):
        raise TypeError("df_count_1 is not a DataFrame!")

    # Extract start and end times
    start_times = df_count_1['FH_Emi'].dt.strftime('%H:%M:%S')
    end_times = (df_count_1['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S')

    # Create the x_labels using the extracted start and end times
    x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(start_times, end_times)]
    
    bars = ax1.bar(x_labels, df_count_1['demanda'], alpha=0.6, label='Demanda', edgecolor='white', width=0.75)
    # Create a second y-axis
    ax2 = ax1.twinx()
    ##############
        # Ensure the data lengths match before plotting
    min_len_1 = min(len(x_labels), len(df_avg_1['espera']))
    min_len_2 = min(len(x_labels), len(df_avg_2['espera']))
    
    # Use the minimum length to slice the data and plot
    ax2.plot(x_labels[:min_len_1], df_avg_1['espera'][:min_len_1], color=color_1, marker='o', linestyle='--', label=label_1)
    ax2.plot(x_labels[:min_len_2], df_avg_2['espera'][:min_len_2], color=color_2, marker='o', linestyle='-', label=label_2)
    # Add a transparent shaded area between the two lines
    # Find the shorter length among the two series
    min_len = min(min_len_1, min_len_2)

    # Modified fill_between section to handle color changes based on conditions
    ax2.fill_between(
        x_labels[:min_len], 
        df_avg_1['espera'][:min_len], 
        df_avg_2['espera'][:min_len], 
        where=(df_avg_1['espera'][:min_len] < df_avg_2['espera'][:min_len]), 
        interpolate=True, 
        color='green', 
        alpha=0.2
    )

    ax2.fill_between(
        x_labels[:min_len], 
        df_avg_1['espera'][:min_len], 
        df_avg_2['espera'][:min_len], 
        where=(df_avg_1['espera'][:min_len] >= df_avg_2['espera'][:min_len]), 
        interpolate=True, 
        color='red', 
        alpha=0.13
    )


    ax1.set_xlabel('')
    ax1.set_ylabel('Demanda (#)', color='black')
    ax2.set_ylabel('SLA (%)', color='black')    
    ax2.set_ylim([0, 1.1*pd.concat([df_avg_1, df_avg_2], axis = 0).espera.max()])
    ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
    ax1.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor", size =7)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 1.37))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 1.37))
    ax1.set_title(f"Serie {serie}", y=1.34) 

    # Add grid and background color
    ax1.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax2.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax1.set_facecolor((0.75, 0.75, 0.75, .9))  
    ax2.set_facecolor((0.75, 0.75, 0.75, .9))  

# Plotting function for all subplots with two lines
def plot_all_reports_two_lines(df_pairs_1, df_pairs_2, label_1, label_2, color_1, color_2, n_rows, n_cols, height, width, main_title):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    axs = axs.ravel()  
    for i, (pair_1, pair_2) in enumerate(zip(df_pairs_1, df_pairs_2)):
        # Unpacking the tuple correctly
        (df_count_1, df_avg_1), _ = pair_1
        (df_count_2, df_avg_2), serie = pair_2
        
        plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, axs[i], label_1, label_2, color_1, color_2, serie=serie)

    fig.subplots_adjust(hspace=1.1,  wspace=.35)  
    fig.suptitle(main_title, y=.98, fontsize=12)
    plt.show()

esperas_x_serie = [(registros_atenciones[registros_atenciones.IdSerie == s].drop('IdSerie',axis=1, inplace = False
                                                               ).set_index('FH_Emi', inplace=False).resample('1H').count().rename(columns={'espera': 'demanda'}).reset_index(),
  registros_atenciones[registros_atenciones.IdSerie == s].drop('IdSerie',axis=1, inplace = False
                                                               ).set_index('FH_Emi', inplace=False).resample('1H').mean().reset_index(),
  s)
 for s in series]

import random
# df_pairs_1 = random.sample(esperas_x_serie, len(esperas_x_serie))
# df_pairs_2 = random.sample(esperas_x_serie, len(esperas_x_serie))
# for i, (pair_1, pair_2) in enumerate(zip(df_pairs_1, df_pairs_2)):
#     print(pair_1)
#     # Unpacking the tuple correctly
#     df_count_1, df_avg_1, _ = pair_1
#     df_count_2, df_avg_2, serie = pair_2
#     plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, axs[i], "hola1", "hola2", "navy", "purple", serie=serie)



# fig.subplots_adjust(hspace=1.1,  wspace=.35)  
# fig.suptitle(t = 'main_title', y=.98, fontsize=12)
# plt.show()
#%%

######################
#------Simulacion-----
######################
un_dia = dataset.un_dia("2023-05-15").sort_values(by="FH_Emi", inplace=False)
start_time = time.time()
hora_cierre = "15:30:00"
# planificacion = sim.plan_desde_skills(skills, inicio="08:00:00", porcentaje_actividad=1)
registros_atenciones_simulacion, fila = sim.simv06(
    un_dia, hora_cierre, planificacion_simulador)#, log_path="dev/simulacion.log")
print(f"{len(registros_atenciones_simulacion) = }, {len(fila) = }")
end_time = time.time()
print(f"tiempo total: {end_time - start_time:.1f} segundos")

#%%
registros_atenciones_simulacion = registros_atenciones_simulacion.astype({'FH_Emi': 'datetime64[s]', 'IdSerie': 'int', 'espera': 'int'})[["FH_Emi","IdSerie","espera"]].reset_index(drop=True)

esperas_x_serie_simulados = [(registros_atenciones_simulacion[registros_atenciones_simulacion.IdSerie == s].drop('IdSerie',axis=1, inplace = False
                                                               ).set_index('FH_Emi', inplace=False).resample('1H').count().rename(columns={'espera': 'demanda'}).reset_index(),
  registros_atenciones_simulacion[registros_atenciones_simulacion.IdSerie == s].drop('IdSerie',axis=1, inplace = False
                                                               ).set_index('FH_Emi', inplace=False).resample('1H').mean().reset_index(),
  s)
 for s in series]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10,12))
axs      = axs.ravel() 
df_pairs_1 = esperas_x_serie #random.sample(esperas_x_serie, len(esperas_x_serie))
df_pairs_2 = esperas_x_serie_simulados #random.sample(registros_atenciones_simulacion, len(esperas_x_serie))
for i, (pair_1, pair_2) in enumerate(zip(df_pairs_1, df_pairs_2)):
    print(pair_1)
    # Unpacking the tuple correctly
    df_count_1, df_avg_1, _ = pair_1
    df_count_2, df_avg_2, serie = pair_2
    plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, axs[i], "hola1", "hola2", "navy", "purple", serie=serie)



fig.subplots_adjust(hspace=1.1,  wspace=.35)  
fig.suptitle(t = 'main_title', y=.98, fontsize=12)
plt.show()
#%%
import os
os.chdir("/DeepenData/Repos/Flux_v0")
import warnings
warnings.filterwarnings("ignore")
from datetime import date, time
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, time
import math
import random 
from src.forecast_utils import *

##----------------------Datos históricos de un día----------------
dataset     = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills      = obtener_skills(el_dia_real)
series      = sorted(list({val for sublist in skills.values() for val in sublist}))
#SLAs        = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {5: (0.7, 20),
                                10: (0.34, 20),
                                11: (0.6, 20),
                                12: (0.34, 20),
                                14: (0.7, 20),
                                17: (0.7, 20)}#{s:random.choice(SLAs) for s in series}
registros_atenciones            = pd.DataFrame()
tabla_atenciones         = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
SLA_index                = 0
SLA_df                   = pd.DataFrame()
Espera_index =0 
Espera_df               = pd.DataFrame()             


def tiempo_espera_x_serie_para_reporte(registros_atenciones, series):   
    esperas_x_serie = {}
    for serie in series:
        espera_una_serie    = registros_atenciones[registros_atenciones.IdSerie == serie]['espera']
        if not espera_una_serie.empty:
            promedio_espera_cum = (espera_una_serie.expanding().mean()/60).iloc[-1]
            esperas_x_serie     = esperas_x_serie | {serie: int(promedio_espera_cum)}
        
    return esperas_x_serie
for cliente_seleccionado in tabla_atenciones.iterrows():
    

    un_cliente       =  pd.DataFrame(cliente_seleccionado[1][['FH_Emi', 'IdSerie', 'espera']]).T
    registros_atenciones    =  pd.concat([registros_atenciones, un_cliente])#.reset_index(drop=True)
    SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(registros_atenciones, niveles_servicio_x_serie).items()), columns=['keys', 'values'])
    SLA_index+=1                        
    SLA_una_emision['index']  = SLA_una_emision.shape[0]*[SLA_index]
    SLA_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    #SLA_una_emision['espera']= un_cliente.espera[un_cliente.espera.index[0]]
    SLA_df                    = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)
    ######################################################################################################################################
    Espera_una_emision        =  pd.DataFrame(list(tiempo_espera_x_serie_para_reporte(registros_atenciones, series).items()), columns=['keys', 'values'])
    Espera_index+=1
    Espera_una_emision['index']  = Espera_una_emision.shape[0]*[Espera_index]
    Espera_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    Espera_df                    = pd.concat([Espera_df, Espera_una_emision], ignore_index=True)

def sla_x_serie(df, interval='1H', corte=45, factor_conversion_T_esp:int=60):
    df = df.reset_index(drop=False)
    df['FH_Emi'] = pd.to_datetime(df['FH_Emi'])  # Convert to datetime
    df['IdSerie'] = df['IdSerie'].astype(str)  # Ensuring IdSerie is string
    df['espera'] = df['espera'].astype(float)  # Convert to float
    df['espera'] = df['espera']/factor_conversion_T_esp  
    # Set FH_Emi as the index for resampling
    df.set_index('FH_Emi', inplace=True)
    # First DataFrame: Count of events in each interval
    df_count = df.resample(interval).size().reset_index(name='Count')
    # Second DataFrame: Percentage of "espera" values below the threshold
    def percentage_below_threshold(x):
        return (x < corte).mean() * 100
    df_percentage = df.resample(interval)['espera'].apply(percentage_below_threshold).reset_index(name='espera')
    
    return df_count, df_percentage

def plot_count_and_avg(df_count, df_avg, ax1, color_sla:str='navy', serie:str="_", tipo_SLA:str="SLA"):
    x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(df_count['FH_Emi'].dt.strftime('%H:%M:%S'), (df_count['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S'))]
    bars = ax1.bar(x_labels, df_count['Count'], alpha=0.6, label='Demanda')
    ax2 = ax1.twinx()
    ax2.plot(x_labels, df_avg['espera'], color=color_sla, marker='o', label=f'{tipo_SLA}')
    ax1.set_xlabel('')
    ax1.set_ylabel('Demanda (#)', color='black')
    ax2.set_ylabel(f'{tipo_SLA} (%)', color='black')    
    ax2.set_ylim([0, 105])
    ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
    ax1.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor", size =7)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 1.22))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 1.22))
    ax1.set_title(f"Serie {serie}", y=1.15) 

    # New code to add a grey grid and light grey background color
    ax1.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  # Adding grey grid to ax1
    ax2.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  # Adding grey grid to ax2
    ax1.set_facecolor((0.75, 0.75, 0.75, 0.85))  # Adding light grey background color to ax1
    ax2.set_facecolor((0.75, 0.75, 0.75, 0.85))  # Adding light grey background color to ax2

    
def plot_all_reports(df_pairs, tipo_SLA, color_sla, n_rows:int=3, n_cols:int=2, heigth:float=10, width:float=10):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, heigth))  # 2x3 grid of subplots
    axs = axs.ravel()  # Flatten the grid to easily loop through it
    for i, ((df_count, df_avg), serie) in enumerate(df_pairs):
        plot_count_and_avg(df_count, df_avg, axs[i], serie = serie, tipo_SLA= tipo_SLA, color_sla=color_sla)
    fig.subplots_adjust(hspace=1, wspace=.45)  # Adjusts the vertical space between subplots. Default is 0.2.
    #plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
   

registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]



df_pairs = [(sla_x_serie(r_x_s, '1H', corte = corte), s) for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                          [20, 20, 20, 20, 20, 20])]

from scipy import stats

def calculate_geometric_mean(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return stats.gmean(series)


all_SLAs = [sla_x_serie(r_x_s, '1H', corte = corte)[1]['espera'] for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                          [20, 20, 20, 20, 20, 20])]

tuple(calculate_geometric_mean(series) for series in all_SLAs)

#plot_all_reports(df_pairs[0:3], tipo_SLA = "SLA", color_sla="darkgoldenrod",n_rows=1,n_cols=3)


#plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,]], tipo_SLA = "SLA histórico", color_sla="darkgoldenrod",n_rows=1,n_cols=2, heigth=10, width=10)
plot_all_reports([df_pairs[idx] for idx in [0,1,2,3,4,5]], tipo_SLA = "SLA histórico", color_sla="navy",n_rows=3,n_cols=2, heigth=10, width=12)

#plot_all_reports(df_pairs, tipo_SLA = "SLA", color_sla="darkgoldenrod")


#%%
trajectorias_SLAs    = SLA_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
trajectorias_esperas = Espera_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
trajectorias_SLAs.droplevel(0).plot(rot=45, ylabel='Nivel de servicio (%)')
trajectorias_esperas.droplevel(0).plot(rot=45, ylabel='Tiempo espera (min.)')
#%%
#-----------------Forecast (MM)--------------------------------

#-----------------------Workforce----------------------------
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
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
storage = optuna.storages.get_storage("sqlite:///workforce_manager_SLA_objs.db")
# Retrieve all study names
all_study_names = optuna.study.get_all_study_summaries(storage)
relevant_study_names = [s.study_name for s in all_study_names if "workforce_" in s.study_name]
# Infer the number of partitions
num_partitions = len(relevant_study_names)



rows = len(el_dia_real)
chunk_size = rows // 3
partitions = [el_dia_real.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]

assert num_partitions == len(tramos)



#%%
# Initialize the plot
# fig = make_subplots(rows=1, cols=num_partitions, shared_yaxes=True, subplot_titles=tramos)
# # Loop to load each partition study and extract all trials
# for idx, tramo in enumerate(tramos):  # Assuming `tramos` is defined elsewhere
#     study_name = f"workforce_{idx}"
#     # Load the study
#     study = optuna.multi_objective.load_study(study_name=study_name, storage=storage)
#     # Extract all trials
#     all_trials = study.get_trials(deepcopy=False)
#     # Initialize lists for the current partition
#     current_partition_values_0 = []
#     current_partition_values_1 = []
#     current_partition_values_2 = []
#     hover_texts = []
#     for trial in all_trials:
#         # Skip if the trial is not complete
#         if trial.state != optuna.trial.TrialState.COMPLETE:
#             continue
#         # Append the values to the lists for the current partition
#         current_partition_values_0.append(trial.values[0])
#         current_partition_values_1.append(trial.values[1])
#         current_partition_values_2.append(trial.values[2])
#         hover_texts.append(f"{format_dict_for_hovertext(trial.user_attrs.get('planificacion'))}")  # Creating hover text from trial parameters
#     # Plotting
#     fig.add_trace(
#         go.Scatter(
#             x=current_partition_values_1,
#             y=current_partition_values_0,
#             mode='markers',
#             marker=dict(opacity=0.5,
#                         size=current_partition_values_2,
#                         line=dict(
#                 width=2,  # width of border line
#                 color='black'  # color of border line
#             )),
#             hovertext=hover_texts,
#             name=f"{tramo}"
#         ),
#         row=1,
#         col=idx + 1  # Plotly subplots are 1-indexed
#     )
#     # Customizing axis
#     fig.update_xaxes(title_text="n Escritorios")#, tickvals=list(range(min(current_partition_values_1), max(current_partition_values_1)+1)), col=idx+1)
#     fig.update_yaxes(title_text="SLA global", row=1, col=idx+1)
# # Show plot
# fig.update_layout(title="Subplots with Hover Information")
# fig.show()
from scipy.stats import gmean
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
        {
            i: ( 1*( np.mean([v for v in trial.values])), trial.user_attrs.get('planificacion')) 
            #i: ( trial.values[1], trial.user_attrs.get('planificacion'))
            for i, trial in enumerate(all_trials) if trial.state == optuna.trial.TrialState.COMPLETE}
        }
mejores_configs = []

for k,v in estudios.items():

   tuple_list = [plan for trial, plan in v.items() ]
   selected_tuple = max(tuple_list, key=lambda x: x[0])
   
   mejores_configs.append(selected_tuple[1])

workforce_recommendation = regenerate_global_keys(mejores_configs)#mejores_configs[0]
prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
Workforce_SLAs, Workforce_Esperas, Workforce_df_count, Workforce_df_avg, WorkForce = simular(workforce_recommendation, niveles_servicio_x_serie, el_dia_real, prioridades)



# plot_count_and_avg(df_count, df_avg)

# plot_count_and_avg(df_count, Workforce_df_avg)




WorkForce['IdSerie'] = WorkForce['IdSerie'].astype(int) 
registros_x_serie               = [WorkForce[WorkForce.IdSerie==s] for s in series]

df_pairs_wkf = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                          [20, 20, 20, 20, 20, 20])]
#plot_all_reports(df_pairs, tipo_SLA = "SLA IA", color_sla="darkred")
#%%
plot_all_reports([df_pairs_wkf[idx] for idx in [0,2]], tipo_SLA = "SLA IA", color_sla="darkred",n_rows=1,n_cols=2, heigth=3, width=10)

#%%
Workforce_SLAs.set_index('hora', inplace=False).plot(rot=45, ylabel='Nivel de servicio (%)')
Workforce_Esperas.droplevel(0).plot(rot=45, ylabel='Tiempo espera (min.)')
print(f"n Escritorios utilizados histórico: {skills.__len__()}")
print(f"n Escritorios utilizados WorkForce: {max([c.__len__() for c in mejores_configs])}")
# %%




#%%
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
SLAs        = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
registros_sla            = pd.DataFrame()
tabla_atenciones         = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
SLA_index                = 0
SLA_df                   = pd.DataFrame()
Espera_index =0 
Espera_df               = pd.DataFrame()             

def tiempo_espera_x_serie(registros_sla, series):   
    esperas_x_serie = {}
    for serie in series:
        espera_una_serie    = registros_sla[registros_sla.IdSerie == serie]['espera']
        if not espera_una_serie.empty:
            promedio_espera_cum = (espera_una_serie.expanding().mean()/60).iloc[-1]
            esperas_x_serie     = esperas_x_serie | {serie: int(promedio_espera_cum)}
        
    return esperas_x_serie

for cliente_seleccionado in tabla_atenciones.iterrows():
    

    un_cliente       =  pd.DataFrame(cliente_seleccionado[1][['FH_Emi', 'IdSerie', 'espera']]).T
    registros_sla    =  pd.concat([registros_sla, un_cliente])#.reset_index(drop=True)
    SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(registros_sla, niveles_servicio_x_serie).items()), columns=['keys', 'values'])
    SLA_index+=1                        
    SLA_una_emision['index']  = SLA_una_emision.shape[0]*[SLA_index]
    SLA_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    #SLA_una_emision['espera']= un_cliente.espera[un_cliente.espera.index[0]]
    SLA_df                    = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)
    ######################################################################################################################################
    Espera_una_emision        =  pd.DataFrame(list(tiempo_espera_x_serie(registros_sla, series).items()), columns=['keys', 'values'])
    Espera_index+=1
    Espera_una_emision['index']  = Espera_una_emision.shape[0]*[Espera_index]
    Espera_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    Espera_df                    = pd.concat([Espera_df, Espera_una_emision], ignore_index=True)


trajectorias_SLAs    = SLA_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
trajectorias_esperas = Espera_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
trajectorias_SLAs.droplevel(0).plot(rot=45, ylabel='Nivel de servicio (%)')
trajectorias_esperas.droplevel(0).plot(rot=45, ylabel='Tiempo espera (min.)')

# %%
#-----------------Forecast (MM)--------------------------------
# %%
#-----------------------Workforce----------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
def format_dict_for_hovertext(dictionary):
    formatted_str = ""
    for key, value in dictionary.items():
        formatted_str += f"'{key}': {json.dumps(value)},<br>"
    return formatted_str[:-4]  # Remove the trailing HTML line break
rows       = len(el_dia_real)
chunk_size = rows // 4
partitions = [el_dia_real.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
tramos     = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]

storage              = optuna.storages.get_storage("sqlite:///multiple_studies.db")
all_study_names      = optuna.study.get_all_study_summaries(storage)
relevant_study_names = [s.study_name for s in all_study_names if "workforce_" in s.study_name]
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
    current_partition_values_2 = []
    hover_texts = []
    for trial in all_trials:
        # Skip if the trial is not complete
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Append the values to the lists for the current partition
        current_partition_values_0.append(trial.values[0])
        current_partition_values_1.append(trial.values[1])
        current_partition_values_2.append(trial.values[2])
        hover_texts.append(f"{format_dict_for_hovertext(trial.user_attrs.get('planificacion'))}")  # Creating hover text from trial parameters
    # Plotting
    fig.add_trace(
        go.Scatter(
            x=current_partition_values_1,
            y=current_partition_values_0,
            mode='markers',
            marker=dict(opacity=0.5,
                        size=current_partition_values_2,
                        line=dict(
                width=2,  # width of border line
                color='black'  # color of border line
            )),
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
#%%
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
            i: ( trial.values[0]/(trial.values[1]+trial.values[2]), trial.user_attrs.get('planificacion')) 
            for i, trial in enumerate(all_trials) if trial.state == optuna.trial.TrialState.COMPLETE}
        }
mejores_configs = []

for k,v in estudios.items():

   tuple_list = [plan for trial, plan in v.items() ]
   selected_tuple = max(tuple_list, key=lambda x: x[0])
   
   mejores_configs.append(selected_tuple[1])
mejores_configs
#%%
workforce_recommendation = regenerate_global_keys(mejores_configs)#mejores_configs[0]
prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
registros_con_workforce  = simular(workforce_recommendation, niveles_servicio_x_serie, el_dia_real, prioridades)
# %%

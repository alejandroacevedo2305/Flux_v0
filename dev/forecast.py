#%%
import warnings
warnings.filterwarnings("ignore")

from datetime import date, time
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datetime import date, time
import math
import random 

class ForecasterTTP:

    def __init__(self, 
        dataset : DatasetTTP, 
        series : list[int], 
        atenciones_diarias_minimas : int = 15, 
        horario : tuple[time, time] = ('8:00', '16:00'), 
        fecha_minima : date | None = None
        ):
        
        self.dataset = dataset
        self.series = series

        self.generador = None # generador vacio

        # LOGICA DE ATENCIONES
        self.dataset.atenciones_agg["Demanda_diaria"] = self.dataset.atenciones_agg_dia["Demanda"]

        self.dataset.atenciones_agg["Demanda_diaria"] = self.dataset.atenciones_agg["Demanda_diaria"].ffill()
        self.dataset.atenciones_agg = self.dataset.atenciones_agg[ self.dataset.atenciones_agg["Demanda_diaria"] >= atenciones_diarias_minimas ]
        
        # Se encarga de recortar los bloques cerrados de la oficina
        self.dataset.atenciones_agg = (  dataset
            .atenciones_agg
            .reset_index(level=['IdOficina','IdSerie'], drop=False)
            .between_time( start_time = horario[0], end_time = horario[1] )
            .fillna(0)
        )

        self.delta_horas : int = ( pd.Timestamp( horario[1] ) - pd.Timestamp( horario[0] ) ).seconds // 3600

        # En este punto el unico indice es una DateTime index, asi que puedo cortar
        if fecha_minima:
            # '2023-05-05'
            self.dataset.atenciones_agg = self.dataset.atenciones_agg[ dataset.atenciones_agg.index.to_series() > pd.Timestamp(fecha_minima) ]

        # Cursed linea que crea un indice monotonicamente ascendente. Nota, no es necesario que sea base 0 para cada grupo
        self.dataset.atenciones_agg = dataset.atenciones_agg.reset_index().rename_axis(index='dt_idx').reset_index()


    @staticmethod # Una cuestion lacra, que deberia ser implementada en Pandas
    def sliding_window(ts : list[float], features : int):
        """
        Retorna dos listas de x [-features, -1] , [0] sobre una lista
        
        Ejemplo: X es `[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 
        6, 7], [6, 7, 8], [7, 8, 9]]`, e Y es `[[4], [5], [6], [7], [8], [9],  
        [10]]`.    
        """
        X : list[list[float]]= []
        Y : list[list[float]]= []

        for i in range(features + 1, len(ts) + 1):
            X.append(ts[i - (features + 1):i - 1])
            Y.append([ts[i - 1]])

        return X #, Y # no necesito Y, de momento


    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def proyectar_demanda_SARIMAX(self , dia_inicio : date ):
        """Genera un forecasting via SARIMAX"""

        dataset = self.dataset
        series = self.series

        timelimit = pd.Timestamp(dia_inicio) # SELECCIONA UN DIA
        timelimit_below = timelimit - pd.Timedelta(weeks=4)
        timelimit_upper = timelimit + pd.Timedelta(weeks=1)

        self.serie_t_espera : pd.Serie = dataset.atenciones_agg[
                (timelimit_below <= dataset.atenciones_agg["FH_Emi"]) & (dataset.atenciones_agg["FH_Emi"] <= timelimit) 
            ].groupby(by=[ "IdSerie"]).median()["T_Ate"]

        demandas_proyectadas = {}
        
        for SERIE in series: 
            
            ts_pre = dataset.atenciones_agg[ 
                (timelimit_below <= dataset.atenciones_agg["FH_Emi"]) & 
                (dataset.atenciones_agg["FH_Emi"] <= timelimit) & 
                (dataset.atenciones_agg["IdSerie"] == SERIE) 
            ]["Demanda"].tolist() 

            ts_post = dataset.atenciones_agg[ 
                (dataset.atenciones_agg["FH_Emi"] > timelimit) & 
                (dataset.atenciones_agg["FH_Emi"] < timelimit_upper) & 
                (dataset.atenciones_agg["IdSerie"] == SERIE) 
            ]["Demanda"].tolist() 


            windows = ForecasterTTP.sliding_window(ts_pre, features=self.delta_horas*10) # [0]
            last_values = []

            for wind in windows:
                model = SARIMAX(
                    endog = wind,
                    exog = None,
                    order = (1, 1, 1),
                    seasonal_order = (1, 1, 1, self.delta_horas)
                )

                results = model.fit(disp = 0)
                forecast = results.forecast()
                last_values.append([forecast[0]])

            demandas_proyectadas[f"{SERIE}_proyectada"] = [ max(0, value[0]) for value in last_values  ] # FIXME: esto esta corrido en idx 1
            demandas_proyectadas[f"{SERIE}_real"] = ts_post

        self.dia_inicio = dia_inicio
        self.demandas_proyectadas = demandas_proyectadas

        return self
    
    def demandas_proyectada_cuantizadas(self):
        """Retorna una funcion cuantizada de la demanda proyectada"""

        demandas_proyectadas = self.demandas_proyectadas
        timeindex = self.dataset.atenciones_agg["FH_Emi"].drop_duplicates()

        for SERIE in demandas_proyectadas.keys():
            n_dias = pd.Series(demandas_proyectadas[SERIE]).apply(math.ceil)
            n_dias[ n_dias < 0 ] = 0 # Reemplaza los ceros
            n_dias_len = n_dias.size
            demandas_proyectadas[SERIE] = n_dias.set_axis(
                timeindex[ (timeindex > (pd.Timestamp(self.dia_inicio) - pd.Timedelta(weeks=20))) ].iloc[:n_dias_len] + pd.Timedelta(weeks=20)
            )
            
        df = pd.DataFrame(demandas_proyectadas)
        df = df[ df.columns[df.columns.str.endswith('_proyectada')] ]
        df.columns = df.columns.str.removesuffix('_proyectada')
        df = df.unstack(level=0).reset_index().rename({"level_0" : "IdSerie", 0 : "Demanda"}, axis='columns')
        df = df.loc[df.index.repeat(df["Demanda"])].reset_index(drop=True)

        df["FH_Emi"] = df["FH_Emi"].apply( lambda x : x + pd.Timedelta(seconds=random.randint(1,3599)) )
        df["T_Ate"] = df["IdSerie"].apply( lambda x : forecaster_ttp.serie_t_espera[int(x)] ) # Cosa para incluir tiempos de atencion estimados

        return df.drop("Demanda", axis="columns").sort_values(by="FH_Emi").reset_index(drop=True)

    @staticmethod
    def un_dia(dia : date, demandas_proyectadas_cuantizada : pd.DataFrame):
        """Retorna las atenciones de un dia predicho"""

        inicio_dia = pd.Timestamp( dia )
        fin_dia = pd.Timestamp( dia ) + pd.Timedelta(days=1)

        return demandas_proyectadas_cuantizada[ (inicio_dia <= demandas_proyectadas_cuantizada["FH_Emi"]) & (demandas_proyectadas_cuantizada["FH_Emi"] <= fin_dia) ]

# EJEMPLO DE USO
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
SERIES = [5, 10, 11, 12, 14, 17]#[10, 14, 12, 17, 5, 11]

forecaster_ttp = ForecasterTTP(dataset=dataset, series=SERIES, atenciones_diarias_minimas=1)
# Genera una proyeccion de demanda, 
demandas_proyectadas_q = forecaster_ttp.proyectar_demanda_SARIMAX('2023-05-15').demandas_proyectada_cuantizadas()
demandas_proyectadas_q = ForecasterTTP.un_dia('2023-05-15', demandas_proyectadas_q) # Retorna solo un dia 2023-05-15


from pandas import Timestamp, Timedelta

df = demandas_proyectadas_q #pd.DataFrame(data)

# Task 1: Convert the dtypes
df['IdSerie'] = df['IdSerie'].astype('Int8')  # Convert IdSerie to Int8
df['T_Ate'] = df['T_Ate'].astype('int32')  # Convert T_Ate to int32
df['FH_Emi'] = pd.to_datetime(df['FH_Emi'])  # Ensure FH_Emi is in datetime64[ns] format

# Task 2: Create a new column FH_AteFin
# Convert T_Ate to Timedelta in minutes, then add to FH_Emi
df['FH_AteFin'] = df['FH_Emi'] + pd.to_timedelta(df['T_Ate'], unit='s')

# Task 3: FH_AteFin is already in the same format as FH_Emi (datetime64[ns])

# Task 4: Create another column FH_AteIni
# Add 2 minutes to each entry in the FH_Emi column
df['FH_AteIni'] = df['FH_Emi'] + Timedelta(minutes=2)

# Show the DataFrame to confirm changes
un_dia = df



def extract_skills_length(data):
    result = {}
    
    # Initialize a variable to store the sum of all lengths.
    total_length = 0
    
    # Iterate over keys and values in the input dictionary.
    for key, entries in data.items():
        # Initialize an empty list to store the lengths for this key.
        lengths_for_key = []
        
        # Iterate over each entry which is a dictionary.
        for entry in entries:
            # Access the 'skills' field, and calculate its length.
            skills_length = len(entry['propiedades']['skills'])
            
            # Add the current length to the total_length.
            total_length += skills_length
            
            # Append this length to the list for this key.
            lengths_for_key.append(skills_length)
        
        # Store the list of lengths in the result dictionary, using the same key.
        result[key] = lengths_for_key
    
    # Return the result dictionary and the total sum of all lengths.
    return total_length #, result
def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))
skills   = {'escritorio_7': [10, 12],
 'escritorio_10': [17, 14],
 'escritorio_12': [17, 14],
 'escritorio_11': [17, 14],
 'escritorio_13': [17, 10, 14],
 'escritorio_2': [10, 11, 5],
 'escritorio_4': [10, 11, 12, 5],
 'escritorio_9': [10, 12, 5],
 'escritorio_6': [10, 5],
 'escritorio_5': [10, 5],
 'escritorio_8': [10, 12],
 'escritorio_1': [10, 11, 12],
 'escritorio_3': [10, 11]}#obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
#SLAs     = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {5: (0.7, 45),
                            10: (0.34, 35),
                            11: (0.6, 30),
                            12: (0.34, 35),
                            14: (0.7, 45),
                            17: (0.7, 45)}#{s:random.choice(SLAs) for s in series}

#%%
def objective(trial, un_dia,skills, subsets, niveles_servicio_x_serie,  modos_atenciones:list = ["Alternancia", "FIFO", "Rebalse"]):
    try:
        bool_vector  = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]
        str_dict     = {i: trial.suggest_categorical(f'{i}',         modos_atenciones) for i in range(len(skills.keys()))} 
        subset_idx = {i: trial.suggest_int(f'ids_{i}', 0, len(subsets) - 1) for i in range(len(skills.keys()))}
        prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        planificacion            =  {}
        inicio                   =  str(un_dia.FH_Emi.min().time())#'08:33:00'
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
                        'skills':list(subsets[subset_idx[key]]),
                        #'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key],
                        #'zz': 0,
                    }
                }
                # Convert integer key to string and add the inner dictionary to a list, then add it to the output dictionary
                planificacion[str(key)] = [inner_dict]
        #print(f"---------------------------{planificacion}")
        trial.set_user_attr('planificacion', planificacion)
        # registros_SLA, Workforce_Esperas,_,_,_ = simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)       
        
        # obj1 = gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna())
        # obj2 = sum(bool_vector)
        # obj3 = extract_skills_length(planificacion)
        # obj4 = np.mean(Workforce_Esperas.iloc[-1].dropna())
        # print(f"SLA global: {obj1}, T espera global {obj4}, n Escritorios: {obj2}, n Series cargadas: {obj3}")
        
        # return obj1, obj2, obj3, obj4 #gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna()), sum(bool_vector), extract_skills_length(planificacion)
        registros_atenciones = optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades)
        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        series   = sorted(list({val for sublist in skills.values() for val in sublist}))
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
        
        from scipy import stats
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

        def calculate_geometric_mean(series):
            series = series.dropna()
            if series.empty:
                return np.nan
            return stats.gmean(series)        
        
        all_SLAs = [sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1)[1]['espera'] for r_x_s, s, corte in zip(registros_x_serie, series,
                                                                          [20, 20, 20, 20, 20, 20])]
        obj1, obj2, obj3, obj4, obj5, obj6 =  tuple(calculate_geometric_mean(series) for series in all_SLAs)
        
        
        print(f"maximizando SLAs: {obj1, obj2, obj3, obj4, obj5, obj6}")
        return  obj1, obj2, obj3, obj4, obj5, obj6

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

rows = len(un_dia)
chunk_size = rows // 4
partitions = [un_dia.iloc[i:i + chunk_size] for i in range(0, rows, chunk_size)]
# Create a SQLite storage to save all studies
storage = optuna.storages.get_storage("sqlite:///workforce_manager_SLA_objs.db")
tramos = [f"{str(p.FH_Emi.min().time())} - {str(p.FH_Emi.max().time())}" for p in partitions]

# Loop to optimize each partition
for idx, part in enumerate(partitions):

    # Create a multi-objective study object and specify the storage
    study_name = f"workforce_{idx}"  # Unique name for each partition
    study = optuna.multi_objective.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize','maximize','maximize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)
    # Optimize the study, the objective function is passed in as the first argument
    subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))
    study.optimize(lambda trial: objective(trial, 
                                        un_dia                   = part,
                                        skills                   = skills,
                                        subsets                  = subsets,
                                        niveles_servicio_x_serie = niveles_servicio_x_serie,
                                        ), 
                                        n_trials                 = 50)

#%%
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
storage = optuna.storages.get_storage("sqlite:///workforce_manager.db")
# Retrieve all study names
all_study_names = optuna.study.get_all_study_summaries(storage)
relevant_study_names = [s.study_name for s in all_study_names if "workforce_" in s.study_name]
# Infer the number of partitions
num_partitions = len(relevant_study_names)
num_partitions, len(tramos)

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


#%% -------------------------------------------

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

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)


workforce_recommendation = regenerate_global_keys(mejores_configs)#mejores_configs[0]
prioridades   =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
registros_SLA_real_con_worforce = simular(workforce_recommendation, niveles_servicio_x_serie, el_dia_real, prioridades)

#%%
# def calcula_SAL_hitorico:


registros_sla            = pd.DataFrame()
tabla_atenciones         = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
SLA_index                = 0
SLA_df                   = pd.DataFrame()

for cliente_seleccionado in tabla_atenciones.iterrows():
    

    un_cliente   = pd.DataFrame(cliente_seleccionado[1][['FH_Emi', 'IdSerie', 'espera']]).T
    registros_sla   =  pd.concat([registros_sla, un_cliente])#.reset_index(drop=True)
    SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(registros_sla, niveles_servicio_x_serie).items()), columns=['keys', 'values'])

    SLA_index+=1                        
    SLA_una_emision['index']            = SLA_una_emision.shape[0]*[SLA_index]
    SLA_una_emision['hora'] = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#SLA_una_emision.time().strftime('%H:%M:%S')
    SLA_df                              = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)


trajectoria_sla = SLA_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)

trajectoria_sla.iloc[0:700]

# # Instancia el generador
# forecaster_ttp.generador = demandas_proyectadas_q.iterrows()

# # COSA PARA GENERAR GRAFICOS A TIEMPOS ARBITRARIOS
# for i, SERIE in enumerate(SERIES):

#     pd.DataFrame({
#         # Empaca series para lidear con el no tener el mismo largo
#         f"{SERIE}_proyectada" : pd.Series(forecaster_ttp.demandas_proyectadas[f"{SERIE}_proyectada"]),
#         f"{SERIE}_real"       : pd.Series(forecaster_ttp.demandas_proyectadas[f"{SERIE}_real"]),
#     }).dropna().plot.line(title = f'Serie {SERIE}')

# %%

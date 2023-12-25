#%%
import os
#os.chdir("/DeepenData/Repos/Flux_v0")
import warnings
warnings.filterwarnings("ignore")
import time
import os
from datetime import date
import pandas as pd  # Dataframes
import polars as pl  # DFs, es más rapido
from pydantic import BaseModel, Field, ConfigDict  # Validacion de datos
from datetime import date, datetime
import releases.simv7 as sim
import copy
import optuna
import random
import time
import math

import numpy as np
def plan_para_wfm(planificacion):   
    for k,v in planificacion.items():
        v[0]['inicio'] = None
        v[0]['propiedades']['configuracion_atencion'] = None
        for att in v[0]['propiedades']['atributos_series']:
            att['pasos'] = None
    return planificacion

ID_DATABASE = 'V24_Fonav30'
# Adicionalmente disponibles: "V24_Provida", "V24_Fonav30", "V24_Cruz", "V24_Afpmodelo"
#V24_Provida: 13 -  'V24_Fonav30': 2 - V24_Afpmodelo: 1
ID_OFICINA = 2
FECHA      = "2023-03-15"
DB_CONN    = "mysql://autopago:Ttp-20238270@totalpackmysql.mysql.database.azure.com:3306/capacity_data_fonasa"
dataset    = sim.DatasetTTP(connection_string=DB_CONN, id_oficina=ID_OFICINA)


# el_dia_real, plan  = dataset.un_dia(fecha=FECHA)

# el_dia_real['T_Ate'] = (el_dia_real['FH_AteFin'] - el_dia_real['FH_AteIni']).astype('timedelta64[s]').dt.total_seconds().astype(int)
# el_dia_real['T_Esp'] = (el_dia_real['FH_AteIni'] - el_dia_real['FH_Emi']).astype('timedelta64[s]').dt.total_seconds().astype(int)
# el_dia_real = el_dia_real.sort_values(by='FH_Emi', inplace=False).astype(
#     {
#         'FH_Emi': 'datetime64[s]',
#         'FH_Llama': 'datetime64[s]',
#         'FH_AteIni': 'datetime64[s]',
#         'FH_AteFin': 'datetime64[s]',}).reset_index(drop=True)
# porcentaje_actividad =.8
# planificacion_wfm    =  plan_para_wfm(sim.plan_desde_skills(skills=sim.obtener_skills(el_dia_real) , 
#                                                 inicio = '08:00:00', 
#                                                 porcentaje_actividad=.8))


def objective(trial, el_dia_real, intervalo, planificacion_wfm, skills_subsets_x_escr, series, optimizar:str='SLA'):
    
    planificacion_optuna    = copy.deepcopy(planificacion_wfm)
    modos_atenciones : list = ["Alternancia", "FIFO", "Rebalse"]
    pasos_trial = {}
    try:
        for j,((k,v),s) in enumerate(zip(planificacion_wfm.items(), skills_subsets_x_escr)):
            escr_skills   = list(s[trial.suggest_int(f'idx_skills_{j}', 0, len(s) - 1)]) if len(s)>1 else v[0]['propiedades']['skills']
            modo_atencion = trial.suggest_categorical(f'modos_{j}', modos_atenciones) if len(s)>1 else "FIFO"
            
            if modo_atencion== 'Alternancia':
                for i, att in enumerate( v[0]['propiedades']['atributos_series']):
                    
                    pasos_trial = pasos_trial | {att['serie']: trial.suggest_int(f'pasos_{i}_{j}', 1, 4)}
            else:
                pasos_trial = {}
        
            planificacion_optuna[k][0]['propiedades']['skills']                 = escr_skills
            planificacion_optuna[k][0]['propiedades']['configuracion_atencion'] = modo_atencion
            planificacion_optuna[k][0]['inicio']                                = intervalo[0]
            planificacion_optuna[k][0]['termino']                               = intervalo[1]  
  
            
            if modo_atencion== 'Alternancia':
                for att, (pasos_k, pasos_v) in zip(planificacion_optuna[k][0]['propiedades']['atributos_series'], pasos_trial.items()):
                    #print(f"pasos_k, pasos_v {pasos_k, pasos_v}")
                    att['pasos'] = pasos_v  
        
        #---SOLO Para minimizar escritorios (SLA + escritorios):----------
        
        if optimizar == "SLA + escritorios" or  optimizar == "SLA + escritorios + skills":    
            vector_length = len(planificacion_wfm.keys())  # Adjust this as needed
            true_index = trial.suggest_int('true_index', 0, vector_length - 1)
            boolean_vector = [False] * vector_length
            boolean_vector[true_index] = True
            for i in range(vector_length):
                if i != true_index:
                    boolean_vector[i] = trial.suggest_categorical(f'bool_{i}', [True, False])
            planificacion_optuna = {k: v for k, v, m in zip(planificacion_optuna.keys(), planificacion_optuna.values(), boolean_vector) if m}
            #print(f"boolean_vector: {boolean_vector} - sum(boolean_vector): {sum(boolean_vector)}")   
            
        
            
        unique_skills_optuna = set(skill for entry in planificacion_optuna.values() for prop in entry for skill in prop['propiedades']['skills'])

        assert unique_skills_optuna == set(series), "no todas las series están en la planificacion sugerida"
        
        
        
        
        #hora_cierre          = "18:30:00"
        
        
        sim.update_configuracion_atencion(planificacion_optuna)

        
        registros_atenciones, fila = sim.simv7_1(
                                    un_dia           = el_dia_real , 
                                    hora_cierre      = intervalo[1], 
                                    planificacion    = planificacion_optuna,
                                    probabilidad_pausas =  0.75,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
                                    factor_pausas       = .020,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
                                    params_pausas       =  [0, 1/3, 1] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
                                )
        
        
        
        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
        pocentajes_SLA        = [int(100*v[0])for k,v in niveles_servicio_x_serie.items()]
        mins_de_corte_SLA     = [int(v[1])for k,v in niveles_servicio_x_serie.items()]        
        df_pairs              = [(sim.sla_x_serie(r_x_s, '1H', corte = corte), s) 
                                    for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]
        porcentajes_reales    = {f"serie: {serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 
        assert not any(math.isnan(x) for x in [v for k, v in porcentajes_reales.items()]), "porcentajes_reales contains at least one nan"

        #print(f"porcentajes_reales {porcentajes_reales}")

        trial.set_user_attr('planificacion', planificacion_optuna) 
        dif_cuadratica        = {k:((sla_real-sla_teorico)**2 if sla_real < sla_teorico else abs((sla_real-sla_teorico)) ) 
                                 for ((k,sla_real),sla_teorico) in zip(porcentajes_reales.items(),pocentajes_SLA)}

        if optimizar == "SLA":
            
            #print(f"--OBJ-- maximizar_SLAs           {tuple(dif_cuadratica.values())}")
            return  tuple((dif_cuadratica.values()))
        
        elif optimizar == "SLA + escritorios":
            print(f"maximizar_SLAs y minimizar_escritorios { tuple(dif_cuadratica.values()) + (sum(boolean_vector),)}")
            return  tuple((dif_cuadratica.values())) + (int(sum(boolean_vector)),)

                                
        elif optimizar == "SLA + skills":
            
            print(f"maximizar_SLAs y minimizar_skills { tuple(dif_cuadratica.values()) + (sim.extract_skills_length(planificacion_optuna)**2,) }")
            return  tuple((dif_cuadratica.values())) + (int(sim.extract_skills_length(planificacion_optuna)**2),)
        
        elif optimizar == "SLA + escritorios + skills":
            
            print(f"SLA + escritorios + skills { tuple(dif_cuadratica.values()) + (sum(boolean_vector),) + (sim.extract_skills_length(planificacion_optuna)**2,)}")
            return  tuple((dif_cuadratica.values())) + (int(sum(boolean_vector),)) + (int(sim.extract_skills_length(planificacion_optuna)**2),)               
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()

# skills_subsets_x_escr    = [sim.non_empty_subsets(sorted(v[0]['propiedades']['skills'])) for k,v in planificacion_wfm.items()]


# niveles_servicio_x_serie = {atr_dict['serie']:
#                             (atr_dict['sla_porcen']/100, atr_dict['sla_corte']) 
#                             for atr_dict in planificacion_wfm['5'][0]['propiedades']['atributos_series']}

# series                   = list(niveles_servicio_x_serie.keys())
# start_time               = time.time()

# optimizar    = "SLA + skills" #"SLA + escritorios" #"SLA" #"SLA + escritorios + skills" #"SLA" | "SLA + escritorios" | "SLA + skills" | "SLA + escritorios + skills"

# hora_cierre  = '17:00:00'  
# n_objs       = int(
#                         len(series)
#                         if optimizar == "SLA"
#                         else len(series) + 1
#                         if optimizar in {"SLA + escritorios", "SLA + skills"}
#                         else len(series) + 2
#                         if optimizar == "SLA + escritorios + skills"
#                         else None
#                         )
# sampler = optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler()
# IA      = optuna.multi_objective.create_study(directions= n_objs*['minimize'], sampler=sampler)
# import logging
# #logging.getLogger('optuna').setLevel(logging.WARNING)
# IA.optimize(lambda trial: objective(trial, 
#                                     el_dia_real = el_dia_real,
#                                     hora_cierre = "17:00:00",
#                                     planificacion_wfm   = planificacion_wfm,
#                                     skills_subsets_x_escr=skills_subsets_x_escr,
#                                     series = series,
#                                     optimizar = optimizar,
#                                            ),
#                    n_trials  = 15, #int(1e4),  # Make sure this is an integer
#                    n_jobs=1,
#                    #timeout   = 2*3600,   #  hours
#                    )  

# planificacion_optuna = [trial for trial in IA.trials if trial.state == optuna.trial.TrialState.COMPLETE][-1].user_attrs.get('planificacion')
# end_time = time.time()

# print(f"tiempo total: {(end_time - start_time)/60:.1f} minutos")


######################
#------Simulacion-----
######################
# import time
# start_time           = time.time()
# hora_cierre          = "18:30:00"
# registros_atenciones, fila = sim.simv7_1(
#                             un_dia           = el_dia_real , 
#                             hora_cierre      = hora_cierre, 
#                             planificacion    = planificacion_optuna,
#                             probabilidad_pausas =  0.75,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
#                             factor_pausas       = .020,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
#                             params_pausas       =  [0, 1/3, 1] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
#                         )
#     #, log_path="dev/simulacion.log")
# print(f"{len(registros_atenciones) = }, {len(fila) = }")
# end_time = time.time()
# print(f"tiempo total: {end_time - start_time:.1f} segundos")
# sim.compare_historico_vs_simulacion(el_dia_real, registros_atenciones,  ID_DATABASE, ID_OFICINA,FECHA ,porcentaje_actividad)

###########################################
#------Implementacion para intevalos--------
############################################

el_dia_real, _  = dataset.un_dia(fecha=FECHA)
el_dia_real, plan  = dataset.un_dia(fecha=FECHA)

el_dia_real['T_Ate'] = (el_dia_real['FH_AteFin'] - el_dia_real['FH_AteIni']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real['T_Esp'] = (el_dia_real['FH_AteIni'] - el_dia_real['FH_Emi']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real = el_dia_real.sort_values(by='FH_Emi', inplace=False).astype(
    {
        'FH_Emi': 'datetime64[s]',
        'FH_Llama': 'datetime64[s]',
        'FH_AteIni': 'datetime64[s]',
        'FH_AteFin': 'datetime64[s]',}).reset_index(drop=True)
porcentaje_actividad =.8
planificacion_wfm    =  plan_para_wfm(sim.plan_desde_skills(skills=sim.obtener_skills(el_dia_real) , 
                                                inicio = '08:00:00', 
                                                porcentaje_actividad=.8))

skills_subsets_x_escr    = [sim.non_empty_subsets(sorted(v[0]['propiedades']['skills'])) for k,v in planificacion_wfm.items()]


niveles_servicio_x_serie = {atr_dict['serie']:
                            (atr_dict['sla_porcen']/100, atr_dict['sla_corte']) 
                            for atr_dict in planificacion_wfm['5'][0]['propiedades']['atributos_series']}

series                   = list(niveles_servicio_x_serie.keys())

intervals  = sim.get_time_intervals(el_dia_real, n = 4) # Una funcion que recibe un dia, un intervalo, y un porcentaje de actividad para todos los intervalos
partitions = sim.partition_dataframe_by_time_intervals(el_dia_real, intervals) # TODO: implementar como un static del simulador? 
n_trials   = 50
optimizar    = "SLA + skills" #"SLA + escritorios" #"SLA" #"SLA + escritorios + skills" #"SLA" | "SLA + escritorios" | "SLA + skills" | "SLA + escritorios + skills"
n_objs       = int(
                        len(series)
                        if optimizar == "SLA"
                        else len(series) + 1
                        if optimizar in {"SLA + escritorios", "SLA + skills"}
                        else len(series) + 2
                        if optimizar == "SLA + escritorios + skills"
                        else None
                        )


start_time           = time.time()
storage = optuna.storages.get_storage("sqlite:///alejandro_wfm7.db")

for idx, (part, intervalo) in enumerate(zip(partitions,intervals)):
    study_name = f"intervalo_{idx}"
    IA = optuna.multi_objective.create_study(directions= n_objs*['minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)
    print(f"idx: {idx}, {intervalo} study_name: {study_name}")
    IA.optimize(lambda trial: objective(trial, 
                                    el_dia_real = part,
                                    intervalo = intervalo,
                                    planificacion_wfm   = planificacion_wfm,
                                    skills_subsets_x_escr=skills_subsets_x_escr,
                                    series = series,
                                    optimizar = optimizar,
                                           ),
                   n_trials  = n_trials, #int(1e4),  # Make sure this is an integer
                   n_jobs=1,
                   #timeout   = 2*3600,   #  hours
                   )  
end_time = time.time()
print(f"tiempo total: {end_time - start_time:.1f} segundos")


recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_wfm7.db") # Objetivos de 6-salidas
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "intervalo_" in s.study_name]
scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: np.mean([x for x in trial.values if x is not None]) 
                for
                    trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
                    } 


trials_optimos          = sim.extract_min_value_keys(scores_studios) # Para cada tramo, extrae el maximo, 
planificaciones_optimas = {}   
for k,v in trials_optimos.items():
    un_estudio               = optuna.multi_objective.load_study(study_name=k, storage=recomendaciones_db)
    trials_de_un_estudio     = un_estudio.get_trials(deepcopy=False)
    planificaciones_optimas  = planificaciones_optimas | {f"{k}":
        trial.user_attrs.get('planificacion')#calcular_optimo(trial.values)
                for
                    trial in trials_de_un_estudio if trial.number == v[0]
                    }   
    
planificacion_optima_para_sim   =  sim.plan_unico([plan for tramo,plan in planificaciones_optimas.items()])



planificacion_optima_df        = sim.transform_to_dataframe(planificaciones_optimas)
# %%
######################
#------Simulacion-----
######################
# import time
start_time           = time.time()
hora_cierre          = "18:30:00"
registros_atenciones, fila = sim.simv7_1(
                            un_dia           = el_dia_real , 
                            hora_cierre      = hora_cierre, 
                            planificacion    = planificacion_optima_para_sim,
                            probabilidad_pausas =  0.75,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
                            factor_pausas       = .020,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
                            params_pausas       =  [0, 1/3, 1] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
                        )
    #, log_path="dev/simulacion.log")
print(f"{len(registros_atenciones) = }, {len(fila) = }")
end_time = time.time()
print(f"tiempo total: {end_time - start_time:.1f} segundos")
sim.compare_historico_vs_simulacion(el_dia_real, registros_atenciones,  ID_DATABASE, ID_OFICINA,FECHA ,porcentaje_actividad)
planificacion_optima_df
###############----FIN------------------------
#%%
# foo = {'intervalo_0': 
#     {'5': [{'inicio': '08:35:47',
#         'termino': '09:58:42',
#         'propiedades': {'skills': [10],
#         'configuracion_atencion': 'FIFO',
#         'porcentaje_actividad': 0.8 }}],
#     '10': [{'inicio': '08:35:47',
#             'termino': '09:58:42',
#             'propiedades': {'skills': [14, 17],
#             'configuracion_atencion': 'Rebalse',
#             'porcentaje_actividad': 0.8}}],
#     '11': [{'inicio': '08:35:47',
#             'termino': '09:58:42',
#             'propiedades': {'skills': [5, 7, 14, 17],
#             'configuracion_atencion': 'Alternancia',
#             'porcentaje_actividad': 0.8}}],
#      },
#  'intervalo_1': 
#      {'9': [{'inicio': '09:58:42',
#             'termino': '11:21:37',
#             'propiedades': {'skills': [7],
#             'configuracion_atencion': 'FIFO',
#             'porcentaje_actividad': 0.8}}],
#     '7': [{'inicio': '09:58:42',
#             'termino': '11:21:37',
#             'propiedades': {'skills': [14, 17],
#             'configuracion_atencion': 'Alternancia',
#             'porcentaje_actividad': 0.8}}],
#       },
#  'intervalo_2': 
#      {'12': [{'inicio': '11:21:37',
#             'termino': '13:21:37',
#             'propiedades': {'skills': [1, 10],
#             'configuracion_atencion': 'Rebalse',
#             'porcentaje_actividad': 0.8}}],
      
#      },
#       'intervalo_3': 
#     {'5': [{'inicio': '13:21:37',
#         'termino': '15:21:37',
#         'propiedades': {'skills': [10],
#         'configuracion_atencion': 'Alternancia',
#         'porcentaje_actividad': 0.8 }}],
#     '10': [{'inicio': '13:21:37',
#             'termino': '15:21:37',
#             'propiedades': {'skills': [ 17],
#             'configuracion_atencion': 'Rebalse',
#             'porcentaje_actividad': 0.8}}],
#     '11': [{'inicio': '13:21:37',
#             'termino': '15:21:37',
#             'propiedades': {'skills': [14],
#             'configuracion_atencion': 'Alternancia',
#             'porcentaje_actividad': 0.8}}],
      
#      }}

# foo2 = {'5': [{'inicio': '08:35:47',
#         'termino': '09:58:42',
#         'propiedades': {'skills': [10],
#         'configuracion_atencion': 'Rebalse',
#         'porcentaje_actividad': 0.8 }}],
#     '10': [{'inicio': '08:35:47',
#             'termino': '09:58:42',
#             'propiedades': {'skills': [14, 17],
#             'configuracion_atencion': 'Rebalse',
#             'porcentaje_actividad': 0.8}}],
#     '11': [{'inicio': '08:35:47',
#             'termino': '09:58:42',
#             'propiedades': {'skills': [5, 7, 14, 17],
#             'configuracion_atencion': 'Alternancia',
#             'porcentaje_actividad': 0.8}}],
#      }
# def update_configuracion_atencion(input_data):
#     # Iterate through the keys (e.g., '5', '10', '11')
#     for key, records in input_data.items():
#         # Process each record
#         for record in records:
#             # Check if 'skills' has length of one
#             if len(record['propiedades']['skills']) == 1:
#                 # Update 'configuracion_atencion' to 'FIFO'
#                 record['propiedades']['configuracion_atencion'] = 'FIFO'

#     #return input_data


# update_configuracion_atencion(foo2)
# foo2
# # def modify_configuracion_atencion(input_data):
# #     # Iterate through the intervals
# #     for interval, values in input_data.items():
# #         # Iterate through the sub-dictionaries in each interval
# #         for sub_interval, records in values.items():
# #             # Process each record
# #             for record in records:
# #                 # Check if skills list has only one element
# #                 if len(record['propiedades']['skills']) == 1:
# #                     # Set configuracion_atencion to 'FIFO'
# #                     record['propiedades']['configuracion_atencion'] = 'FIFO'
    
#     #return input_data

# # pd.DataFrame({
# #               "08:35:47-09:58:42": ["[10], FIFO", "[14, 17], Rebalse", "[5, 7, 14, 17], Alternancia"],
# #               "09:58:42-11:21:37": ["[7],  FIFO", "[14, 17], Alternancia", ''],
# #               "11:21:37-13:21:37":["[1, 10], Rebalse", "", ""],
# #               })

# #modify_configuracion_atencion(foo)


# # def transform_to_dataframe(input_data):
# #     # Initialize a dictionary to hold the processed data
# #     processed_data = {}

# #     # Iterate through the intervals (e.g., 'intervalo_0')
# #     for interval, values in input_data.items():
# #         # Iterate through the sub-dictionaries in each interval
# #         for sub_interval, records in values.items():
# #             # Process each record
# #             for record in records:
# #                 inicio = record['inicio']
# #                 termino = record['termino']
# #                 time_key = f"{inicio}-{termino}"

# #                 # Prepare the string for DataFrame entry
# #                 skills = record['propiedades']['skills']
# #                 config_atencion = record['propiedades']['configuracion_atencion']
# #                 data_entry = f"{skills}, {config_atencion}"

# #                 # Add the entry to the processed data
# #                 if time_key not in processed_data:
# #                     processed_data[time_key] = []
# #                 processed_data[time_key].append(data_entry)

# #     # Ensure each time interval has the same number of entries
# #     max_length = max(len(v) for v in processed_data.values())
# #     for key in processed_data:
# #         while len(processed_data[key]) < max_length:
# #             processed_data[key].append('')

# #     # Create the DataFrame
# #     df = pd.DataFrame(processed_data)
# #     return df


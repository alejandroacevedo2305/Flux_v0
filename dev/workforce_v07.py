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
# import copy
# import optuna
# import random
# import time
# import math

# import numpy as np
# def plan_para_wfm(planificacion):   
#     for k,v in planificacion.items():
#         v[0]['inicio'] = None
#         v[0]['propiedades']['configuracion_atencion'] = None
#         for att in v[0]['propiedades']['atributos_series']:
#             att['pasos'] = None
#     return planificacion


# def objective(trial, part_dia, intervalo,hora_cierre ,planificacion_wfm, skills_subsets_x_escr, series,pesos_x_serie, niveles_servicio_x_serie, optimizar:str='SLA'):
    
#     planificacion_optuna    = copy.deepcopy(planificacion_wfm)
#     modos_atenciones : list = ["Alternancia", 'Rebalse', "FIFO"]
#     pasos_trial = {}
#     try:
#         for j,((k,v),s) in enumerate(zip(planificacion_wfm.items(), skills_subsets_x_escr)):
#             escr_skills   = list(s[trial.suggest_int(f'idx_skills_{j}', 0, len(s) - 1)]) if len(s)>1 else v[0]['propiedades']['skills']
#             modo_atencion = trial.suggest_categorical(f'modos_{j}', modos_atenciones) if len(s)>1 else "FIFO"
            
#             if modo_atencion== 'Alternancia':
#                 for i, att in enumerate( v[0]['propiedades']['atributos_series']):
                    
#                     pasos_trial = pasos_trial | {att['serie']: trial.suggest_int(f'pasos_{i}_{j}', 1, 4)}
#             else:
#                 pasos_trial = {}
        
#             planificacion_optuna[k][0]['propiedades']['skills']                 = escr_skills
#             planificacion_optuna[k][0]['propiedades']['configuracion_atencion'] = modo_atencion
#             planificacion_optuna[k][0]['inicio']                                = intervalo[0]
#             planificacion_optuna[k][0]['termino']                               = intervalo[1] if intervalo[1] is not None else None
  
            
#             if modo_atencion== 'Alternancia':
#                 for att, (pasos_k, pasos_v) in zip(planificacion_optuna[k][0]['propiedades']['atributos_series'], pasos_trial.items()):
#                     #print(f"pasos_k, pasos_v {pasos_k, pasos_v}")
#                     att['pasos'] = pasos_v  
        
#         #---SOLO Para minimizar escritorios (SLA + escritorios):----------
        
#         if optimizar == "SLA + escritorios" or  optimizar == "SLA + escritorios + skills":    
#             vector_length = len(planificacion_wfm.keys())  # Adjust this as needed
#             true_index = trial.suggest_int('true_index', 0, vector_length - 1)
#             boolean_vector = [False] * vector_length
#             boolean_vector[true_index] = True
#             for i in range(vector_length):
#                 if i != true_index:
#                     boolean_vector[i] = trial.suggest_categorical(f'bool_{i}', [True, False])
#             planificacion_optuna = {k: v for k, v, m in zip(planificacion_optuna.keys(), planificacion_optuna.values(), boolean_vector) if m}
#             #print(f"boolean_vector: {boolean_vector} - sum(boolean_vector): {sum(boolean_vector)}")   
            
        
            
#         unique_skills_optuna = set(skill for entry in planificacion_optuna.values() for prop in entry for skill in prop['propiedades']['skills'])

#         assert unique_skills_optuna == set(series), "no todas las series están en la planificacion sugerida"
        
        
        
        
#         #hora_cierre          = "18:30:00"
        
        
#         sim.update_configuracion_atencion(planificacion_optuna)

        
#         registros_atenciones, fila = sim.simv7_1(
#                                     un_dia           = part_dia , 
#                                     hora_cierre      = hora_cierre,#"15:00:00",#intervalo[1], 
#                                     planificacion    = planificacion_optuna,
#                                     probabilidad_pausas =  .5,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
#                                     factor_pausas       = .05,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
#                                     params_pausas       =  [0, 1/10, 1] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
#                                 )
        
        
        
#         registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
#         registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
#         pocentajes_SLA        = [int(100*v[0])for k,v in niveles_servicio_x_serie.items()]
#         mins_de_corte_SLA     = [int(v[1])for k,v in niveles_servicio_x_serie.items()]        
#         df_pairs              = [(sim.sla_x_serie(r_x_s, '1H', corte = corte), s) 
#                                     for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]
#         porcentajes_reales    = {f"{serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 
#         #assert not any(math.isnan(x) for x in [v for k, v in porcentajes_reales.items()]), "porcentajes_reales contains at least one nan"

#         #print(f"porcentajes_reales {porcentajes_reales}")

#         trial.set_user_attr('planificacion', planificacion_optuna) 
#         dif_cuadratica        = {k:((sla_real-sla_teorico)**2 if sla_real < sla_teorico else 0 ) #abs((sla_real-sla_teorico)) 
#                                  for ((k,sla_real),sla_teorico) in zip(porcentajes_reales.items(),pocentajes_SLA)}
        
#         dif_cuadratica = {key: int(dif_cuadratica[key] * pesos_x_serie[key]) for key in dif_cuadratica}

#         #print(f"dif_cuadratica {dif_cuadratica}")

#         if optimizar == "SLA":
            
#             print(f"--OBJ-- maximizar_SLAs           {tuple((dif_cuadratica.values()))  + (len(fila)**2,)}")
#             return                                    tuple((dif_cuadratica.values()))  + (len(fila)**2,)
        
#         elif optimizar == "SLA + escritorios":
#             print(f"maximizar_SLAs y minimizar_escritorios {tuple((dif_cuadratica.values())) + (int(sum(boolean_vector)),) + (len(fila)**2,)}")
#             return                                          tuple((dif_cuadratica.values())) + (int(sum(boolean_vector)),) + (len(fila)**2,)

                                
#         elif optimizar == "SLA + skills":
            
#             print(f"maximizar_SLAs y minimizar_skills { tuple((dif_cuadratica.values())) + (int(sim.extract_skills_length(planificacion_optuna)**2),)  + (len(fila)**2,)}")
#             return                                      tuple((dif_cuadratica.values())) + (int(sim.extract_skills_length(planificacion_optuna)**2),)  + (len(fila)**2,)
        
#         elif optimizar == "SLA + escritorios + skills":
            
#             print(f"SLA + escritorios + skills {tuple((dif_cuadratica.values())) + (int(sum(boolean_vector)),) + (int(sim.extract_skills_length(planificacion_optuna)**2),)  + (len(fila)**2,)}")
#             return                              tuple((dif_cuadratica.values())) + (int(sum(boolean_vector)),) + (int(sim.extract_skills_length(planificacion_optuna)**2),)  + (len(fila)**2,)              
#     except Exception as e:
#         print(f"An exception occurred: {e}")
#         raise optuna.TrialPruned()



# def workforce(el_dia_real, planificacion_wfm, n_intervalos, pesos_x_serie, hora_cierre, tiempo_max_resultados, niveles_servicio_x_serie, optimizar, series):

#     el_dia_real['T_Ate'] = (el_dia_real['FH_AteFin'] - el_dia_real['FH_AteIni']).astype('timedelta64[s]').dt.total_seconds().astype(int)
#     el_dia_real['T_Esp'] = (el_dia_real['FH_AteIni'] - el_dia_real['FH_Emi']).astype('timedelta64[s]').dt.total_seconds().astype(int)
#     el_dia_real = el_dia_real.sort_values(by='FH_Emi', inplace=False).astype(
#         {
#             'FH_Emi': 'datetime64[s]',
#             'FH_Llama': 'datetime64[s]',
#             'FH_AteIni': 'datetime64[s]',
#             'FH_AteFin': 'datetime64[s]',}).reset_index(drop=True)
#     #porcentaje_actividad =.8


#     skills_subsets_x_escr    = [sim.non_empty_subsets(sorted(v[0]['propiedades']['skills'])) for k,v in planificacion_wfm.items()]
#     [l.reverse() for l in skills_subsets_x_escr]



    
#     intervals  = sim.get_time_intervals(el_dia_real, n = n_intervalos) # Una funcion que recibe un dia, un intervalo, y un porcentaje de actividad para todos los intervalos
#     #intervals  = [(x[0], None) if i == len(intervals) - 1 else x for i, x in enumerate(intervals)]

#     partitions = sim.partition_dataframe_by_time_intervals(el_dia_real, intervals) # TODO: implementar como un static del simulador? 
#     n_trials   = 1000
#     n_objs       = int(
#                             len(series) + 1
#                             if optimizar == "SLA"
#                             else len(series) + 2
#                             if optimizar in {"SLA + escritorios", "SLA + skills"}
#                             else len(series) + 3
#                             if optimizar == "SLA + escritorios + skills"
#                             else None
#                             )


#     start_time           = time.time()
#     storage = optuna.storages.get_storage(optuna_storage)

#     for idx, (part, intervalo) in enumerate(zip(partitions,intervals)):
#         study_name = f"{optimizar.replace(' + ', '_')}_n_intervalos_{n_intervalos}_intervalo_{idx}"
#         IA = optuna.multi_objective.create_study(directions= n_objs*['minimize'],
#                                                     study_name=study_name,
#                                                     storage=storage, load_if_exists=True)
#         print(f"idx: {idx}, {intervalo} study_name: {study_name}")
#         IA.optimize(lambda trial: objective(trial, 
#                                         part_dia = part,
#                                         intervalo = intervalo,
#                                         hora_cierre = hora_cierre,
#                                         planificacion_wfm   = planificacion_wfm,
#                                         skills_subsets_x_escr=skills_subsets_x_escr,
#                                         series = series,
#                                         optimizar = optimizar,
#                                         niveles_servicio_x_serie = niveles_servicio_x_serie,
#                                         pesos_x_serie = pesos_x_serie
#                                             ),
#                     n_trials  = n_trials, #int(1e4),  # Make sure this is an integer
#                     n_jobs=1,
#                     timeout   = int(tiempo_max_resultados/n_intervalos),   #  hours
#                     )  
#     end_time = time.time()
#     print(f"tiempo total: {end_time - start_time:.1f} segundos")


#     recomendaciones_db   = optuna.storages.get_storage(optuna_storage) # Objetivos de 6-salidas
#     resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
#     nombres              = [s.study_name for s in resumenes if f"{optimizar.replace(' + ', '_')}_n_intervalos_{n_intervalos}_intervalo_" in s.study_name]
#     scores_studios = {}
#     for un_nombre in nombres:
#         un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
#         trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
#         scores_studios        = scores_studios | {f"{un_nombre}":
#             { trial.number: np.mean([x for x in trial.values if x is not None]) 
#                     for
#                         trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
#                         } 


#     trials_optimos          = sim.extract_min_value_keys(scores_studios) # Para cada tramo, extrae el minmo, 
#     planificaciones_optimas = {}   
#     for k,v in trials_optimos.items():
#         un_estudio               = optuna.multi_objective.load_study(study_name=k, storage=recomendaciones_db)
#         trials_de_un_estudio     = un_estudio.get_trials(deepcopy=False)
#         planificaciones_optimas  = planificaciones_optimas | {f"{k}":
#             trial.user_attrs.get('planificacion')#calcular_optimo(trial.values)
#                     for
#                         trial in trials_de_un_estudio if trial.number == v[0]
#                         }   
        
#     planificacion_optima_para_sim   =  sim.plan_unico([plan for tramo,plan in planificaciones_optimas.items()])
#     planificacion_optima_df         =    sim.transform_to_dataframe(planificaciones_optimas)

#     ######################
#     #------Simulacion-----
#     ######################
#     # import time

#     #start_time           = time.time()
#     registros_atenciones, _ = sim.simv7_1(
#                                 un_dia           = el_dia_real , 
#                                 hora_cierre      = hora_cierre, 
#                                 planificacion    =  planificacion_optima_para_sim,#sim.plan_desde_skills(skills=sim.obtener_skills(el_dia_real), inicio = '08:00:00',porcentaje_actividad=.8) ,#planificacion_optima_para_sim,
#                                 probabilidad_pausas =  0.0,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
#                                 factor_pausas       = 1,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
#                                 params_pausas       =  [0, 1/10, 1/5] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
#                             )
#         #, log_path="dev/simulacion.log")
#     #print(f"{len(registros_atenciones) = }, {len(fila) = }")
#     #end_time = time.time()
#     #print(f"tiempo total: {end_time - start_time:.1f} segundos")
#     #sim.compare_historico_vs_simulacion(el_dia_real, registros_atenciones,  ID_DATABASE, ID_OFICINA,FECHA ,porcentaje_actividad)
#     return planificacion_optima_df, registros_atenciones
###############----FIN------------------------
#%%

###########################################
#------Implementacion para intevalos--------
############################################

ID_DATABASE = 'V24_Fonav30'
# Adicionalmente disponibles: "V24_Provida", "V24_Fonav30", "V24_Cruz", "V24_Afpmodelo"
#V24_Provida: 13 -  'V24_Fonav30': 2 - V24_Afpmodelo: 1
ID_OFICINA = 2
FECHA      = "2023-03-15"
DB_CONN    = "mysql://autopago:Ttp-20238270@totalpackmysql.mysql.database.azure.com:3306/capacity_data_fonasa"
dataset    = sim.DatasetTTP(connection_string=DB_CONN, id_oficina=ID_OFICINA)

el_dia_real, _  = dataset.un_dia(fecha=FECHA)

planificacion_wfm    =  sim.plan_para_wfm(sim.plan_desde_skills(skills=sim.obtener_skills(el_dia_real) , 
                                                    inicio = '08:00:00', 
                                                    porcentaje_actividad=.8))
n_intervalos = 3



hora_cierre = "15:00:00"
tiempo_max_resultados = 60*1 #secs
optuna_storage = "sqlite:///alejandro_wfm7.db"

optimizar    = "SLA + escritorios + skills" #SLA + escritorios + skills"# "SLA + skills" #SLA + escritorios" #"SLA + escritorios + skills" #"SLA + escritorios" #"SLA" #"SLA + escritorios + skills" #"SLA" | "SLA + escritorios" | "SLA + skills" | "SLA + escritorios + skills"
niveles_servicio_x_serie = {atr_dict['serie']:
                                (atr_dict['sla_porcen']/100, atr_dict['sla_corte']) 
                                for atr_dict in planificacion_wfm['5'][0]['propiedades']['atributos_series']}

series                   = list(niveles_servicio_x_serie.keys())
#pesos_x_serie = {'5': 1, '10': 20, '11': 1, '12': 10, '14': 20, '17':15}
pesos_x_serie = {str(s):1 for s in series}


planificacion_optima_df, registros_atenciones = sim.workforce7(el_dia_real, planificacion_wfm, n_intervalos, pesos_x_serie, hora_cierre, tiempo_max_resultados, niveles_servicio_x_serie,optimizar, series, optuna_storage)
# %%

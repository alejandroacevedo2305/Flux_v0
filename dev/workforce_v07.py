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

ID_DATABASE = 'V24_Fonav30'
# Adicionalmente disponibles: "V24_Provida", "V24_Fonav30", "V24_Cruz", "V24_Afpmodelo"
#V24_Provida: 13 -  'V24_Fonav30': 2 - V24_Afpmodelo: 1
ID_OFICINA = 2
FECHA      = "2023-03-15"
DB_CONN    = "mysql://autopago:Ttp-20238270@totalpackmysql.mysql.database.azure.com:3306/capacity_data_fonasa"
dataset    = sim.DatasetTTP(connection_string=DB_CONN, id_oficina=ID_OFICINA)

el_dia_real, plan  = dataset.un_dia(fecha=FECHA)

el_dia_real['T_Ate'] = (el_dia_real['FH_AteFin'] - el_dia_real['FH_AteIni']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real['T_Esp'] = (el_dia_real['FH_AteIni'] - el_dia_real['FH_Emi']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real = el_dia_real.sort_values(by='FH_Emi', inplace=False).astype(
    {
        'FH_Emi': 'datetime64[s]',
        'FH_Llama': 'datetime64[s]',
        'FH_AteIni': 'datetime64[s]',
        'FH_AteFin': 'datetime64[s]',})
import copy
porcentaje_actividad =.8
planificacion        = sim.plan_desde_skills(skills=sim.obtener_skills(el_dia_real) , 
                                                inicio = '08:00:00', 
                                                porcentaje_actividad=.8)

#%%
######################
#------Simulacion-----
######################
import time
start_time           = time.time()
hora_cierre          = "17:30:00"

registros_atenciones_simulacion, fila = sim.simv7_1(
                            un_dia           = el_dia_real , 
                            hora_cierre      = hora_cierre, 
                            planificacion    = planificacion,
                            probabilidad_pausas =  0.75,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
                            factor_pausas       = .020,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
                            params_pausas       =  [0, 1/3, 1] ,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
                        )
    #, log_path="dev/simulacion.log")
print(f"{len(registros_atenciones_simulacion) = }, {len(fila) = }")
end_time = time.time()
print(f"tiempo total: {end_time - start_time:.1f} segundos")
sim.compare_historico_vs_simulacion(el_dia_real, registros_atenciones_simulacion,  ID_DATABASE, ID_OFICINA,FECHA ,porcentaje_actividad)
#%%
#######################################
def plan_para_wfm(planificacion):   
    for k,v in planificacion.items():
        v[0]['inicio'] = None
        v[0]['propiedades']['configuracion_atencion'] = None
        for att in v[0]['propiedades']['atributos_series']:
            att['pasos'] = None
    return planificacion


planificacion_wfm =  plan_para_wfm(planificacion)
#skills            = sim.obtener_skills(el_dia_real)

#subsets           = sim.non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))

#sim.non_empty_subsets(sorted([1,2,3]))


#merged_skills = [skill for value in planificacion_wfm.values() for item in value for skill in item['propiedades']['skills']]
#sum(merged_skills)

import optuna
import random

def objective(trial, planificacion_wfm, skills_subsets_x_escr):
    modos_atenciones : list = ["Alternancia", "FIFO", "Rebalse"]
    pasos_trial = {}
    for j,((k,v),s) in enumerate(zip(planificacion_wfm.items(), skills_subsets_x_escr)):
        escr_skills            = list(s[trial.suggest_int(f'idx_skills_{j}', 0, len(s) - 1)]) if len(s)>1 else f"skills: {v[0]['propiedades']['skills']} <--"
        modo_atencion          = trial.suggest_categorical(f'modos_{j}', modos_atenciones) if len(s)>1 else f"FIFO forzado <--"
        
        if modo_atencion== 'Alternancia':
            for i, att in enumerate( v[0]['propiedades']['atributos_series']):
                
                pasos_trial = pasos_trial | {att['serie']: trial.suggest_int(f'pasos_{i}_{j}', 1, 4)}
        else:
            pasos_trial = {}
            
                
                
    #print(f"suggestion skills {escr_skills}")
    #print(f"suggestion modo_atencion {modo_atencion}")                            
    #print(f"suggestion pasos {pasos_trial}")

        
        
        #v[0]['propiedades']['skills'] = escr_skills
        #v[0]['propiedades']['configuracion_atencion'] = modo_atencion
        # for att in v[0]['propiedades']['atributos_series']:
        #     #att['pasos'] = pasos
        #     foo = 1+1
        # planificacion_trial |  {k: [{'propiedades': 
        #                              {'skills': escr_skills,
        #              'configuracion_atencion' : modo_atencion,
        #              'atributos_series': [{
                         
        #              }]}]}


    #----------------------
    #------reconstruir planificación para simv7 sin modificar la planificación input
    #--------------------    
    trial.set_user_attr('planificacion', planificacion_wfm)


        
                                    
                               
    return random.randint(1,5), random.randint(1,5) #sum([skill for value in planificacion_wfm.values() for item in value for skill in item['propiedades']['skills']]), 1
    


IA      = optuna.multi_objective.create_study(directions= 2*['minimize'])


skills_subsets_x_escr = [sim.non_empty_subsets(sorted(v[0]['propiedades']['skills'])) for k,v in planificacion_wfm.items()]




IA.optimize(lambda trial: objective(trial, planificacion_wfm   = planificacion_wfm,skills_subsets_x_escr=skills_subsets_x_escr,
                                           ),
                   n_trials  = 100, #int(1e4),  # Make sure this is an integer
                   #timeout   = 2*3600,   #  hours
                   )  

#%%
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import random


dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
modos    = ['Rebalse','Alternancia', 'Rebalse']
atributos_series = atributos_x_serie(ids_series=series, 
                                    sla_porcen_user=None, 
                                    sla_corte_user=None, 
                                    pasos_user=None, 
                                    prioridades_user=None)

niveles_servicio_x_serie = {atr_dict['serie']:
                           (atr_dict['sla_porcen']/100, atr_dict['sla_corte']/60) 
                           for atr_dict in atributos_series}

prioridades =       {atr_dict['serie']:
                    atr_dict['prioridad']
                    for atr_dict in atributos_series}

planificacion = {'0': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '1': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '12': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '33': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '34': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '35': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '49': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series), 
    'configuracion_atencion':random.sample(modos, 1)[0]}}],
 '50': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}],
 '51': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills':get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0]}}]}


#%%
registros_atenciones, l_fila =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) # 
registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
# %%

extract_skills_length(planificacion)

pocentajes_SLA    = [int(100*v[0])for k,v in niveles_servicio_x_serie.items()]
mins_de_corte_SLA = [int(v[1])for k,v in niveles_servicio_x_serie.items()]
df_pairs                        = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) 
                                    for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]


porcentajes_reales  = {f"serie: {serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 
dif_cuadratica      = {k:(v-p)**2 for ((k,v),p) in zip(porcentajes_reales.items(),pocentajes_SLA)}


import optuna
def objective(trial, 
    optimizar: str, 
    un_dia : pd.DataFrame,  # IdOficina  IdSerie  IdEsc, FH_Emi, FH_Llama  -- Deberia llamarse 'un_tramo'
    skills,  # {'escritorio_7': [10, 12], 'escritorio_10': [17, 14], 'escritorio_12': [17, 14], 'escritorio_11': <...> io_5': [10, 5], 'escritorio_8': [10, 12], 'escritorio_1': [10, 11, 12], 'escritorio_3': [10, 11]}
    subsets, # [(5,), (10,), (11,), (12,), (14,), (17,), (5, 10), (5, 11), (5, 12), (5, 14), (5, 17), (10, 11),  <...> 14, 17), (5, 10, 12, 14, 17), (5, 11, 12, 14, 17), (10, 11, 12, 14, 17), (5, 10, 11, 12, 14, 17)]
    niveles_servicio_x_serie,  # {5: (0.34, 35), 10: (0.34, 35), 11: (0.7, 45), 12: (0.34, 35), 14: (0.34, 35), 17: (0.6, 30)}
    
    modos_atenciones : list = ["Alternancia", "FIFO", "Rebalse"]
    ):    
    try:

        bool_vector              = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]
        str_dict                 = {i: trial.suggest_categorical(f'{i}',         modos_atenciones) for i in range(len(skills.keys()))} 
        subset_idx               = {i: trial.suggest_int(f'ids_{i}', 0, len(subsets) - 1) for i in range(len(skills.keys()))}   
        prioridades              =  prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        planificacion            =  {} # Arma una planificacion con espacios parametricos. 
        inicio                   =  str(un_dia.FH_Emi.min().time())#'08:33:00'
        termino                  =  str(un_dia.FH_Emi.max().time())#'14:33:00'

        for key in str_dict.keys():
            if bool_vector[key]:
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills':list(subsets[subset_idx[key]]), # Set -> Lista, para el subset 'subset_idx', para el escritorio 'key'
                        #'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key], # FI FAI FO FU
                        #'zz': 0,
                    }
                }
                planificacion[str(key)] = [inner_dict] # NOTE: Es una lista why -- Config por trial por tramo del escritorio 

        trial.set_user_attr('planificacion', planificacion) # This' actually cool 
        registros_atenciones, l_fila    =  optuna_simular(planificacion, niveles_servicio_x_serie, un_dia, prioridades) 
        registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int) 
        registros_x_serie               = [registros_atenciones[registros_atenciones.IdSerie==s] for s in series]
        
        
        pocentajes_SLA        = [int(100*v[0])for k,v in niveles_servicio_x_serie.items()]
        mins_de_corte_SLA     = [int(v[1])for k,v in niveles_servicio_x_serie.items()]        
        df_pairs              = [(sla_x_serie(r_x_s, '1H', corte = corte, factor_conversion_T_esp=1), s) 
                                    for r_x_s, s, corte in zip(registros_x_serie, series, mins_de_corte_SLA)]
        porcentajes_reales    = {f"serie: {serie}": np.mean(esperas.espera) for ((demandas, esperas), serie) in df_pairs} 
        dif_cuadratica        = {k:(v-p)**2 for ((k,v),p) in zip(porcentajes_reales.items(),pocentajes_SLA)}
        #Objetivos:        
        maximizar_SLAs        = tuple(dif_cuadratica.values())
        minimizar_escritorios = (sum(bool_vector),)
        minimizar_skills      = (extract_skills_length(planificacion),)
        
        if optimizar == "SLA":
            
            print(f"maximizar_SLAs {maximizar_SLAs}")
            return  maximizar_SLAs
        
        elif optimizar == "SLA + escritorios":
            
            print(f"maximizar_SLAs y minimizar_escritorios {maximizar_SLAs, minimizar_escritorios}")
            return  maximizar_SLAs + minimizar_escritorios
        
        elif optimizar == "SLA + skills":
            
            print(f"maximizar_SLAs y minimizar_skills {maximizar_SLAs, minimizar_skills}")
            return  maximizar_SLAs + minimizar_skills
        
        elif optimizar == "SLA + escritorios + skills":
            
            print(f"maximizar_SLAs, minimizar_escritorios y minimizar_skills  {maximizar_SLAs, minimizar_escritorios, minimizar_skills }")
            return  maximizar_SLAs + minimizar_escritorios + minimizar_skills           
        
    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()
    
    
storage    = optuna.storages.get_storage("sqlite:///alejandro_objs_v2.db")
intervals  = get_time_intervals(un_dia, 4, 100) # Una funcion que recibe un dia, un intervalo, y un porcentaje de actividad para todos los intervalos
partitions = partition_dataframe_by_time_intervals(un_dia, intervals) # TODO: implementar como un static del simulador? 
optimizar  = "SLA + escritorios + skills" #"SLA" | "SLA + escritorios" | "SLA + skills" | "SLA + escritorios + skills"
n_objs = int(
            len(series)
            if optimizar == "SLA"
            else len(series) + 1
            if optimizar in {"SLA + escritorios", "SLA + skills"}
            else len(series) + 2
            if optimizar == "SLA + escritorios + skills"
            else None
        )
n_trials   = 1
#%%
for idx, part in enumerate(partitions):
    study_name = f"tramo_{idx}"
    study = optuna.multi_objective.create_study(directions= n_objs*['minimize'],
                                                study_name=study_name,
                                                storage=storage, load_if_exists=True)
    # TODO: sacar fuera
    subsets = non_empty_subsets(sorted(list({val for sublist in skills.values() for val in sublist})))
    # Optimize with a timeout (in seconds)
    study.optimize(lambda trial: objective(trial,
                                           optimizar                = optimizar,
                                           un_dia                   = part,
                                           skills                   = skills,
                                           subsets                  = subsets,
                                           niveles_servicio_x_serie = niveles_servicio_x_serie),
                   n_trials  = n_trials, #int(1e4),  # Make sure this is an integer
                   timeout   = 2*3600,    #  hours
                   )  # 
#%%-------------Extraer el mejor objetivo--------------------
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import random

dataset  = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia   = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))


recomendaciones_db   = optuna.storages.get_storage("sqlite:///alejandro_objs_v2.db") # Objetivos de 6-salidas
resumenes            = optuna.study.get_all_study_summaries(recomendaciones_db)
nombres              = [s.study_name for s in resumenes if "tramo_" in s.study_name]

scores_studios = {}
for un_nombre in nombres:
    un_estudio            = optuna.multi_objective.load_study(study_name=un_nombre, storage=recomendaciones_db)
    trials_de_un_estudio  = un_estudio.get_trials(deepcopy=False) #or pareto trials??
    scores_studios        = scores_studios | {f"{un_nombre}":
        { trial.number: np.mean([x for x in trial.values if x is not None]) 
                for
                    trial in trials_de_un_estudio if trial.state == optuna.trial.TrialState.COMPLETE}
                    } 
#%%

def extract_min_value_keys(input_dict):
    output_dict = {}  # Initialize an empty dictionary to store the result
    # Loop through each item in the input dictionary
    for workforce, values_dict in input_dict.items():
        max_key = min(values_dict, key=values_dict.get)  # Find the key with the maximum value in values_dict
        max_value = values_dict[max_key]  # Get the maximum value
        output_dict[workforce] = (max_key, max_value)  # Add the key and value to the output dictionary
    return output_dict  # Return the output dictionary
    

trials_optimos          = extract_min_value_keys(scores_studios) # Para cada tramo, extrae el maximo, 
planificaciones_optimas = {}   
for k,v in trials_optimos.items():
    un_estudio               = optuna.multi_objective.load_study(study_name=k, storage=recomendaciones_db)
    trials_de_un_estudio     = un_estudio.get_trials(deepcopy=False)
    planificaciones_optimas  = planificaciones_optimas | {f"{k}":
        trial.user_attrs.get('planificacion')#calcular_optimo(trial.values)
                for
                    trial in trials_de_un_estudio if trial.number == v[0]
                    }   

planificacion                =  plan_unico([plan for tramo,plan in planificaciones_optimas.items()])
planificacion
# %%

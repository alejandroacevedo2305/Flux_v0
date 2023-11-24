#%%
import os
#os.chdir('/DeepenData/Repos/Flux_v0/')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import random  
import optuna
from src.datos_utils import DatasetTTP, obtener_skills
from src.optuna_utils import non_empty_subsets
import itertools
import numpy as np
from src.optuna_utils import plan_unico
from dev.atributos_de_series import atributos_x_serie
import optuna
import numpy as np
from datetime import datetime
import time
from src.optuna_utils import (
    sla_x_serie, 
    extract_skills_length, 
    non_empty_subsets
    )
from   itertools                       import (count, islice)
import pandas                          as pd
from datetime import datetime
from typing import List
import math
from   src.utils_Escritoriosv05_Simv05 import (
                                            actualizar_keys_tramo,
                                            separar_por_conexion,
                                            reset_escritorios_OFF,
                                            pasos_alternancia_v03,
                                            DatasetTTP,
                                            generar_planificacion,
                                            reloj_rango_horario,
                                            generate_integer, 
                                            match_emisiones_reloj,
                                            match_emisiones_reloj_historico,
                                            reloj_rango_horario,
                                            remove_selected_row,
                                            FIFO,
                                            balancear_carga_escritorios,
                                            extract_highest_priority_and_earliest_time_row
                                            )
from dev.Escritoriosv05_Simv05 import simv05, Escritoriosv05
from dev.atributos_de_series import atributos_x_serie
import math

# dataset                                 = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz") # IdOficina=2)
# un_dia                                  = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
#%%
######################################
#-----Modo parámetros históricos------
#######################################

def plan_desde_skills(skills, inicio):    
    return  {id: [
                    {'inicio':inicio,
                    'termino':None,
                    'propiedades': {
                        'skills': sks,
                        'configuracion_atencion': random.choice(["FIFO", "Rebalse", "Alternancia"]), #"Rebalse", # "Alternancia", #"Rebalse", #random.choice(["FIFO", "Rebalse", "Alternancia"]) "FIFO",
                        'porcentaje_actividad'  : random.uniform(0.8, .95),
                        'atributos_series':atributos_x_serie(
                            ids_series=sorted(list({val for sublist in skills.values() for val in sublist})), 
                            sla_porcen_user=None, 
                            sla_corte_user=None, 
                            pasos_user=None, 
                            prioridades_user=None),
                    }}
                        ] for id, sks in skills.items()}
# skills                      = obtener_skills(un_dia)

# planificacion               = plan_desde_skills(skills, inicio = '08:00:00')


# hora_cierre           = "16:30:00"
import logging

def simv06(un_dia, hora_cierre, planificacion, log_path:str="dev/simv05.log"):
    

    un_dia.FH_AteIni = None


    reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
    registros_atenciones  = pd.DataFrame()
    matcher_emision_reloj = match_emisiones_reloj(un_dia)
    supervisor            = Escritoriosv05(planificacion = planificacion)
    registros_atenciones  = pd.DataFrame()
    fila                  = pd.DataFrame()

    tiempo_total          = (datetime.strptime(hora_cierre, '%H:%M:%S') - 
                                datetime.strptime(str(un_dia.FH_Emi.min().time()), '%H:%M:%S')).total_seconds() / 60

    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')


    for i , hora_actual in enumerate(reloj):
        total_mins_sim =i
        logging.info(f"--------------------------------NUEVA hora_actual {hora_actual}---------------------")
        supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion, tiempo_total= tiempo_total)   
        matcher_emision_reloj.match(hora_actual)
        
        if not matcher_emision_reloj.match_emisiones.empty:
            logging.info(f"nuevas emisiones")

            emisiones      = matcher_emision_reloj.match_emisiones
            
            logging.info(f"hora_actual: {hora_actual} - series en emisiones: {list(emisiones['IdSerie'])}")
            fila           = pd.concat([fila, emisiones])
        else:
            logging.info(f"no hay nuevas emisiones hora_actual {hora_actual}")   
        
        
        if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
            en_atencion            = supervisor.filtrar_x_estado('atención') or []
            en_pausa               = supervisor.filtrar_x_estado('pausa') or []
            escritorios_bloqueados = set(en_atencion + en_pausa)            
            escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
            logging.info(f"iterar_escritorios_bloqueados: {escritorios_bloqueados_conectados}")        
            supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

        if  supervisor.filtrar_x_estado('disponible'):
            conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in supervisor.filtrar_x_estado('disponible')]
            logging.info(f'iterar_escritorios_disponibles: {conectados_disponibles}')
            logging.info('tiempo_actual_disponible',
            {k: v['tiempo_actual_disponible'] for k,v in supervisor.escritorios_ON.items() if k in conectados_disponibles})        
            supervisor.iterar_escritorios_disponibles(conectados_disponibles)
            

            
            conectados_disponibles = balancear_carga_escritorios(
                                                                {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                    'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                for k,v in supervisor.escritorios_ON.items() if k in conectados_disponibles}
                                                                )        
            
            for un_escritorio in conectados_disponibles:
                logging.info(f"iterando en escritorio {un_escritorio}")
                
                configuracion_atencion = supervisor.escritorios_ON[un_escritorio]['configuracion_atencion']            
                fila_filtrada          = fila[fila['IdSerie'].isin(supervisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])
                
                if fila_filtrada.empty:
                    continue
                elif configuracion_atencion == "FIFO":
                    un_cliente = FIFO(fila_filtrada)
                    logging.info(f"Cliente seleccionado x FIFO {tuple(un_cliente)}")
                elif configuracion_atencion == "Rebalse":
                    un_cliente = extract_highest_priority_and_earliest_time_row(fila_filtrada, supervisor.escritorios_ON[un_escritorio].get('prioridades'))
                    logging.info(f"Cliente seleccionado x Rebalse {tuple(un_cliente)}")
                elif configuracion_atencion == "Alternancia":
                    un_cliente = supervisor.escritorios_ON[un_escritorio]['pasos_alternancia'].buscar_cliente(fila_filtrada)
                    logging.info(f"Cliente seleccionado x Alternancia {tuple(un_cliente)}")
                                        
                fila                 = remove_selected_row(fila, un_cliente)
                logging.info(f"INICIANDO ATENCION de {tuple(un_cliente)}")
                supervisor.iniciar_atencion(un_escritorio, un_cliente)
                logging.info(f"numero_de_atenciones de escritorio {un_escritorio}: {supervisor.escritorios_ON[un_escritorio]['numero_de_atenciones']}")
                logging.info(f"---escritorios disponibles: { supervisor.filtrar_x_estado('disponible')}")
                logging.info(f"---escritorios en atención: { supervisor.filtrar_x_estado('atención')}")
                logging.info(f"---escritorios en pausa: { supervisor.filtrar_x_estado('pausa')}")



                un_cliente.IdEsc     = int(un_escritorio)
                un_cliente.FH_AteIni = hora_actual                               
                registros_atenciones = pd.concat([registros_atenciones, 
                pd.DataFrame(un_cliente).T])                   

        if i == 0:
            fila['espera'] = 0
        else:
            fila['espera'] += 1*60      
    logging.info(f"minutos simulados {total_mins_sim} minutos reales {tiempo_total}")
    return registros_atenciones, fila
    
#%%    
# import time
# start_time = time.time()
# registros_atenciones, fila =  simv06(un_dia, hora_cierre, planificacion)
# print(f"{len(registros_atenciones) = }, {len(fila) = }")
# end_time = time.time()
# print(f"tiempo total: {end_time - start_time:.1f} segundos")  
    
#%%             
# pd.set_option('display.max_rows', None)
# registros_atenciones


#%%
import os
os.chdir('/DeepenData/Repos/Flux_v0/')
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

dataset                                 = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz") # IdOficina=2)
un_dia                                  = dataset.un_dia("2023-05-15").sort_values(by='FH_AteIni', inplace=False)

######################################
#-----Modo parámetros históricos------
#######################################

def plan_desde_skills(skills, porcentaje_actividad, inicio):    
    return  {id: [
                    {'inicio':inicio,
                    'termino':None,
                    'propiedades': {
                        'skills': sks,
                        'configuracion_atencion': None,
                        'porcentaje_actividad'  :  porcentaje_actividad,
                        'atributos_series':atributos_x_serie(
                            ids_series=sorted(list({val for sublist in skills.values() for val in sublist})), 
                            sla_porcen_user=None, 
                            sla_corte_user=None, 
                            pasos_user=None, 
                            prioridades_user=None),
                    }}
                        ] for id, sks in skills.items()}
skills                      = obtener_skills(un_dia)
planificacion               = plan_desde_skills(skills, porcentaje_actividad = .9, inicio = '08:00:00')
planificacion_un_escritorio = plan_desde_skills({'0': list({val for sublist in skills.values() for val in sublist})}, porcentaje_actividad = .9, inicio = '08:00:00')


#%%
hora_cierre           = "8:42:00"
reloj                 = reloj_rango_horario(str(un_dia.FH_AteIni.min().time()), hora_cierre)
registros_atenciones  = pd.DataFrame()
matcher_emision_reloj = match_emisiones_reloj_historico(un_dia)
supervisor            = Escritoriosv05(planificacion = planificacion_un_escritorio)
registros_atenciones  = pd.DataFrame()
fila                  = pd.DataFrame()

tiempo_total          = (datetime.strptime(hora_cierre, '%H:%M:%S') - 
                            datetime.strptime(str(un_dia.FH_AteIni.min().time()), '%H:%M:%S')).total_seconds() / 60

for i , hora_actual in enumerate(reloj):
    total_mins_sim =i
    print(f"--------------------------------NUEVA hora_actual {hora_actual}---------------------")
    supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion_un_escritorio)
    
    if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
        en_atencion            = supervisor.filtrar_x_estado('atención') or []
        en_pausa               = supervisor.filtrar_x_estado('pausa') or []
        escritorios_bloqueados = set(en_atencion + en_pausa)            
        escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
        print(f"iterar_escritorios_bloqueados: {escritorios_bloqueados_conectados}")        
        supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

    if disponibles:= supervisor.filtrar_x_estado('disponible'):
        conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
        print(f'iterar_escritorios_disponibles: {conectados_disponibles}')

        print('tiempo_actual_disponible',
        {k: v['tiempo_actual_disponible'] for k,v in supervisor.escritorios_ON.items() if k in disponibles})
        
        supervisor.iterar_escritorios_disponibles(conectados_disponibles)

    matcher_emision_reloj.match(hora_actual)
    
    if not matcher_emision_reloj.match_emisiones.empty:
        print(f"nuevas emisiones")

        emisiones      = matcher_emision_reloj.match_emisiones
        
        print(f"hora_actual: {hora_actual} - emisiones: {list(emisiones['FH_AteIni'])}")
        fila           = pd.concat([fila, emisiones])
    else:
        print(f"no hay nuevas emisiones hora_actual {hora_actual}")   


    if disponibles:= supervisor.filtrar_x_estado('disponible'):
        print(f"hay escritorios disponibles: {disponibles}")
        
        for un_escritorio in disponibles:
            print(f"iterando en escritorio {un_escritorio}")           

            for _, un_cliente in fila.iterrows():
                print(f"iterando cliente {tuple(un_cliente)}")
                
                if un_cliente.IdSerie in supervisor.escritorios_ON[un_escritorio].get('skills', []) and  supervisor.filtrar_x_estado('disponible'):
                                    
                    fila                 = remove_selected_row(fila, un_cliente)
                    print(f"INICIANDO ATENCION de {tuple(un_cliente)}")
                    supervisor.iniciar_atencion(un_escritorio, un_cliente)


                    registros_atenciones = pd.concat([registros_atenciones, 
                    pd.DataFrame(un_cliente).T])
    else:
        print(f"NO hay escritorios disponibles")      

    if i == 0:
        fila['espera'] = 0
    else:
        fila['espera'] += 1*1
print(f"minutos simulados {total_mins_sim} minutos reales {tiempo_total}")
fila, registros_atenciones


#%%
on_off = True
idEsc  = '7'


if supervisor.escritorios_ON[idEsc]['conexion'] == on_off == True: 
    print("hola")

#self.escritorios_ON[idEsc]['contador_tiempo_disponible'] if {**self.escritorios_ON, **self.escritorios_OFF}[idEsc]['conexion'] == on_off == True else iter(count(start=0, step=1))


#supervisor.propiedades_tramos[1]
#%%


    # if disponibles:= supervisor.filtrar_x_estado('disponible'):      
    #     for i, cliente_seleccionado in fila.iterrows():
    #         print(f"for {i}, cliente_seleccionado in fila.iterrows():")
    #         if cliente_seleccionado.IdEsc in [int(c) for c in conectados_disponibles]:                
    #             idx_escritorio_seleccionado= [int(c) for c in conectados_disponibles].index(cliente_seleccionado.IdEsc)                
                
    #             escritorio_seleccionado = conectados_disponibles[idx_escritorio_seleccionado]                #print(escritorio_seleccionado)
    #             print(f"°°MATCH escritorio_seleccionado: {escritorio_seleccionado} - cliente_seleccionado.IdEsc: {cliente_seleccionado.IdEsc}")
    #             print(f"°°MATCH hora_actual: {hora_actual} - FH_AteIni: {cliente_seleccionado.FH_AteIni}")
    #             assert int(escritorio_seleccionado) == cliente_seleccionado.IdEsc
                
    #             supervisor.iniciar_atencion(escritorio_seleccionado, cliente_seleccionado)
    #             conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in supervisor.filtrar_x_estado('disponible')]
                
    #             fila = remove_selected_row(fila, cliente_seleccionado)
    #             registros_atenciones = pd.concat([registros_atenciones, pd.DataFrame(cliente_seleccionado).T ])
    # else:
    #     print(f"!!!!!!!!!!!!!!!!!!!!!!!todos los escritios ocupados")    
    # fila['espera'] += 1*60
    
    
    
    
#%%             
pd.set_option('display.max_rows', None)
registros_atenciones
#%%
un_dia.sort_values(by='FH_AteIni', inplace=False)[['FH_AteIni',	'IdSerie',	'T_Ate',	'IdEsc', 'T_Esp']].head(101)#, fila, registros_atenciones

#%%

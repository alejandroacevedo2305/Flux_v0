
#%%
import os
os.chdir('/home/trabajo/repos/Flux_v0')
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
                                            reloj_rango_horario,
                                            remove_selected_row,
                                            FIFO,
                                            balancear_carga_escritorios,
                                            extract_highest_priority_and_earliest_time_row
                                            )
class Escritoriosv06:
    def __init__(self,
                 planificacion:dict=None, 
                 historico:bool=False,
                 escritorios_historicos:list=None,
                 ):    
        if historico:

            self.escritorios = escritorios_historicos

        else:

            self.planificacion            = planificacion
            self.escritorios              = {k: {
                                                'inicio' : None, 
                                                'termino' : None, 
                                                'estado':                     'disponible',  #
                                                'tiempo_actual_disponible':   0,  # 
                                                'skills':                     None, #v[0]['propiedades'].get('skills'),  #                                    
                                                'configuracion_atencion':     None, #v[0]['propiedades'].get('configuracion_atencion'),  # 
                                                'contador_tiempo_disponible': None,  # 
                                                'numero_de_atenciones':       0,  #                        
                                                'porcentaje_actividad':      None,
                                                'duracion_inactividad':      None,                                    
                                                'contador_inactividad':      None,                                    
                                                'duracion_pausas':            (1, 5, 15),  # --- pausas ---
                                                'probabilidad_pausas':        .5,          # --- pausas ---
                                                'numero_pausas':              None,        # --- pausas ---
                                                'prioridades':                None,                                                                        
                                                'pasos':                      None,               
                                                'conexion':                   True,
                                                'pasos_alternancia': None,                                       
                                                }
                                                for k, v in planificacion.items()}  #        
            
            self.escritorios_OFF                    = self.escritorios
            self.escritorios_ON                     = {}
            self.propiedades_tramos                 = []

supervisor = Escritoriosv06(historico=True) 
supervisor.escritorios
# %%

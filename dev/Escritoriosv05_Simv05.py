#%%
from   itertools                       import (count, islice)
import pandas                          as pd
from   src.utils_Escritoriosv05_Simv05 import (
                                            actualizar_keys_tramo,
                                            separar_por_conexion,
                                            reset_escritorios_OFF,
                                            poner_pasos_alternancia_v02,
                                            pasos_alternancia_v02)

class Escritoriosv05:
    
    def __init__(self,
                 inicio_tramo:  pd.Timestamp, 
                 fin_tramo:     pd.Timestamp,
                 planificacion: dict, 
                 niveles_servicio_x_serie:dict,
                 ):
      
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        self.planificacion            = planificacion
        self.escritorios              = {k: {  
                                            'estado':                     'disponible',  #
                                            'tiempo_actual_disponible':   0,  # 
                                            'skills':                     v[0]['propiedades'].get('skills'),  #                                    
                                            'configuracion_atencion':     v[0]['propiedades'].get('configuracion_atencion'),  # 
                                            'contador_tiempo_disponible': iter(count(start=0, step=1)),  # 
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':       v[0]['propiedades'].get('porcentaje_actividad'),
                                            'duracion_inactividad':       int(
                                                                            (1 - v[0]['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            (fin_tramo - inicio_tramo).total_seconds() / 60
                                                                            ) if v[0]['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                            'contador_inactividad':       iter(islice(
                                                                                count(start=0, step=1),
                                                                                int((1 - v[0]['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                                    (fin_tramo - inicio_tramo).total_seconds() / 60)
                                                                                )) if v[0]['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                            'duracion_pausas':            (1, 4, 30),  # --- pausas ---
                                            'probabilidad_pausas':        .5,          # --- pausas ---
                                            'numero_pausas':              None,        # --- pausas ---
                                            'atributos_series':           v[0]['propiedades'].get('atributos_series')
                                                                        if v[0]['propiedades'].get('atributos_series') is not None else None,
                                            'prioridades':                {dict_series['serie']: dict_series['prioridad'] for dict_series in 
                                                                           v[0]['propiedades'].get('atributos_series')}
                                                                        if v[0]['propiedades'].get('atributos_series') is not None else None,
                                            'conexion':                   True,                                       
                                            }
                                            for k, v in planificacion.items()}  # 
        
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.propiedades_tramos                 = []        
        
    def aplicar_planificacion(self, hora_actual, planificacion):
        
        propiedades_tramo = dict()
        for idEsc, un_escritorio in planificacion.items():
            for un_tramo in un_escritorio:
                on_off = hora_actual >= un_tramo['inicio'] and (lambda: 
                    hora_actual <= un_tramo['termino'] if un_tramo['termino'] is not None else True)()
                #print(f"{idEsc}: {on} {hora_actual} >= {un_tramo['inicio']} and {hora_actual} <= {un_tramo['termino']}")           
                propiedades_tramo = propiedades_tramo | {idEsc: {
                                                'conexion': on_off, #se setean conexiones
                                                'skills':                 un_tramo['propiedades']['skills'],
                                                'configuracion_atencion': un_tramo['propiedades']['configuracion_atencion'],
                                                'atributos_series':       un_tramo['propiedades']['atributos_series'],
                                                #'duracion_pausas':        un_tramo['propiedades']['duracion_pausas'],
                                                #'probabilidad_pausas':    un_tramo['propiedades']['probabilidad_pausas'],
                                                'porcentaje_actividad':   un_tramo['propiedades']['porcentaje_actividad'],                                                 
                                                }}
                if on_off:
                    break       
        self.propiedades_tramos.append(propiedades_tramo)        
        actualizar_keys_tramo(self.escritorios_ON, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.
        actualizar_keys_tramo(self.escritorios_OFF, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.

        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion({**self.escritorios_ON, **self.escritorios_OFF})
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)  #En los escritorios desconectados, resetear los iteradores que cuentan el tiempo de bloqueo y  poner los escritorios en estados disponible, así quedan listo para volver a conectarse
        self.escritorios_ON                       = poner_pasos_alternancia_v02(self.escritorios_ON, pasos_alternancia_v02)
#%%

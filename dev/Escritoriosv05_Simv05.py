#%%
from   itertools                       import (count, islice)
import pandas                          as pd
from datetime import datetime

from   src.utils_Escritoriosv05_Simv05 import (
                                            actualizar_keys_tramo,
                                            separar_por_conexion,
                                            reset_escritorios_OFF,
                                            poner_pasos_alternancia_v02,
                                            pasos_alternancia_v02,
                                            DatasetTTP,
                                            generar_planificacion,
                                            reloj_rango_horario, 
                                            match_emisiones_reloj)

class Escritoriosv05:
    
    def __init__(self,
                 inicio_tramo:  pd.Timestamp, 
                 fin_tramo:     pd.Timestamp,
                 planificacion: dict, 
                 niveles_servicio_x_serie:dict,
                 ):
      
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        self.planificacion            = planificacion
        # los escritorios se inicializan con los atributos del primer tramo
        self.escritorios              = {k: {
                                            'inicio' : None, 
                                            'termino' : None, 
                                            'estado':                     'disponible',  #
                                            'tiempo_actual_disponible':   0,  # 
                                            'skills':                     None, #v[0]['propiedades'].get('skills'),  #                                    
                                            'configuracion_atencion':     None, #v[0]['propiedades'].get('configuracion_atencion'),  # 
                                            'contador_tiempo_disponible': iter(count(start=0, step=1)),  # 
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':      None,
                                            'duracion_inactividad':      None,                                    
                                            'contador_inactividad':      None,                                    
                                            'duracion_pausas':            (1, 4, 30),  # --- pausas ---
                                            'probabilidad_pausas':        .5,          # --- pausas ---
                                            'numero_pausas':              None,        # --- pausas ---
                                            'prioridades':                None,                                                                        
                                            'pasos':                      None,               
                                            'conexion':                   True,                                       
                                            }
                                            for k, v in planificacion.items()}  # 
        
        #chequear construcción con el primer escritorio cargado
        # common_keys =  ['skills', 'configuracion_atencion', 'porcentaje_actividad','atributos_series']
        # assert [next(iter(self.escritorios.items()))[1] [k] for k in common_keys] == [
        #         next(iter(planificacion.items()))[1][0]['propiedades'][k] for k in common_keys]                
        # assert {p['serie']: p['prioridad'] for p in next(iter(planificacion.items()))[1][0]['propiedades']['atributos_series']} == {
        #         p['serie']: p['prioridad'] for p in   next(iter(self.escritorios.items()))[1]['atributos_series']}
        # assert {p['serie']: p['pasos'] for p in   next(iter(self.escritorios.items()))[1]['atributos_series']} == {
        #         p['serie']: p['pasos'] for p in next(iter(planificacion.items()))[1][0]['propiedades']['atributos_series']}
        
        
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.propiedades_tramos                 = []        
        
    def aplicar_planificacion(self, hora_actual, planificacion):
        
        propiedades_tramo = dict()
        for idEsc, un_escritorio in planificacion.items():
            for un_tramo in un_escritorio:
                on_off = hora_actual >= un_tramo['inicio'] and (lambda: 
                         hora_actual <= un_tramo['termino'] if un_tramo['termino'] is not None else True)()
                propiedades_tramo = propiedades_tramo | {idEsc: {
                                                'inicio'  : un_tramo['inicio'], 
                                                'termino' : un_tramo['termino'], 
                                                'conexion': on_off, #se setean conexiones
                                                'skills':                 un_tramo['propiedades']['skills'],
                                                'configuracion_atencion': un_tramo['propiedades']['configuracion_atencion'],
                                                'porcentaje_actividad':   un_tramo['propiedades']['porcentaje_actividad'],
                                                'prioridades':            {dict_series['serie']: dict_series['prioridad'] for dict_series in 
                                                                           un_tramo['propiedades'].get('atributos_series')},
                                                'pasos':                  {dict_series['serie']: dict_series['pasos'] for dict_series in 
                                                                           un_tramo['propiedades'].get('atributos_series')},
                                                
                                                
                                                
                                                
                                                
                                            'contador_tiempo_disponible': iter(count(start=0, step=1)),  # 
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':       un_tramo['propiedades'].get('porcentaje_actividad'),
                                            'duracion_inactividad':       int(
                                                                            (1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60)
                                                                            ) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                            'contador_inactividad':       iter(islice(
                                                                                count(start=0, step=1),
                                                                                int((1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                                   ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60))
                                                                                )) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,        
                                                
                                                                                                                                                 
                                                }}
                if on_off:
                    break       
        self.propiedades_tramos.append(propiedades_tramo)        
        actualizar_keys_tramo(self.escritorios_ON, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.
        actualizar_keys_tramo(self.escritorios_OFF, propiedades_tramo)  #se actualizan las propiedades del tramo.TO-DO: FALTAN LOS PASOS Y PRIORIDADES.

        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion({**self.escritorios_ON, **self.escritorios_OFF})
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)  #En los escritorios desconectados, resetear los iteradores que cuentan el tiempo de bloqueo y  poner los escritorios en estados disponible, así quedan listo para volver a conectarse
        #self.escritorios_ON                       = poner_pasos_alternancia_v02(self.escritorios_ON, pasos_alternancia_v02)

dataset                                 = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia                                  = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
planificacion, niveles_servicio_x_serie = generar_planificacion(un_dia)
# %%

supervisor = Escritoriosv05(inicio_tramo             = un_dia['FH_Emi'].min(),
                            fin_tramo                = un_dia['FH_Emi'].max(),
                            planificacion            = planificacion,
                            niveles_servicio_x_serie = niveles_servicio_x_serie)

hora_cierre           = "16:00:00"
reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)

print(supervisor.escritorios_ON)


hora_actual           =  next(reloj)
supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion)


supervisor.escritorios_ON
#%%
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
class Escritoriosv05:    
    def __init__(self,
                 planificacion: dict, 
                 ):      
        self.planificacion            = planificacion
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
                                            'duracion_pausas':            None, #(1, 5, 15),  # --- pausas ---
                                            'probabilidad_pausas':        None,          # --- pausas ---
                                            'numero_pausas':              None,        # --- pausas ---
                                            'prioridades':                None,                                                                        
                                            'pasos':                      None,               
                                            'conexion':                   False,
                                            'pasos_alternancia': None,                                       
                                            }
                                            for k, v in planificacion.items()}  #        
        
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.propiedades_tramos                 = []        
        
    def aplicar_planificacion(self, hora_actual, planificacion, tiempo_total):
        
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
                                            'numero_de_atenciones':       0,  #                        
                                            'porcentaje_actividad':       un_tramo['propiedades'].get('porcentaje_actividad'),
                                            'duracion_inactividad':       int(
                                                                            (1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60)
                                                                            ) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,                                    
                                            'contador_inactividad':  # None,    
                                                                        iter(islice(
                                                                        count(start=0, step=1),
                                                                        int((1 - un_tramo['propiedades'].get('porcentaje_actividad', 0)) * 
                                                                            ((datetime.strptime('13:00:00', '%H:%M:%S')-datetime.strptime('12:00:00', '%H:%M:%S')).total_seconds()/60))
                                                                        )) if un_tramo['propiedades'].get('porcentaje_actividad') is not None else None,
                                                                        
                                            'duracion_pausas': (lambda x: (int(x / 2), int(x), int(2 * x)))(
                                                               ((1-un_tramo['propiedades'].get('porcentaje_actividad')
                                                                )*tiempo_total)/ (tiempo_total/30)),
                                            
                                            
                                            
                                            'probabilidad_pausas': 0 if un_tramo['propiedades'].get('porcentaje_actividad')==1 else 0.6,
                                            #(1, 5, 15),
                        #                     'contador_tiempo_disponible':  
                        # self.escritorios_ON[idEsc]['contador_tiempo_disponible'] if {**self.escritorios_ON, **self.escritorios_OFF}[idEsc]['conexion'] == on_off == True else iter(count(start=0, step=1)), 
                        
                                            'pasos_alternancia': pasos_alternancia_v03(prioridades = 
                                                                                                   {dict_series['serie']: dict_series['prioridad'] for dict_series in 
                                                                                                   un_tramo['propiedades'].get('atributos_series')},
                                pasos = 
                                            {dict_series['serie']: dict_series['pasos'] for dict_series in 
                                                un_tramo['propiedades'].get('atributos_series')})
                                                if un_tramo['propiedades']['configuracion_atencion'] == 'Alternancia' else None, 
                                            }}
                if on_off:
                    break   
                
                
        self.propiedades_tramos.append(propiedades_tramo)        
        actualizar_keys_tramo(self.escritorios_ON, propiedades_tramo)  #
        actualizar_keys_tramo(self.escritorios_OFF, propiedades_tramo)  #        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion({**self.escritorios_ON, **self.escritorios_OFF})
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)  #
        
    def iniciar_atencion(self, escritorio, cliente_seleccionado):

        minutos_atencion =  int(math.floor(cliente_seleccionado.T_Ate/60))#round((cliente_seleccionado.T_Ate - cliente_seleccionado.FH_AteIni).total_seconds()/60)
            
        self.escritorios_ON[escritorio]['contador_tiempo_atencion'] = iter(islice(count(start=0, step=1), minutos_atencion))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']           = 'atención'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_atencion']  = minutos_atencion#tiempo de atención     
        self.escritorios[escritorio]['numero_de_atenciones'] += 1 #se guarda en self.escritorios para que no se resetee.
        self.escritorios_ON[escritorio]['numero_de_atenciones'] = self.escritorios[escritorio]['numero_de_atenciones'] 
        
        print(f"el escritorio {escritorio} inició atención x {minutos_atencion} minutos ({cliente_seleccionado.T_Ate} segundos)")
        


    def filtrar_x_estado(self, state: str):     
        #obtener estados
        self.estados = {escr_i: {'estado': propiedades['estado'], 'configuracion_atencion': 
                        propiedades['configuracion_atencion']} for escr_i, propiedades in self.escritorios_ON.items()} 
        #extraer por disponibilidad    
        if disponibilidad := [
            key for key, value in self.estados.items() if value['estado'] == state
        ]:
            return disponibilidad
        else:
            #print(f"No hay escritorio {state}")
            return False
    def iniciar_pausa(self, escritorio, tipo_inactividad:str = "Pausas", generador_pausa = generate_integer):
        # sourcery skip: extract-method, move-assign
      
        if tipo_inactividad == "Porcentaje":            
            self.escritorios_ON[escritorio]['estado'] = 'pausa'            
        else:
            min_val, avg_val, max_val = self.escritorios_ON[escritorio]['duracion_pausas']
            probabilidad_pausas       = self.escritorios_ON[escritorio]['probabilidad_pausas']
            minutos_pausa             = generador_pausa(min_val, avg_val, max_val, probabilidad_pausas)

            self.escritorios_ON[escritorio]['contador_tiempo_pausa'] = iter(islice(count(start=0, step=1), minutos_pausa)) #if minutos_pausa > 0 else None
            self.escritorios_ON[escritorio]['estado']                = 'pausa'#estado
            self.escritorios_ON[escritorio]['minutos_pausa']         = minutos_pausa#tiempo 
            
            
        print(f"el escritorio {escritorio} inició pausa x {minutos_pausa} minutos")

            
    def iniciar_tiempo_disponible(self,escritorio):
        self.escritorios_ON[escritorio]['contador_tiempo_disponible'] = iter(count(start=0, step=1))
        self.escritorios_ON[escritorio]['estado']                     = 'disponible'#     

        print(f"**el escritorio {escritorio} quedó **disponible**")

        
    def iterar_escritorios_bloqueados(self, escritorios_bloqueados: List[str], tipo_inactividad:str = "Pausas"):

        for escri_bloq in escritorios_bloqueados:
            #ver si está en atención:
            
            
            if self.escritorios_ON[escri_bloq]['estado'] == 'pausa': 
                        tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
                        if tiempo_pausa is None: 
                            #si termina tiempo en pausa pasa a estado disponible
                            self.iniciar_tiempo_disponible(escri_bloq)
                            
                        else:
                            print(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_pausa'] - tiempo_pausa} min de pausa")      
                                     
            if self.escritorios_ON[escri_bloq]['estado'] == 'atención':                
                #avanzamos en un minuto el tiempo de atención
                tiempo_atencion = next(self.escritorios_ON[escri_bloq]['contador_tiempo_atencion'], None)
                 
                #si terminó la atención
                if tiempo_atencion is None: 
                    #iniciar pausa 
                    self.iniciar_pausa(escri_bloq, tipo_inactividad)
                  #chequeamos si la inactividad es por pocentaje o pausas históricas
                    if tipo_inactividad == "Porcentaje":
                    #Avanzamos el contador de inactividad en un minuto
                        tiempo_inactividad = next(self.escritorios_ON[escri_bloq]['contador_inactividad'],None)
                    #si termina el tiempo de inactividad
                        if tiempo_inactividad is None:
                            #pasa a estado disponible
                            self.iniciar_tiempo_disponible(escri_bloq)
                    else: #pausas históricas                 
                        #iteramos contador_tiempo_pausa:
                        tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
                        if tiempo_pausa is None: 
                            #si termina tiempo en pausa pasa a estado disponible
                            self.iniciar_tiempo_disponible(escri_bloq)
                            
                        else:
                            print(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_pausa'] - tiempo_pausa} min de pausa")    
                                        
                    
                else:
                    tiempo_atencion += 1
                    print(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_atencion'] - tiempo_atencion} min de atención") 
                    
           


            # #si el escritorio está en pausa:            
            # elif self.escritorios_ON[escri_bloq]['estado'] == 'pausa':
            #       #chequeamos si la inactividad es por pocentaje o pausas históricas
            #      if tipo_inactividad == "Porcentaje":
            #        #Avanzamos el contador de inactividad en un minuto
            #        tiempo_inactividad = next(self.escritorios_ON[escri_bloq]['contador_inactividad'],None)
            #        #si termina el tiempo de inactividad
            #        if tiempo_inactividad is None:
            #          #pasa a estado disponible
            #          self.iniciar_tiempo_disponible(escri_bloq)
            #      else: #pausas históricas                 
            #         #iteramos contador_tiempo_pausa:
            #         tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
            #         if tiempo_pausa is None: 
            #             #si termina tiempo en pausa pasa a estado disponible
            #             self.iniciar_tiempo_disponible(escri_bloq)
                        
            #         else:
            #             print(f"al escritorio {escri_bloq} le quedan {self.escritorios_ON[escri_bloq]['minutos_pausa'] - tiempo_pausa} min de pausa") 

                        
    def iterar_escritorios_disponibles(self, escritorios_disponibles: List[str]):
        
        for escri_dispon in escritorios_disponibles:               
            #avanzamos en un minuto el tiempo que lleva disponible.
            tiempo_disponible = next(self.escritorios_ON[escri_dispon]['contador_tiempo_disponible'])
            self.escritorios_ON[escri_dispon]['tiempo_actual_disponible'] = tiempo_disponible +1
        
def simv05(un_dia, hora_cierre, planificacion):

    reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
    registros_atenciones  = pd.DataFrame()
    matcher_emision_reloj = match_emisiones_reloj(un_dia)
    supervisor            = Escritoriosv05(planificacion = planificacion)
    registros_atenciones  = pd.DataFrame()
    fila                  = pd.DataFrame()
    #i=0
    for hora_actual in reloj:
        supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion)
        if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
            en_atencion            = supervisor.filtrar_x_estado('atención') or []
            en_pausa               = supervisor.filtrar_x_estado('pausa') or []
            escritorios_bloqueados = set(en_atencion + en_pausa)            
            escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
            #print("iterar_escritorios_bloqueados")        
            supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

        if disponibles:= supervisor.filtrar_x_estado('disponible'):
            conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
            #print('iterar_escritorios_disponibles')
            supervisor.iterar_escritorios_disponibles(conectados_disponibles)

        matcher_emision_reloj.match(hora_actual)
        
        if not matcher_emision_reloj.match_emisiones.empty:
            emisiones      = matcher_emision_reloj.match_emisiones
            fila           = pd.concat([fila, emisiones])    
        #if not fila.empty:   
        if disponibles:= supervisor.filtrar_x_estado('disponible'):
            conectados_disponibles       = balancear_carga_escritorios(
                                                                        {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                            'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                        for k,v in supervisor.escritorios_ON.items() if k in disponibles}
                                                                        )
            for un_escritorio in conectados_disponibles:
                configuracion_atencion = supervisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                fila_filtrada          = fila[fila['IdSerie'].isin(supervisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])
                if  fila_filtrada.empty:
                        continue
                elif configuracion_atencion == "FIFO":
                    cliente_seleccionado = FIFO(fila_filtrada)
                    
                elif configuracion_atencion == "Rebalse":
                    cliente_seleccionado = extract_highest_priority_and_earliest_time_row(fila_filtrada, supervisor.escritorios_ON[un_escritorio].get('prioridades'))

                elif configuracion_atencion == "Alternancia":
                    
                    cliente_seleccionado = supervisor.escritorios_ON[un_escritorio]['pasos_alternancia'].buscar_cliente(fila_filtrada)
                
                cliente_seleccionado['IdEsc'] = int(un_escritorio)
                fila = remove_selected_row(fila, cliente_seleccionado)                  
                supervisor.iniciar_atencion(un_escritorio, cliente_seleccionado)            
                registros_atenciones = pd.concat([registros_atenciones, pd.DataFrame(cliente_seleccionado).T ])
                    

        fila['espera'] += 1*60
        #i+=1
    return registros_atenciones, fila




#%%
# dataset               = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
# un_dia                = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
# planificacion, _      = generar_planificacion(un_dia, modos = ['FIFO', 'Rebalse','Alternancia'])
# hora_cierre           = "18:30:00"
# registros_atenciones, fila =  simv05(un_dia, hora_cierre, planificacion)
# len(fila)

#%%



#registros_atenciones, fila =  simv05(un_dia, hora_cierre, planificacion, niveles_servicio_x_serie)


# supervisor = Escritoriosv05(inicio_tramo             = un_dia['FH_Emi'].min(),
#                             fin_tramo                = un_dia['FH_Emi'].max(),
#                             planificacion            = planificacion,
#                             niveles_servicio_x_serie = niveles_servicio_x_serie)
# hora_cierre           = "16:00:00"
# reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
# hora_actual = "10:38:09"
# supervisor.aplicar_planificacion(hora_actual= hora_actual, planificacion = planificacion)

# print(f"hora_actual {hora_actual}")
# print(f"ON  {[(k, v['inicio'],  v['termino'],  v['conexion'],  v['prioridades'],  v['pasos'] ,  v['pasos_alternancia'],  v['configuracion_atencion']  ) for k,v in supervisor.escritorios_ON.items()]}")
# print(f"ON  {[(k,  v['configuracion_atencion'] , v['pasos_alternancia'] ) for k,v in supervisor.escritorios_ON.items()]}")


# print(f"OFF  {[(k, v['inicio'],  v['termino'],  v['conexion'],  v['prioridades'],  v['pasos'],  v['pasos_alternancia'],  v['configuracion_atencion'] ) for k,v in supervisor.escritorios_OFF.items()]}")
# print(f"OFF  {[(k,  v['configuracion_atencion'] ,v['pasos_alternancia'] ) for k,v in supervisor.escritorios_OFF.items()]}")

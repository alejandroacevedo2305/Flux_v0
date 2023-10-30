#%%
from typing import Dict, List, Tuple
import random
from copy import deepcopy
from itertools import count, islice
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import pandas as pd
  
class MisEscritorios_v02:
    
    def __init__(self,
                 inicio_tramo:  pd.Timestamp, 
                 fin_tramo:     pd.Timestamp,
                 planificacion: dict, 
                 conexiones:    dict = None,
                 niveles_servicio_x_serie=None,
                 ):
      
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        self.planificacion = planificacion
        self.escritorios   = {k:{
                                'estado':'disponible',
                                'tiempo_actual_disponible':   0, 
                                "skills": v[0]['propiedades']['skills'],
                                'configuracion_atencion' : v[0]['propiedades']['configuracion_atencion'],
                                'contador_tiempo_disponible': iter(count(start=0, step=1)),
                                'numero_de_atenciones':0,
                                'porcentaje_actividad': v[0]['propiedades']['porcentaje_actividad'],
                                'duracion_inactividad':int(
                                (1- v[0]['propiedades']['porcentaje_actividad'])*(fin_tramo - inicio_tramo).total_seconds()/60),
                                
                                'contador_inactividad': iter(islice(count(start=0, step=1), 
                                                                    int(
                                (1- v[0]['propiedades']['porcentaje_actividad'])*(fin_tramo - inicio_tramo).total_seconds()/60)
                                                                    )),
                                'duracion_pausas': (1, 4, 47), #min, avg, max (desde pausas históricas).
                                'probabilidad_pausas':.5, #probabilidad que la pausa ocurra  (desde pausas históricas).
                                'numero_pausas':       None,
                                } 
                              for k,v in self.planificacion.items()}
        if not conexiones:
        #     #si no se provee el estado de los conexiones se asumen todas como True (todos conectados):
             conexiones                         = {f"{key}": random.choices([True, False], [1, 0])[0] for key in self.escritorios}
        self.escritorios                        = actualizar_conexiones(self.escritorios, conexiones)       
        self.escritorios_OFF                    = self.escritorios
        self.escritorios_ON                     = {}
        self.nuevos_escritorios_programados     = []
        self.registros_escritorios              = []
    def iniciar_atencion(self, escritorio, cliente_seleccionado):
        # iterar los escritorios y emisiones
        #for escr_bloq, emi in zip(escritorios_a_bloqueo, emision):
            
            #extraer los minutos que dura la atención asociada a la emision
        minutos_atencion = round((cliente_seleccionado.FH_AteFin - cliente_seleccionado.FH_AteIni).total_seconds()/60)
            #reescribir campos:
            
        self.escritorios_ON[escritorio]['contador_tiempo_atencion'] = iter(islice(count(start=0, step=1), minutos_atencion))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']           = 'atención'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_atencion']  = minutos_atencion#tiempo de atención     
        self.escritorios[escritorio]['numero_de_atenciones'] += 1 #se guarda en self.escritorios para que no se resetee.
        self.escritorios_ON[escritorio]['numero_de_atenciones'] = self.escritorios[escritorio]['numero_de_atenciones'] 
    def filtrar_x_estado(self, state: str):   
        """
        extrae los escritorios por el estado (disponible o bloqueado)
        """     
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
    def cambiar_propiedades_escritorio(self, 
                                       escritorio:str, 
                                       skills:List[int]=None, 
                                       configuracion_atencion:str=None, 
                                       conexion:bool=None, 
                                       duracion_pausas: tuple = None, # min_val:int=None, avg_val:int=None, max_val:int=None, 
                                       probabilidad_pausas:float=None,
                                       porcentaje_actividad=None) -> None:
        """_summary_
        Modifica las propiedades del escritorio. Si una propiedad entra vacía se ignora. 
        Args:
            escritorio (str): key del escritorio a modificar.
            skills (List[int], optional): Nueva lista de series para cargar como skills. Defaults to None.
            configuracion_atencion (str, optional): Nueva configuracion de atención. Defaults to None.
            conexion (bool, optional): Nuevo estado de conexion. Defaults to None.
        """

        campos = {
                  'skills': skills,
                  'configuracion_atencion': configuracion_atencion,
                  'conexion': conexion,
                  'duracion_pausas': duracion_pausas, #(min_val, avg_val, max_val),
                  'probabilidad_pausas': probabilidad_pausas,
                  'porcentaje_actividad':porcentaje_actividad,
                  }
        #remover propiedades del escritorio que no se modifican
        campos = {k: v for k, v in campos.items() if v is not None}
        #actualizar escritorio
        update_escritorio(escritorio, campos, self.escritorios_OFF, self.escritorios_ON)
        #print(f"{campos} de {escritorio} fue modificado.")
    def actualizar_conexiones_y_propiedades(self, un_escritorio, tramo, accion):
        
        
        propiedades = tramo['propiedades'] | {'conexion': accion == 'iniciar'}
    
        self.cambiar_propiedades_escritorio(un_escritorio, **propiedades)
        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion(
            {**self.escritorios_ON, **self.escritorios_OFF}
        )
        
        self.escritorios_ON  = poner_pasos_alternancia(self.escritorios_ON, pasos_alternancia, self.niveles_servicio_x_serie)
    def aplicar_agenda(self, hora_actual, agenda):
        
        for idEsc, tramos_un_escritorio in agenda.items():
            
            if tramo_idx_tramo := terminar_un_tramo(hora_actual, tramos_un_escritorio):
                tramo     = tramo_idx_tramo[0]
                idx_tramo = tramo_idx_tramo[1]
                #print(f"{idEsc} termina tramo (eliminado de agenda): {tramo}")
                self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'terminar')
                del agenda[idEsc][idx_tramo]   
            
            if tramo:=  iniciar_un_tramo(hora_actual, tramos_un_escritorio):
                #se va seguir ejecutando mientras el tramo sea válido
                #poner alguna flag para q no se vuelva a ejecutar
                #print(f"{idEsc} inicia tramo: {tramo}")
                self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'iniciar')
    def iniciar_pausa(self, escritorio, tipo_inactividad:str = "Porcentaje", generador_pausa = generate_integer):
        # sourcery skip: extract-method, move-assign
      
        if tipo_inactividad == "Porcentaje":            
            self.escritorios_ON[escritorio]['estado'] = 'pausa'            
        else:
            min_val, avg_val, max_val = self.escritorios_ON[escritorio]['duracion_pausas']
            probabilidad_pausas       = self.escritorios_ON[escritorio]['probabilidad_pausas']
            minutos_pausa             = generador_pausa(min_val, avg_val, max_val, probabilidad_pausas)

            self.escritorios_ON[escritorio]['contador_tiempo_pausa'] = iter(islice(count(start=0, step=1), minutos_pausa))#nuevo contador de minutos limitado por n_minutos
            self.escritorios_ON[escritorio]['estado']                = 'pausa'#estado
            self.escritorios_ON[escritorio]['minutos_pausa']         = minutos_pausa#tiempo 
    def iniciar_tiempo_disponible(self,escritorio):
        self.escritorios_ON[escritorio]['contador_tiempo_disponible'] = iter(count(start=0, step=1))
        self.escritorios_ON[escritorio]['estado']                     = 'disponible'#     
    def iterar_escritorios_bloqueados(self, escritorios_bloqueados: List[str], tipo_inactividad:str = "Porcentaje"):

        for escri_bloq in escritorios_bloqueados:
            #ver si está en atención:
            if self.escritorios_ON[escri_bloq]['estado'] == 'atención':                
                #avanzamos en un minuto el tiempo de atención
                tiempo_atencion = next(self.escritorios_ON[escri_bloq]['contador_tiempo_atencion'], None)
                #si terminó la atención
                if tiempo_atencion is None: 
                    #iniciar pausa 
                    self.iniciar_pausa(escri_bloq, tipo_inactividad)
            #si el escritorio está en pausa:            
            elif self.escritorios_ON[escri_bloq]['estado'] == 'pausa':
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
    def iterar_escritorios_disponibles(self, escritorios_disponibles: List[str]):
        
        for escri_dispon in escritorios_disponibles:               
            #avanzamos en un minuto el tiempo que lleva disponible.
            tiempo_disponible = next(self.escritorios_ON[escri_dispon]['contador_tiempo_disponible'], None)
            if tiempo_disponible is not None:
            #guardar el tiempo que lleva disponible
                self.escritorios_ON[escri_dispon]['tiempo_actual_disponible'] = tiempo_disponible


def optuna_simular_v02(agenda_INPUT, niveles_servicio_x_serie, un_dia, prioridades, tipo_inactividad = "Porcentaje"):
  
  planificacion = copy.deepcopy(agenda_INPUT)  
  supervisor    = MisEscritorios_v02(inicio_tramo            = un_dia['FH_Emi'].min(),
                                    fin_tramo                = un_dia['FH_Emi'].max(),
                                    planificacion            = planificacion,
                                    niveles_servicio_x_serie = niveles_servicio_x_serie)
  tiempo_inicial         = list(generador_emisiones(un_dia))[0].FH_Emi#[0]['FH_Emi']
  generador_emisiones_in = generador_emisiones(un_dia)
  contador_tiempo        = timestamp_iterator(tiempo_inicial)
  reloj_simulacion       = next(contador_tiempo)
  fila                   = pd.DataFrame()
  registros_atenciones   = pd.DataFrame()
  SLA_df                 = pd.DataFrame()
  SLA_index              = 0
  Espera_index           = 0 
  Espera_df              = pd.DataFrame() 
  una_emision            = next(generador_emisiones_in)
  emi                    = una_emision['FH_Emi']
  num_emisiones          = un_dia.shape[0]
  
  for _ in range(2*num_emisiones):    
    supervisor.aplicar_agenda(hora_actual=  reloj_simulacion, agenda = planificacion)
    if not mismo_minuto(emi, reloj_simulacion):
      
      reloj_simulacion  = next(contador_tiempo)
      #flag seteada avanzar
      #print(f"avanza el reloj un minuto, nuevo tiempo: {reloj_simulacion}, avanza tiempo de espera y tiempo en escritorios bloqueados (atencion y pausa)")
      #print("tiempos de espera incrementados en un minuto")
      fila['espera'] += 1
      if (supervisor.filtrar_x_estado('atención') or  supervisor.filtrar_x_estado('pausa')):
          en_atencion            = supervisor.filtrar_x_estado('atención') or []
          en_pausa               = supervisor.filtrar_x_estado('pausa') or []
          escritorios_bloqueados = set(en_atencion + en_pausa)            
          #print(f"escritorios ocupados (bloqueados) por servicio: {escritorios_bloqueados}")
          #Avanzar un minuto en todos los tiempos de atención en todos los escritorios bloquedos  
          escritorios_bloqueados_conectados    = [k for k,v in supervisor.escritorios_ON.items() if k in escritorios_bloqueados]
                  
          supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados, tipo_inactividad = tipo_inactividad )

      if disponibles:= supervisor.filtrar_x_estado('disponible'):
          conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
          supervisor.iterar_escritorios_disponibles(conectados_disponibles)
    else:
          #print(
          #f"emisioń dentro del mismo minuto (diferencia {abs(emi-reloj_simulacion).total_seconds()} seg.), actualizamos fila, gestionamos escritorios y pasamos a la siguiente emisión")    
          #poner la emisión en una fila de df
          emision_cliente = pd.DataFrame(una_emision).T
          #insertar una nueva columna de tiempo de espera y asignar con cero.
          emision_cliente['espera'] = 0
          #concatenar la emisión a la fila de espera
          fila = pd.concat([fila, emision_cliente]).reset_index(drop=True)
          #fila_para_SLA = copy.deepcopy(fila[['FH_Emi', 'IdSerie', 'espera']])
          #print(f"fila actualizada: \n{fila[['FH_Emi','IdSerie','espera']]}")

          if not fila.empty:
              if disponibles:= supervisor.filtrar_x_estado('disponible'):
                  #extraer las skills de los escritorios conectados que están disponibles
                  #conectados_disponibles       = [k for k,v in supervisor.escritorios_ON.items() if k in disponibles]
                  conectados_disponibles       = balancear_carga_escritorios(
                                                                              {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                                  'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                              for k,v in supervisor.escritorios_ON.items() if k in disponibles}
                                                                              )    

                  skills_disponibles           = {k:v['skills'] for k,v in supervisor.escritorios_ON.items() if k in disponibles}
                  configuraciones_disponibles  = {k:v['configuracion_atencion'] for k,v in supervisor.escritorios_ON.items() if k in disponibles}

                  for un_escritorio in conectados_disponibles:

                      configuracion_atencion = supervisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                      #print(f"buscando cliente para {un_escritorio} con {configuracion_atencion}")
                      fila_filtrada          = fila[fila['IdSerie'].isin(supervisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])
                      #print(f"en base a las skills: {supervisor.escritorios_ON[un_escritorio].get('skills', [])}, fila_filtrada \n{fila_filtrada}")
                      if  fila_filtrada.empty:
                              #print("No hay match entre idSeries en fila y skills del escritorio, saltar al siguiente escritorio")
                              continue #
                      elif configuracion_atencion == "Alternancia":
                          #print("----Alternancia------")
                          #print(
                          #    f"prioridades: {prioridades} skills: {supervisor.escritorios_ON[un_escritorio]['skills']} \n{supervisor.escritorios_ON[un_escritorio]['pasos_alternancia'].pasos}"                       
                          #)                        
                          cliente_seleccionado = supervisor.escritorios_ON[un_escritorio]['pasos_alternancia'].buscar_cliente(fila_filtrada)
                          #break
                      elif configuracion_atencion == "FIFO":
                          cliente_seleccionado = FIFO(fila_filtrada)
                          #print(f"cliente_seleccionado por {un_escritorio} en configuración FIFO: su emisión fue a las: {cliente_seleccionado.FH_Emi}")
                          #break
                      elif configuracion_atencion == "Rebalse":
                          cliente_seleccionado = extract_highest_priority_and_earliest_time_row(fila_filtrada, prioridades)
                          #print(f"cliente_seleccionado por {un_escritorio} en configuración Rebalse: su emisión fue a las: {cliente_seleccionado.FH_Emi}")
                      fila = remove_selected_row(fila, cliente_seleccionado)
                      supervisor.iniciar_atencion(un_escritorio, cliente_seleccionado)
                      un_cliente   = pd.DataFrame(cliente_seleccionado[['FH_Emi', 'IdSerie', 'espera','IdEsc','T_Ate']]).T
                      registros_atenciones   =  pd.concat([registros_atenciones, un_cliente])#.reset_index(drop=True)

          try:
              #Iterar a la siguiente emisión
              #supervisor.aplicar_agenda(hora_actual=  reloj_simulacion, agenda = agenda)    
              una_emision            = next(generador_emisiones_in)
              emi                    = una_emision['FH_Emi']
              #print(f"siguiente emisión {emi}")
          except StopIteration:
              #print(f"-----------------------------Se acabaron las emisiones en la emision numero {numero_emision} ---------------------------")
              break   
  return pd.concat([fila[['FH_Emi','IdSerie','espera']], registros_atenciones]).sort_values(by='FH_Emi', inplace=False).reset_index(drop=True), len(fila)

              
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



""" 
Modificar clase `MisEscritorios` para que se instancie con `planificacion`, `niveles_servicio_x_serie` y `prioridades`
"""

planificacion = {'0': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills' : get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,          
    }}],
 '1': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '12': [{'inicio': '08:40:11',
   'termino': '10:07:40',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '33': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '34': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '35': [{'inicio': '11:36:03',
   'termino': '13:02:33',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '49': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series), 
    'configuracion_atencion':random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '50': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills': get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}],
 '51': [{'inicio': '13:02:56',
   'termino': '14:30:23',
   'propiedades': {'skills':get_random_non_empty_subset(series),
    'configuracion_atencion': random.sample(modos, 1)[0],
    'porcentaje_actividad'  : np.random.randint(75, 90)/100,
    }}]}

supervisor = MisEscritorios_v02(inicio_tramo  = un_dia['FH_Emi'].min(),
                                     fin_tramo     = un_dia['FH_Emi'].max(),
                                     planificacion = planificacion)

""" 
El NUEVO parámetro en optuna_simular_v02 "tipo_inactividad" puede ser 'Histórica' o "Porcentaje". 
Si es porcentaje usará el campo 'porcentaje_actividad' en la planificación, 
si no ('Histórica'), usará parámetros internos inferidos desde los datos: 
              'duracion_pausas': (1, 4, 47), #min, avg, max (desde pausas históricas).
              'probabilidad_pausas':.5, #probabilidad que la pausa ocurra  (desde pausas históricas).
              
              
optuna_simular_v02 llama a la clase MisEscritorios_v02.
"""
registros_atenciones, l_fila =  optuna_simular_v02(planificacion, niveles_servicio_x_serie, un_dia, prioridades, tipo_inactividad = 'Porcentaje' ) # 

# %%

#%%
import numpy as np
from datetime import datetime
from dev.atributos_de_series import atributos_x_serie
from src.datos_utils import *
from src.optuna_utils import *
from src.simulador_v02 import *  
import random
def reloj_rango_horario(start: str, end: str):
    start_time = datetime.strptime(start, '%H:%M:%S').time()
    end_time = datetime.strptime(end, '%H:%M:%S').time()
    start_minutes = start_time.hour * 60 + start_time.minute
    end_minutes = end_time.hour * 60 + end_time.minute
    for minute in range(start_minutes, end_minutes + 1):
        hours, remainder = divmod(minute, 60)
        yield "{:02d}:{:02d}:{:02d}".format(hours, remainder, start_time.second)


def emision_reloj_sync(timestamp1: str, timestamp2: str) -> bool:

    time1 = datetime.strptime(timestamp1, '%H:%M:%S').time()
    time2 = datetime.strptime(timestamp2, '%H:%M:%S').time()
    seconds1 = time1.hour * 3600 + time1.minute * 60 + time1.second
    seconds2 = time2.hour * 3600 + time2.minute * 60 + time2.second
    return abs(seconds1 - seconds2) < 60

def generador_emisiones(df):
    for _ , df.row in df.iterrows():
        yield df.row[['FH_Emi','IdSerie', 'T_Ate']] #, str(df.row.FH_Emi.time())  

def filter_by_time_range(df, start_time_str, end_time_str):
    start_time = datetime.strptime(start_time_str, '%H:%M:%S').time()  # Converts the start time string to a time object
    end_time = datetime.strptime(end_time_str, '%H:%M:%S').time()      # Converts the end time string to a time object
    mask = df['FH_Emi'].dt.time.between(start_time, end_time)          # Creates a boolean mask
    return df[mask] 


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
#%%
class match_emisiones_reloj():
    def __init__(self, bloque_atenciones) -> bool:
        
        self.bloque_atenciones = bloque_atenciones[['FH_Emi', 'IdSerie', 'T_Ate']]     
        
           
    def match(self, tiempo_actual):

        self.bloque_atenciones['FH_Emi'] = pd.to_datetime(self.bloque_atenciones['FH_Emi'])

        # Convert the given time string to a timedelta object
        h, m, s = map(int, tiempo_actual.split(':'))
        given_time = timedelta(hours=h, minutes=m, seconds=s)

        # Filter rows based on the given condition
        mask = self.bloque_atenciones['FH_Emi'].apply(
            lambda x: abs(timedelta(hours=x.hour, minutes=x.minute, seconds=x.second) - given_time) <= timedelta(seconds=60))
        
        # Rows that satisfy the condition
        self.match_emisiones   = self.bloque_atenciones[mask].copy()        
        self.bloque_atenciones = self.bloque_atenciones[~mask]#.copy()
        return self


       
hora_cierre           = '9:50:00'
reloj                 = reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
registros_atenciones  = pd.DataFrame()
matcher_emision_reloj = match_emisiones_reloj(un_dia)

#%%

for hora_actual in reloj:
    matcher_emision_reloj.match(hora_actual)#.match_emisiones
    #print(matcher_emision_reloj.match_emisiones)

    
    

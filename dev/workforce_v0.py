#%%
import warnings
warnings.filterwarnings("ignore")
from datetime import date, time
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import date, time
import math
import random 
from src.forecast_utils import *
##----------------------Datos históricos de un día----------------
dataset     = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills      = obtener_skills(el_dia_real)
series      = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs        = [(0.6, 30), (0.34, 35), (0.7, 45)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
registros_sla            = pd.DataFrame()
tabla_atenciones         = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
SLA_index                = 0
SLA_df                   = pd.DataFrame()
Espera_index =0 
Espera_df               = pd.DataFrame()             

def tiempo_espera_x_serie(registros_sla, series):   
    esperas_x_serie = {}
    for serie in series:
        espera_una_serie    = registros_sla[registros_sla.IdSerie == serie]['espera']
        if not espera_una_serie.empty:
            promedio_espera_cum = (espera_una_serie.expanding().mean()/60).iloc[-1]
            esperas_x_serie     = esperas_x_serie | {serie: int(promedio_espera_cum)}
        
    return esperas_x_serie

for cliente_seleccionado in tabla_atenciones.iterrows():
    

    un_cliente       =  pd.DataFrame(cliente_seleccionado[1][['FH_Emi', 'IdSerie', 'espera']]).T
    registros_sla    =  pd.concat([registros_sla, un_cliente])#.reset_index(drop=True)
    SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(registros_sla, niveles_servicio_x_serie).items()), columns=['keys', 'values'])
    SLA_index+=1                        
    SLA_una_emision['index']  = SLA_una_emision.shape[0]*[SLA_index]
    SLA_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    #SLA_una_emision['espera']= un_cliente.espera[un_cliente.espera.index[0]]
    SLA_df                    = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)
    ######################################################################################################################################
    Espera_una_emision        =  pd.DataFrame(list(tiempo_espera_x_serie(registros_sla, series).items()), columns=['keys', 'values'])
    Espera_index+=1
    Espera_una_emision['index']  = Espera_una_emision.shape[0]*[Espera_index]
    Espera_una_emision['hora']   = un_cliente.FH_Emi[un_cliente.FH_Emi.index[0]].time().strftime('%H:%M:%S')#
    Espera_df                    = pd.concat([Espera_df, Espera_una_emision], ignore_index=True)


trajectorias_SLAs    = SLA_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
trajectorias_esperas = Espera_df.pivot(index=['index', 'hora'], columns=['keys'], values='values').rename_axis(None, axis=1)
#%%



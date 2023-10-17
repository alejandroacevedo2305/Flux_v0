#%%
import warnings
warnings.filterwarnings("ignore")

from datetime import date, time
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from src.datos_utils import *
import optuna
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from datetime import date, time
import math
import random 



class ForecasterTTP:

    def __init__(self, 
        dataset : DatasetTTP, 
        series : list[int], 
        atenciones_diarias_minimas : int = 15, 
        horario : tuple[time, time] = ('8:00', '16:00'), 
        fecha_minima : date | None = None
        ):
        
        self.dataset = dataset
        self.series = series

        self.generador = None # generador vacio

        # LOGICA DE ATENCIONES
        self.dataset.atenciones_agg["Demanda_diaria"] = self.dataset.atenciones_agg_dia["Demanda"]

        self.dataset.atenciones_agg["Demanda_diaria"] = self.dataset.atenciones_agg["Demanda_diaria"].ffill()
        self.dataset.atenciones_agg = self.dataset.atenciones_agg[ self.dataset.atenciones_agg["Demanda_diaria"] >= atenciones_diarias_minimas ]
        
        # Se encarga de recortar los bloques cerrados de la oficina
        self.dataset.atenciones_agg = (  dataset
            .atenciones_agg
            .reset_index(level=['IdOficina','IdSerie'], drop=False)
            .between_time( start_time = horario[0], end_time = horario[1] )
            .fillna(0)
        )

        self.delta_horas : int = ( pd.Timestamp( horario[1] ) - pd.Timestamp( horario[0] ) ).seconds // 3600

        # En este punto el unico indice es una DateTime index, asi que puedo cortar
        if fecha_minima:
            # '2023-05-05'
            self.dataset.atenciones_agg = self.dataset.atenciones_agg[ dataset.atenciones_agg.index.to_series() > pd.Timestamp(fecha_minima) ]

        # Cursed linea que crea un indice monotonicamente ascendente. Nota, no es necesario que sea base 0 para cada grupo
        self.dataset.atenciones_agg = dataset.atenciones_agg.reset_index().rename_axis(index='dt_idx').reset_index()


    @staticmethod # Una cuestion lacra, que deberia ser implementada en Pandas
    def sliding_window(ts : list[float], features : int):
        """
        Retorna dos listas de x [-features, -1] , [0] sobre una lista
        
        Ejemplo: X es `[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 
        6, 7], [6, 7, 8], [7, 8, 9]]`, e Y es `[[4], [5], [6], [7], [8], [9],  
        [10]]`.    
        """
        X : list[list[float]]= []
        Y : list[list[float]]= []

        for i in range(features + 1, len(ts) + 1):
            X.append(ts[i - (features + 1):i - 1])
            Y.append([ts[i - 1]])

        return X #, Y # no necesito Y, de momento


    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def proyectar_demanda_SARIMAX(self , dia_inicio : date ):
        """Genera un forecasting via SARIMAX"""

        dataset = self.dataset
        series = self.series

        timelimit = pd.Timestamp(dia_inicio) # SELECCIONA UN DIA
        timelimit_below = timelimit - pd.Timedelta(weeks=4)
        timelimit_upper = timelimit + pd.Timedelta(weeks=1)

        self.serie_t_espera : pd.Serie = dataset.atenciones_agg[
                (timelimit_below <= dataset.atenciones_agg["FH_Emi"]) & (dataset.atenciones_agg["FH_Emi"] <= timelimit) 
            ].groupby(by=[ "IdSerie"]).median()["T_Ate"]

        demandas_proyectadas = {}
        
        for SERIE in series: 
            
            ts_pre = dataset.atenciones_agg[ 
                (timelimit_below <= dataset.atenciones_agg["FH_Emi"]) & 
                (dataset.atenciones_agg["FH_Emi"] <= timelimit) & 
                (dataset.atenciones_agg["IdSerie"] == SERIE) 
            ]["Demanda"].tolist() 

            ts_post = dataset.atenciones_agg[ 
                (dataset.atenciones_agg["FH_Emi"] > timelimit) & 
                (dataset.atenciones_agg["FH_Emi"] < timelimit_upper) & 
                (dataset.atenciones_agg["IdSerie"] == SERIE) 
            ]["Demanda"].tolist() 


            windows = ForecasterTTP.sliding_window(ts_pre, features=self.delta_horas*10) # [0]
            last_values = []

            for wind in windows:
                model = SARIMAX(
                    endog = wind,
                    exog = None,
                    order = (1, 1, 1),
                    seasonal_order = (1, 1, 1, self.delta_horas)
                )

                results = model.fit(disp = 0)
                forecast = results.forecast()
                last_values.append([forecast[0]])

            demandas_proyectadas[f"{SERIE}_proyectada"] = [ max(0, value[0]) for value in last_values  ] # FIXME: esto esta corrido en idx 1
            demandas_proyectadas[f"{SERIE}_real"] = ts_post

        self.dia_inicio = dia_inicio
        self.demandas_proyectadas = demandas_proyectadas

        return self
    
    def demandas_proyectada_cuantizadas(self):
        """Retorna una funcion cuantizada de la demanda proyectada"""

        demandas_proyectadas = self.demandas_proyectadas
        timeindex = self.dataset.atenciones_agg["FH_Emi"].drop_duplicates()

        for SERIE in demandas_proyectadas.keys():
            n_dias = pd.Series(demandas_proyectadas[SERIE]).apply(math.ceil)
            n_dias[ n_dias < 0 ] = 0 # Reemplaza los ceros
            n_dias_len = n_dias.size
            demandas_proyectadas[SERIE] = n_dias.set_axis(
                timeindex[ (timeindex > (pd.Timestamp(self.dia_inicio) - pd.Timedelta(weeks=20))) ].iloc[:n_dias_len] + pd.Timedelta(weeks=20)
            )
            
        df = pd.DataFrame(demandas_proyectadas)
        df = df[ df.columns[df.columns.str.endswith('_proyectada')] ]
        df.columns = df.columns.str.removesuffix('_proyectada')
        df = df.unstack(level=0).reset_index().rename({"level_0" : "IdSerie", 0 : "Demanda"}, axis='columns')
        df = df.loc[df.index.repeat(df["Demanda"])].reset_index(drop=True)

        df["FH_Emi"] = df["FH_Emi"].apply( lambda x : x + pd.Timedelta(seconds=random.randint(1,3599)) )
        df["T_Ate"] = df["IdSerie"].apply( lambda x : forecaster_ttp.serie_t_espera[int(x)] ) # Cosa para incluir tiempos de atencion estimados

        return df.drop("Demanda", axis="columns").sort_values(by="FH_Emi").reset_index(drop=True)

    @staticmethod
    def un_dia(dia : date, demandas_proyectadas_cuantizada : pd.DataFrame):
        """Retorna las atenciones de un dia predicho"""

        inicio_dia = pd.Timestamp( dia )
        fin_dia = pd.Timestamp( dia ) + pd.Timedelta(days=1)

        return demandas_proyectadas_cuantizada[ (inicio_dia <= demandas_proyectadas_cuantizada["FH_Emi"]) & (demandas_proyectadas_cuantizada["FH_Emi"] <= fin_dia) ]

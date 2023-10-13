#%%
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from scipy.stats import gmean
from dataclasses import dataclass
from datetime import date, time, datetime
@dataclass
class DatasetTTP:
    atenciones : pd.DataFrame
    atenciones_agg : pd.DataFrame
    atenciones_agg_dia : pd.DataFrame

    @staticmethod
    def _reshape_df_atenciones(df : pd.DataFrame, resample = "60t") -> 'pd.DataFrame':

        resampler = ( df
            .set_index("FH_Emi")
            .groupby(by=["IdOficina","IdSerie"])
            .resample(resample)
        )

        medianas = resampler.median()[["T_Esp","T_Ate"]]

        medianas["Demanda"] = ( resampler
            .count().IdSerie
            .rename("Demanda") # Esto es una serie, asi que solo tiene un nombre
            # .reset_index(level="IdSerie") #
        )

        return medianas[["Demanda","T_Esp","T_Ate"]]

    @staticmethod
    def _atenciones_validas( df ) -> 'pd.DataFrame':
        """Un helper para limpiar la data en base a ciertas condiciones logicas"""

        # TODO: implementar loggin de cosas (n) que se eliminan en estas atenciones raras
        # TODO: Posiblemente limpiar esto en un unico . . ., o usar Polars
        df = df.dropna( how = 'any' ) # Inmediatamente elimina los NaN
        df = df[~(df["FH_Emi"] == df["FH_AteFin"])]
        df = df[~(df["FH_AteIni"] == df["FH_Emi"])]
        df = df[~(df["FH_AteIni"] == df["FH_AteFin"])]
        df = df[~(df["FH_Emi"] > df["FH_Llama"])]
        df = df[~((df["FH_Llama"] - df["FH_Emi"]) > pd.Timedelta(hours=12))]
        df = df[~((df["FH_AteIni"] - df["FH_Llama"]) > pd.Timedelta(hours=1))]
        df = df[~((df["FH_AteFin"] - df["FH_AteIni"]) < pd.Timedelta(seconds=5))]

        return df

    @staticmethod
    def desde_csv_atenciones( csv_path : str ) -> 'DatasetTTP':
        df = pd.read_csv( csv_path, 
            usecols = ["IdOficina","IdSerie","IdEsc","FH_Emi","FH_Llama","FH_AteIni","FH_AteFin"], 
            parse_dates = [3, 4, 5, 6]
            ).astype({
                "IdOficina" : "Int32",
                "IdSerie" : "Int8",
                "IdEsc" : "Int8",
                "FH_Emi" : "datetime64[s]",
                "FH_Llama" : "datetime64[s]",
                "FH_AteIni" : "datetime64[s]",
                "FH_AteFin" : "datetime64[s]",
            })
        
        df = DatasetTTP._atenciones_validas(df) # Corre la funcion de limpieza

        df["T_Esp"] = ( df["FH_AteIni"] - df["FH_Emi"] ).dt.seconds 
        df["T_Ate"] = ( df["FH_AteFin"] - df["FH_AteIni"] ).dt.seconds 

        return DatasetTTP( 
            atenciones = df, 
            atenciones_agg = DatasetTTP._reshape_df_atenciones( df ),
            atenciones_agg_dia = DatasetTTP._reshape_df_atenciones( df, resample = '1D' )
         )


    @staticmethod
    def desde_sql_server() -> 'DatasetTTP':

        sql_query = """-- La verdad esto podria ser un View, pero no sé si eso es mas rapido en MS SQL --
        SELECT
            -- TOP 5000 -- Por motivos de rendimiento
            IdOficina, -- Sucursal fisica
            IdSerie,   -- Motivo de atencion
            IdEsc,     -- Escritorio de atencion

            FH_Emi,    -- Emision del tiquet, con...
            -- TIEMPO DE ESPERA A REDUCIR --
            FH_Llama,  -- ... hora de llamada a resolverlo,  ...
            FH_AteIni, -- ... inicio de atencion, y
            FH_AteFin  -- ... termino de atencion

        FROM Atenciones -- Unica tabla que estamos usando por ahora -- MOCK
        -- FROM dbo.Atenciones -- Unica tabla que estamos usando por ahora -- REAL

        WHERE
            (IdSerie IN (10, 11, 12, 14, 5)) AND 
            (FH_Emi > '2023-01-01 00:00:00') AND     -- De este año
            (FH_Llama IS NOT NULL) AND (Perdido = 0) -- CON atenciones
            
        ORDER BY FH_Emi DESC; -- Ordenado de mas reciente hacia atras (posiblemente innecesario)"""
        raise NotImplementedError("Este metódo aún no está implementado.")


    @staticmethod
    def desde_csv_batch() -> 'DatasetTTP':
        raise NotImplementedError("Este metódo aún no está implementado.")

    def un_dia(self, dia : date):
        """Retorna las atenciones de un dia historico"""

        inicio_dia = pd.Timestamp( dia )
        fin_dia = pd.Timestamp( dia ) + pd.Timedelta(days=1)

        return self.atenciones[ (inicio_dia <= self.atenciones["FH_Emi"]) & (self.atenciones["FH_Emi"] <= fin_dia) ]



def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))


#un_dia   = forecast().un_mes().un_dia().df.sort_values(by='FH_Emi', inplace=False)
#sucursal = 2
#un_dia   = un_dia[un_dia.IdOficina == sucursal] #9 2
# esto funciona
dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
un_dia = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.84, 10), (0.34, 25), (0.8, 33)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
# modos_atenciones=["Alternancia", "FIFO", "Rebalse"]
# max_escritorios = len(skills.keys())
# prioridades = prioridad_x_serie(niveles_servicio_x_serie,2,1) 
# action_space = get_action_space(
#     skills     = skills, 
#     config_set = modos_atenciones, 
#     time_start = str(un_dia.FH_Emi.min().time()), 
#     time_end   = str(un_dia.FH_Emi.max().time())
# )
# plan = convert_output(action_space.sample(), skills_set = series, config_set = modos_atenciones)
# Optuna multi-objective function
def objective(trial, un_dia,skills, niveles_servicio_x_serie,  modos_atenciones:list = ["Alternancia", "FIFO", "Rebalse"]):
    try:
        bool_vector = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]  
        #modos_atenciones = ["Alternancia", "FIFO", "Rebalse"]
        series = sorted(list({val for sublist in skills.values() for val in sublist}))
        str_dict = {i: trial.suggest_categorical('modo atención', modos_atenciones) for i in range(len(skills.keys()))}
        subset_dict = {i : trial.suggest_categorical('series', non_empty_subsets(series)) for i in range(len(skills.keys()))}
        # Storing user attributes
        trial.set_user_attr('bool_vector', bool_vector)
        trial.set_user_attr('str_dict', str_dict)
        trial.set_user_attr('subset_dict', subset_dict)

        #SLAs                     = [(0.84, 10), (0.34, 25), (0.8, 33)]
        #niveles_servicio_x_serie = {s: random.choice(SLAs) for s in series}
        prioridades              = prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        p                        = {}
        inicio                   = str(un_dia.FH_Emi.min().time())#'08:33:00'
        termino                  =  str(un_dia.FH_Emi.max().time())#'14:33:00'
        # Loop through the keys
        for key in str_dict.keys():
            # Apply boolean mask
            if bool_vector[key]:
                # Create the inner dictionary
                inner_dict = {
                    'inicio': inicio,
                    'termino': termino,
                    'propiedades': {
                        'skills': list(subset_dict[key]),
                        'configuracion_atencion': str_dict[key]
                    }
                }
                # Convert integer key to string and add the inner dictionary to a list, then add it to the output dictionary
                p[str(key)] = [inner_dict]
        
        registros_SLA = simular(p, niveles_servicio_x_serie, un_dia, prioridades)
        
        print(f"{str_dict}")
        print(f"{subset_dict}")

        
        return gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna()), sum(bool_vector)

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()


study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'])
study.optimize(lambda trial: objective(trial, 
                                       un_dia                   = un_dia,
                                       skills                   = skills,
                                       niveles_servicio_x_serie = niveles_servicio_x_serie,
                                       ), 
                                       n_trials                 = 20,
                                       n_jobs=-1)

# Results: the first Pareto front (i.e., the best trade-offs between the two objectives)
pareto_front_trials = study.get_pareto_front_trials()
for i, trial in enumerate(pareto_front_trials):
    print(f"Trial {i+1}")
    print(f"Nivel de servicio global (maximize): {trial.values[0]}")
    print(f"Número de escritorios (minimize): {trial.values[1]}")
    print(f"Configuración: {trial.params}")


# %%

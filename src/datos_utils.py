from dataclasses import dataclass
from datetime import date, time, datetime
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def obtener_skills(un_dia):   

    skills_defaultdict  = defaultdict(list)  

    #for index, row in un_dia.df.iterrows():
    for index, row in un_dia.iterrows():

        skills_defaultdict[row['IdEsc']].append(row['IdSerie'])
    for key in skills_defaultdict:
        skills_defaultdict[key] = list(set(skills_defaultdict[key]))
        
    skills = dict(skills_defaultdict)   
    return {f"{k}": v for k, v in skills.items()}#
    #return {f"escritorio_{k}": v for k, v in skills.items()}#

def process_dataframe(df, freq='1H', factor_conversion_T_esp:int=60):
    df['FH_Emi'] = pd.to_datetime(df['FH_Emi'])    
    # Convert 'espera' column to float dtype
    df['espera'] = df['espera'].astype(float)    
    # Set FH_Emi as the index for easier resampling
    df.set_index('FH_Emi', inplace=True)    
    # Create the first output dataframe: Count of number of events per custom interval
    df_count = df.resample(freq).size().reset_index(name='Count')    
    # Create the second output dataframe: Average of 'espera' values per custom interval
    df_avg = df.resample(freq).agg({'espera': 'mean'}).reset_index()
    df_avg['espera'] = df_avg['espera']/factor_conversion_T_esp  
    return df_count, df_avg


# def plot_count_and_avg(df_count, df_avg):

#     x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(df_count['FH_Emi'].dt.strftime('%H:%M:%S'), (df_count['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S'))]
    
#     # Create the bar plot
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     bars = ax1.bar(x_labels, df_count['Count'], alpha=0.6, label='Count')    
#     # Create the line plot for average values
#     ax2 = ax1.twinx()
#     ax2.plot(x_labels, df_avg['espera'], color='r', marker='o', label='Average')    
#     # Add labels and title
#     ax1.set_xlabel('Time Interval')
#     ax1.set_ylabel('Count', color='b')
#     ax2.set_ylabel('Average', color='r')
#     plt.title('Count of Events and Average of Espera per Time Interval')    
#     # Set x-ticks to be centered and rotated
#     ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
#     ax1.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
    
#     # Add legends
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
    
#     # Show the plot
#     plt.show()



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


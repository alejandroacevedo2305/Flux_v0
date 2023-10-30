
import pandas as pd
from modelos_datos.api.simulador import Input_Simulador, Output_Simulador

def atenciones_a_sim_out(atenciones : pd.DataFrame, sim_input : Input_Simulador) -> Output_Simulador:
    """
    Calcula los SLA y datos resumen

    Toma una Tabla de atenciones y un diccionario de Cortes de tiempos de espera (en segundos) 
    y genera un SLA global

    Puedes generar el diccionario de entrada con un `{ int(serie.idSerie) : serie.tiempoMaximoEspera for serie in sim_input.series }`

    """

    # Un diccionario con los tiempos de espera en segundos mapeados a cada serie
    timepos_espera = { int(serie.idSerie) : serie.tiempoMaximoEspera for serie in sim_input.series }

    def check_cumple_sla(row):
        """
        Retorna True si esta bajo el umbral de espera del SLA
        
        En caso de no estar descrito, retorna None. 
        """
        # TODO: implementar una version vectorizada

        serie = row['IdSerie']
        waiting_time = row['T_Esp']
        
        try:
            threshold = timepos_espera[serie]
        except KeyError:
            return None 
        
        return waiting_time <= threshold

    atenciones['Cumple_SLA'] = atenciones.apply(check_cumple_sla, axis=1)

    # atenciones = atenciones[ atenciones['IdOficina'] == IdOficina ]

    # atenciones.set_index(['IdSerie','FH_AteFin']).sort_index()
    atenciones['Cumulative_SLA'] = ( atenciones
        .groupby(by=['IdSerie', atenciones["FH_AteFin"].dt.hour ])['Cumple_SLA']
        .expanding(1).mean()
        .reset_index(level=['IdSerie','FH_AteFin'])['Cumple_SLA']
    )

    sla_instantaneo = atenciones.set_index('FH_AteFin').groupby(by='IdSerie').resample('H').aggregate({
        'T_Esp' : 'mean', 
        'T_Ate' : 'mean', 
        'Cumple_SLA' : 'mean',
        'IdOficina' : 'count'
    }).reset_index().rename({
        'Cumple_SLA' : 'SLA',
        'IdOficina' : 'n_Ate',
        'FH_AteFin' : 'FH'
    }, axis='columns').fillna(0)

    sla_cumulativo = atenciones.groupby(by='IdSerie').aggregate({
        'T_Esp' : 'mean', 
        'T_Ate' : 'mean', 
        'Cumple_SLA' : 'mean',
        'IdOficina' : 'count'
    }).reset_index().rename({
        'Cumple_SLA' : 'SLA',
        'IdOficina' : 'n_Ate'
    }, axis='columns').fillna(0)

    resumen = atenciones.aggregate({
        'T_Esp' : 'mean', 
        'T_Ate' : 'mean', 
        'Cumple_SLA' : 'mean',
        'IdOficina' : 'count'
    }).rename({
        'Cumple_SLA' : 'sla_glob',
        'IdOficina' : 'n_Ate'
    }, axis='index') # Falta numero de escritorios

    uso_escritorios = atenciones.set_index('FH_AteFin').groupby(by='IdEsc').resample('H').aggregate({
        'T_Ate' : 'sum', 
        'IdOficina' : 'count'
    }).reset_index().rename({
        'IdOficina' : 'n_Ate',
        'FH_AteFin' : 'FH'
    }, axis='columns')

    # print(uso_escritorios)

    uso_escritorios['T_Ate'] = uso_escritorios['T_Ate'] / 60 # A MINUTOS

    resumen["n_Eje"] = 10 # uso_escritorios['IdEsc'].nunique()

    return Output_Simulador(**{
        "resumen" : resumen.to_dict(),
        "sla_instantaneo" : sla_instantaneo.to_dict(orient='records'), # FUNCIONA
        "sla_cumulativo" : sla_cumulativo.to_dict(orient='records'),   # FUNCIONA
        "uso_escritorios" : uso_escritorios.to_dict(orient='records')  # FUNCIONA
    })

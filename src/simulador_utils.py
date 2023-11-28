import pandas as pd
from modelos_datos.api.simulador import Input_Simulador, Output_Simulador


def atenciones_a_sim_out(
    atenciones: pd.DataFrame, series: list
):  # -> Output_Simulador:
    """
    Calcula los SLA y datos resumen

    Toma una Tabla de atenciones y un diccionario de Cortes de tiempos de espera (en segundos)
    y genera un SLA global

    Puedes generar el diccionario de entrada con un `{ int(serie.idSerie) : serie.tiempo_maximo_espera for serie in sim_input.series }`

    """

    # Un diccionario con los tiempos de espera en segundos mapeados a cada serie
    # que luego es convertido a un timedelta en segundos, para comparar
    try:
        timepos_espera = {
            int(serie.IdSerie): serie.tiempo_maximo_espera for serie in series
        }
    except:
        timepos_espera = {
            int(serie["IdSerie"]): serie["tiempo_maximo_espera"] for serie in series
        }

    atenciones["Cumple_SLA"] = (
        atenciones["IdSerie"].map(timepos_espera) >= atenciones["t_esp"]
    )

    # atenciones = atenciones[ atenciones['IdOficina'] == IdOficina ]

    # atenciones.set_index(['IdSerie','FH_AteFin']).sort_index()
    atenciones["Cumulative_SLA"] = (
        atenciones.groupby(by=["IdSerie", atenciones["FH_AteFin"].dt.hour])[
            "Cumple_SLA"
        ]
        .expanding(1)
        .mean()
        .reset_index(level=["IdSerie", "FH_AteFin"])["Cumple_SLA"]
    )

    sla_instantaneo = (
        atenciones.set_index("FH_AteFin")
        .groupby(by="IdSerie")
        .resample("H")
        .aggregate(
            {
                "t_esp": "mean",
                "t_ate": "mean",
                "Cumple_SLA": "mean",
                "IdOficina": "count",
            }
        )
        .reset_index()
        .rename(
            {"Cumple_SLA": "sla", "IdOficina": "n_ate", "FH_AteFin": "FH"},
            axis="columns",
        )
        .fillna({"sla": 1})
        .fillna(0)
        .sort_values(by="FH")
    )

    sla_instantaneo_global = (
        atenciones.set_index("FH_AteFin")
        .resample("H")
        .aggregate(
            {
                "t_esp": "mean",
                "t_ate": "mean",
                "Cumple_SLA": "mean",
                "IdOficina": "count",
            }
        )
        .reset_index()
        .assign(IdSerie=0)
        .rename(
            {"Cumple_SLA": "sla", "IdOficina": "n_ate", "FH_AteFin": "FH"},
            axis="columns",
        )
        .fillna({"sla": 1, "n_ate": 0, "t_esp": 0, "t_ate": 0})
    )

    sla_cumulativo = (
        atenciones.groupby(by="IdSerie")
        .aggregate(
            {
                "t_esp": "mean",
                "t_ate": "mean",
                "Cumple_SLA": "mean",
                "IdOficina": "count",
            }
        )
        .reset_index()
        .rename({"Cumple_SLA": "sla", "IdOficina": "n_ate"}, axis="columns")
        .fillna(0)
    )

    resumen = atenciones.aggregate(
        {"t_esp": "mean", "t_ate": "mean", "Cumple_SLA": "mean", "IdOficina": "count"}
    ).rename(
        {"Cumple_SLA": "sla_glob", "IdOficina": "n_ate"}, axis="index"
    )  # Falta numero de escritorios

    uso_escritorios = (
        atenciones.set_index("FH_AteFin")
        .groupby(by="IdEsc")
        .resample("H")
        .aggregate({"t_ate": "sum", "IdOficina": "count"})
        .reset_index()
        .rename({"IdOficina": "n_ate", "FH_AteFin": "FH"}, axis="columns")
        .sort_values(by=["IdEsc", "FH"])
    )

    # print(uso_escritorios)

    uso_escritorios["porcentaje_ocupacion"] = (
        uso_escritorios["t_ate"] / 3600
    )  # Horas en atencion

    resumen["n_Eje"] = uso_escritorios["IdEsc"].nunique()

    return {
        "resumen": resumen.to_dict(),
        "sla_cumulativo": sla_cumulativo.to_dict(orient="records"),  # FUNCIONA
        "sla_instantaneo": sla_instantaneo.to_dict(orient="records")
        + sla_instantaneo_global.to_dict(orient="records"),  # FUNCIONA
        "uso_escritorios": uso_escritorios.to_dict(orient="records"),  # FUNCIONA
    }

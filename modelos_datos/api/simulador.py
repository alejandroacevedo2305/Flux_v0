"""
Modelos de datos para la API del Simulador

Esto son modelos de Pydantic para hacer la serializacion (y viceversa) de datos pasados desde el Front en Angular, 
que entran en un formato adecuado para las funciones del back. No necesariamente similar a las funciones del Front 
demo implementado con streamlit. 
"""

from datetime import date, time, datetime

from pydantic import BaseModel

# INPUTS

class Input_Simulador(BaseModel):
    horario_oficina: tuple[time, time]

    class Escritorio(BaseModel):
        """Eventualmente tendremos mas data que solo las configuraciones"""

        class Esc_Configuracion(BaseModel):
            """En principio queremos guardar varias"""

            inicio_fin: tuple[time, time]

            class Esc_Conf_Atencion(BaseModel):
                """Esto existe por las APIs existentes"""

                modo: int

                class JsonSerie(BaseModel):
                    """Esto existe por las APIs existentes, again"""

                    IdSerie: int
                    Prioridad: int = 2
                    Alterna: int = 1

                jsonSerie: list[JsonSerie]

            configuracion_atencion: Esc_Conf_Atencion

        configuraciones: list[Esc_Configuracion]

    planificacion: dict[int, Escritorio]

    class Serie(BaseModel):
        idSerie: int
        serie: str
        tiempoMaximoEspera: int = 6000 # En segundos
        SLA_objetivo: float = 0.8
        # Horarios de las series
        hI1: str = "07070707070707" # HH x 7
        mI1: str = "00000000000000" # mm x 7
        hF1: str = "12121212121212" # HH x 7
        mF1: str = "00000000000000" # mm x 7
        hI2: str = "12121212121212" # HH x 7
        mI2: str = "00000000000000" # mm x 7
        hF2: str = "22222222222222" # HH x 7
        mF2: str = "00000000000000" # mm x 7

    series: list[Serie]



def Input_Simulador_a_Simulador( input_sim : Input_Simulador ):
    """La estructura de datos _para_ el Simulador"""
    # FIXME: Esto deberia estar en otra parte. Posiblmente un manicomio.

    planificacion = {
        k : [{
            "inicio" : config.inicio_fin[0],
            "termino" : config.inicio_fin[1], 
            "propiedades" : {
                "skills" : [ serie.IdSerie for serie in config.configuracion_atencion.jsonSerie ],
                "configuracion_atencion" : ("Alternancia", "FIFO", "Rebalse")[config.configuracion_atencion.modo]
            }
        } for config in esc.configuraciones ]
    for k, esc in input_sim.planificacion.items() }

    return planificacion


# RESPUESTAS
class Output_Simulador(BaseModel):
    """Modelo de validacion de salida del simulador"""

    class Resumen(BaseModel):
        """Data resumen de la simulacion"""
        T_Esp : float # Cummean del tiempo de espera
        T_Ate : float # Cummean del tiempo de atencion
        n_Eje : int # Count de escritorios
        n_Ate : int # Count de atenciones
        sla_glob : float # Weighted SLA average

    resumen : Resumen # Clase resumen

    class SLA_cumulativo(BaseModel): 
        """Row de los SLA cumulativo"""
        IdSerie : int
        SLA : float
        n_Ate : int
        T_Ate : float
        T_Esp : float

    class SLA_intantaneo(SLA_cumulativo):
        """Row de los SLA intantaneos"""
        # Extiende la clase anterior
        FH : datetime


    sla_cumulativo : list[SLA_cumulativo]
    sla_instantaneo : list[SLA_intantaneo]

    class Uso_Escritorio(BaseModel):
        """Row de los usos de escritorios"""
        FH : datetime
        IdEsc : int
        n_Ate : int = 1
        T_Ate   : float = 0 # Tiempo de atencion total
        T_Inact : float = 0 # Tiempo de inactividad total (pausas, desconectado)
        T_Disp  : float = 0 # Tiempo conectado esperando una atencion

    uso_escritorios : list[Uso_Escritorio]

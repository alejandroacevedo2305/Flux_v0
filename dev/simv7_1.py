#%%
import os

os.chdir("/DeepenData/Repos/Flux_v0")
import warnings

warnings.filterwarnings("ignore")
import time

import os
from datetime import date

import pandas as pd  # Dataframes
import polars as pl  # DFs, es más rapido
from pydantic import BaseModel, Field, ConfigDict  # Validacion de datos
from datetime import date, datetime

import releases.simv7 as sim
class DatasetTTP(BaseModel):
    """
    Tablas de atenciones y configuraciones

    Un wrapper que permite consultar una base de datos sobre las atenciones y configuraciones de una oficina.
    """

    # Parametros de instanciacion, esto no llena de data hasta llamar otros metodos
    connection_string: str = Field(description="Un string de conexion a una base de datos")
    id_oficina: int = Field(description="Identificador numerico de una oficina")

    # FIXME: esto existe porque la clase no esta definida como un tipo valido
    # (quiero tipos porque atrapan errores antes de que se propaguen)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Componentes que deben poder entrar inmediatamente al simulador
    atenciones: pd.DataFrame | pl.DataFrame = None  # TODO: reemplazar por Pandera
    planificacion: dict = None  # TODO: reemplazar por Pydantic
    configuraciones: pd.DataFrame | pl.DataFrame = None  # Post-inicializado

    # Fechas que se extiende el dataset
    FH_min: datetime = None
    FH_max: datetime = None

    # Fechas que se extiende el dataset cargado
    atenciones_FH_min: datetime = None
    atenciones_FH_max: datetime = None

    # Mete todo en un caché para este objeto, para asi tener que es cacheable o no
    def __hash__(self):
        return hash(f"{self.id_oficina}, {self.FH_max}") # {self.atenciones_FH_max},{self.atenciones_FH_min}")


    def model_post_init(self, _) -> None:
        # Llama a la ultima configuracion de la oficina
        self.configuraciones = pl.read_database_uri(
            uri=self.connection_string,
            query=f"""
            SELECT * FROM Configuraciones WHERE (IdOficina = {self.id_oficina});
            """,
        ).sort(by=["IdEsc", "prioridad"])

        # Determina los rangos en que se puede llamar este dataset
        self.FH_min, self.FH_max = (
            pl.read_database_uri(
                uri=self.connection_string,
                query="SELECT min(FH_Emi) AS min, max(FH_Emi) AS max FROM Atenciones;",
            )
            .to_dicts()[0]
            .values()
        )

    # Metodos para rellenar atenciones y planificacion
    def forecast(self):
        # Conectare la logica luego. La idea es poder dejar los forecast en la base de datos, 
        # para pre-calcularlos, dado que eso demora unos minutos y es algo tedioso
        raise NotImplementedError

    def un_dia(self, fecha: date) -> tuple[pd.DataFrame, dict]:
        """
        Modifica las `atenciones` y `planificacion` a una fecha historica

        Usa una fecha en un formato reconocible por Pandas, devuelve error en caso de no
        tener atenciones para ese dia. Retorna la tabla de atenciones y el diccionario de
        planificacion para el dia, y modifica esto en el objeto `self` para no tener que
        ejecutar esta funcion de nuevo para un mismo dia.
        """
        # PARTE TRIVIAL, LEE LAS ATENCIONES DE UNA FECHA
        fecha = pd.Timestamp(fecha)  # Converte la fecha a un timestamp

        # NOTA: el query llama a las columnas por nombre para asi verificar que estas están
        # en el schema de la base de datos, else, se rompe por un error de MySQL.
        self.atenciones = pl.read_database_uri(
            uri=self.connection_string,
            query=f"""
            SELECT 
                IdOficina, -- Sucursal fisica
                IdSerie,   -- Motivo de atencion
                IdEsc,     -- Escritorio de atencion
                FH_Emi,    -- Emision del tiquet, con...
                FH_Llama,  -- ...hora de llamada a resolverlo,...
                FH_AteIni, -- ...inicio de atencion, y...
                FH_AteFin, -- ...termino de atencion
                t_esp,     -- Tiempo de espera    
                t_ate      -- Tiempo de atención
            FROM 
                Atenciones
            WHERE
                (IdOficina = {self.id_oficina} ) AND -- Selecciona solo una oficina, segun arriba
                (FH_Emi > '{fecha}') AND (FH_Emi < '{fecha + pd.Timedelta(days=1)}') -- solo el dia

            ORDER BY FH_Emi DESC -- Ordenado de mas reciente hacia atras (posiblemente innecesario);
            """,
        )

        # NOTA: esto seria solo .empty (atributo) en un pd.DataFrame, pero Polars es mas rapido
        if self.atenciones.is_empty():
            raise Exception("Tabla de atenciones vacia", f"Fecha sin atenciones: {fecha}")

        # Update del caché de atenciones 
        self.atenciones_FH_min = self.atenciones["FH_Emi"].min()
        self.atenciones_FH_max = self.atenciones["FH_Emi"].max()


        # PRIORIDADES DE SERIES, COMPATIBLE CON INFERIDAS Y GUARDADAS EN CONFIG
        lista_config = (
            # Lista de prioridades por serie, en la configuracion ultima
            self.configuraciones.group_by(by=["IdSerie"])  # BUG: el linter no reconoce esto correctamente
            .agg(pl.mean("prioridad"), pl.count("IdEsc"))
            .sort(by=["prioridad", "IdEsc"], descending=[False, True])["IdSerie"]
            .to_list()
        )

        lista_atenciones = (
            # Usa un contador de atenciones para rankear mas arriba con mas atenciones
            self.atenciones["IdSerie"].value_counts().sort(by="counts", descending=True)["IdSerie"].to_list()
        )

        # Esto genera una lista global de prioridades, donde todo lo demas puede ir a la cola
        map_prioridad = {
            id_serie: rank + 1
            for rank, id_serie in enumerate(
                # Combina ambas listas, prefiriendo la de la configuracion ultima sobre la inferida del dia
                [s for s in lista_config if (s in lista_atenciones)]
                + [s for s in lista_atenciones if (s not in lista_config)]
            )
        }

        # TODO: Esta parte genera una lista de prioridades en la configuracion del dia,
        # resolviendo incompatibilidades de que algo no esté originalmente.
        df_configuraciones = (
            self.atenciones.select(["IdEsc", "IdSerie"])
            .unique(keep="first")
            .sort(by=["IdEsc", "IdSerie"], descending=[True, True])
            .with_columns(prioridad=pl.col("IdSerie").replace(map_prioridad, default=None))
        )

        # Esta es la tabla de metadata para las series, por separado por comprension
        df_series_meta = (
            self.configuraciones.select(["IdSerie", "serie", "tiempo_maximo_espera"])
            .unique(keep="first")
            .with_columns(prioridad=pl.col("IdSerie").replace(map_prioridad, default=None))
        )

        # Se unifica con la de configuracion diaria
        df_configuraciones = df_configuraciones.join(df_series_meta, on="IdSerie")

        # EMPIEZA A CONSTRUIR LA PLANIFICACION DESDE HORARIOS Y ESCRITORIOS
        horarios = self.atenciones.group_by(by=["IdEsc"]).agg(
            inicio=pl.min("FH_Emi").dt.round("1h"),
            termino=pl.max("FH_AteIni").dt.round("1h"),
        )

        self.planificacion = {
            e["IdEsc"]: [
                {
                    "inicio": str(e["inicio"].time()),
                    "termino": str(e["termino"].time()),
                    "propiedades": {
                        # Esta parte saca los skills comparando con el dict de configs
                        "skills": df_configuraciones.filter(pl.col("IdEsc") == e["IdEsc"])["IdSerie"].to_list(),
                        # asumimos que la configuracion de todos es rebalse
                        # TODO: el porcentaje de actividad es mas complicado, asumire un 80%
                        "configuracion_atencion": "Rebalse",
                        "porcentaje_actividad": 0.80,
                        # Esto es innecesariamente nested
                        "atributos_series": [
                            {
                                "serie": s["IdSerie"],
                                "sla_porcen": 80,  # Un valor por defecto
                                "sla_corte": s["tiempo_maximo_espera"],
                                "pasos": 1,  # Un valor por defecto
                                "prioridad": s["prioridad"],
                            }
                            for s in df_configuraciones.filter(pl.col("IdEsc") == e["IdEsc"]).to_dicts()
                        ],
                    },
                }
            ]
            for e in horarios.to_dicts()
        }

        # No usamos Polars depues de esto
        if True:  # not DF_POLARS:
            self.atenciones = self.atenciones.to_pandas()

        return self.atenciones, self.planificacion
    
    
ID_DATABASE = 'V24_Fonav30'
# Adicionalmente disponibles: "V24_Provida", "V24_Fonav30", "V24_Cruz", "V24_Afpmodelo"
#V24_Provida: 13 -  'V24_Fonav30': 2 - V24_Afpmodelo: 1
ID_OFICINA = 2
FECHA = "2023-03-15"
DB_CONN = "mysql://autopago:Ttp-20238270@totalpackmysql.mysql.database.azure.com:3306/capacity_data_fonasa"

dataset = DatasetTTP(connection_string=DB_CONN, id_oficina=ID_OFICINA)



#%%
el_dia_real, plan  = dataset.un_dia(fecha=FECHA)
el_dia_real['T_Ate'] = (el_dia_real['FH_AteFin'] - el_dia_real['FH_AteIni']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real['T_Esp'] = (el_dia_real['FH_AteIni'] - el_dia_real['FH_Emi']).astype('timedelta64[s]').dt.total_seconds().astype(int)
el_dia_real = el_dia_real.sort_values(by='FH_Emi', inplace=False).astype(
    {
        'FH_Emi': 'datetime64[s]',
        'FH_Llama': 'datetime64[s]',
        'FH_AteIni': 'datetime64[s]',
        'FH_AteFin': 'datetime64[s]',})
######################
#------Simulacion-----
######################
import logging

import time
start_time           = time.time()
hora_cierre          = "17:30:00"
porcentaje_actividad =.8
#registros_atenciones_simulacion, fila = sim.simv7_1(
                                                    #el_dia_real, hora_cierre, 
un_dia                  =   el_dia_real
planificacion           =   sim.plan_desde_skills(skills                  =   sim.obtener_skills(el_dia_real),  # , 
                                                  inicio                  =  '08:00:00', #, 
                                                  porcentaje_actividad    =   porcentaje_actividad) #,
probabilidad_pausas     =     0.75 #,      # 'V24_Afpmodelo': .75 ,           #'V24_Provida' .7,            # V24_Fonav30:   0.8, 
factor_pausas           =    .020 #,        #'V24_Afpmodelo': 0.20,           #'V24_Provida' .2,            # V24_Fonav30:  .05,
params_pausas           =     [0, 1/3, 1] #,#'V24_Afpmodelo': [0, 1/3, 1]    #'V24_Provida' [0, 1/5, 1/4], #V24_Fonav30:   [0, 1/10, 1/2]
log_path                = "dev/simulacion.log"
un_dia["FH_AteIni"] = None
un_dia["FH_AteFin"] = None
un_dia["IdEsc"] = None

reloj = sim.reloj_rango_horario(str(un_dia.FH_Emi.min().time()), hora_cierre)
registros_atenciones = pd.DataFrame()
matcher_emision_reloj = sim.match_emisiones_reloj(un_dia)
supervisor = sim.Escritoriosv07(planificacion=planificacion, probabilidad_pausas=probabilidad_pausas, factor_pausas=factor_pausas, params_pausas=params_pausas)
registros_atenciones = pd.DataFrame()
fila = pd.DataFrame()

tiempo_total = (
    datetime.strptime(hora_cierre, "%H:%M:%S")
    - datetime.strptime(str(un_dia.FH_Emi.min().time()), "%H:%M:%S")
).total_seconds() / 60

logging.basicConfig(filename=log_path, level=logging.INFO, filemode="w")

for i, hora_actual in enumerate(reloj):
    total_mins_sim = i
    logging.info(
        f"--------------------------------NUEVA hora_actual {hora_actual}---------------------"
    )
    supervisor.aplicar_planificacion(
        hora_actual=hora_actual,
        planificacion=planificacion,
        tiempo_total=tiempo_total,
    )
    matcher_emision_reloj.match(hora_actual)

    if not matcher_emision_reloj.match_emisiones.empty:
        logging.info(f"nuevas emisiones")

        emisiones = matcher_emision_reloj.match_emisiones

        logging.info(
            f"hora_actual: {hora_actual} - series en emisiones: {list(emisiones['IdSerie'])}"
        )
        fila = pd.concat([fila, emisiones])
    else:
        logging.info(f"no hay nuevas emisiones hora_actual {hora_actual}")

    if supervisor.filtrar_x_estado("atención") or supervisor.filtrar_x_estado(
        "pausa"
    ):
        en_atencion = supervisor.filtrar_x_estado("atención") or []
        en_pausa = supervisor.filtrar_x_estado("pausa") or []
        escritorios_bloqueados = set(en_atencion + en_pausa)
        escritorios_bloqueados_conectados = [
            k
            for k, v in supervisor.escritorios_ON.items()
            if k in escritorios_bloqueados
        ]
        logging.info(
            f"iterar_escritorios_bloqueados: {escritorios_bloqueados_conectados}"
        )
        supervisor.iterar_escritorios_bloqueados(escritorios_bloqueados_conectados)

    if supervisor.filtrar_x_estado("disponible"):
        conectados_disponibles = [
            k
            for k, v in supervisor.escritorios_ON.items()
            if k in supervisor.filtrar_x_estado("disponible")
        ]
        logging.info(f"iterar_escritorios_disponibles: {conectados_disponibles}")
        logging.info(
            "tiempo_actual_disponible",
            {
                k: v["tiempo_actual_disponible"]
                for k, v in supervisor.escritorios_ON.items()
                if k in conectados_disponibles
            },
        )
        supervisor.iterar_escritorios_disponibles(conectados_disponibles)

        conectados_disponibles = sim.balancear_carga_escritorios(
            {
                k: {
                    "numero_de_atenciones": v["numero_de_atenciones"],
                    "tiempo_actual_disponible": v["tiempo_actual_disponible"],
                }
                for k, v in supervisor.escritorios_ON.items()
                if k in conectados_disponibles
            }
        )

        for un_escritorio in conectados_disponibles:
            logging.info(f"iterando en escritorio {un_escritorio}")

            configuracion_atencion = supervisor.escritorios_ON[un_escritorio][
                "configuracion_atencion"
            ]
            fila_filtrada = fila[
                fila["IdSerie"].isin(
                    supervisor.escritorios_ON[un_escritorio].get("skills", [])
                )
            ]  # filtrar_fila_por_skills(fila, supervisor.escritorios_ON[un_escritorio])

            if fila_filtrada.empty:
                continue
            elif configuracion_atencion == "FIFO":
                un_cliente = sim.FIFO(fila_filtrada)
                logging.info(f"Cliente seleccionado x FIFO {tuple(un_cliente)}")
            elif configuracion_atencion == "Rebalse":
                un_cliente = sim.extract_highest_priority_and_earliest_time_row(
                    fila_filtrada,
                    supervisor.escritorios_ON[un_escritorio].get("prioridades"),
                )
                logging.info(f"Cliente seleccionado x Rebalse {tuple(un_cliente)}")
            elif configuracion_atencion == "Alternancia":
                un_cliente = supervisor.escritorios_ON[un_escritorio][
                    "pasos_alternancia"
                ].buscar_cliente(fila_filtrada)
                logging.info(
                    f"Cliente seleccionado x Alternancia {tuple(un_cliente)}"
                )

            fila = sim.remove_selected_row(fila, un_cliente)
            logging.info(f"INICIANDO ATENCION de {tuple(un_cliente)}")
            supervisor.iniciar_atencion(un_escritorio, un_cliente)
            logging.info(
                f"numero_de_atenciones de escritorio {un_escritorio}: {supervisor.escritorios_ON[un_escritorio]['numero_de_atenciones']}"
            )
            logging.info(
                f"---escritorios disponibles: { supervisor.filtrar_x_estado('disponible')}"
            )
            logging.info(
                f"---escritorios en atención: { supervisor.filtrar_x_estado('atención')}"
            )
            logging.info(
                f"---escritorios en pausa: { supervisor.filtrar_x_estado('pausa')}"
            )

            un_cliente.IdEsc = int(un_escritorio)
            un_cliente.FH_AteIni = hora_actual
            registros_atenciones = pd.concat(
                [registros_atenciones, pd.DataFrame(un_cliente).T]
            )

    if i == 0:
        fila["espera"] = 0
    else:
        fila["espera"] += 1 * 60
logging.info(f"minutos simulados {total_mins_sim} minutos reales {tiempo_total}")
#                                                )
    #, log_path="dev/simulacion.log")
print(f"{len(registros_atenciones) = }, {len(fila) = }")
end_time = time.time()
print(f"tiempo total: {end_time - start_time:.1f} segundos")
sim.compare_historico_vs_simulacion(el_dia_real, registros_atenciones,  ID_DATABASE, ID_OFICINA,FECHA ,porcentaje_actividad)
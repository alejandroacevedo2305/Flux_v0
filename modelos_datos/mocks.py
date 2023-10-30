"""
Este es un contenedor para varios mocks de los modelos de datos, para probar cosas y no saturar los 
cuadernos principales y otros objetos. Tambien se usan para el testing de los objetos de modelos_datos
"""

sim_in = {
    "horario_oficina": ["08:00", "18:00"],
    "planificacion": {
        "1": {
            "configuraciones": [
                {
                    "inicio_fin": ["08:00", "18:00"],
                    "configuracion_atencion": {
                        "modo": 1, # FIFO
                        "jsonSerie": [{"IdSerie": 1, "Prioridad": 1, "Alterna": 1}]
                    }
                }
            ]
        },
        "2": {
            "configuraciones": [
                {
                    "inicio_fin": ["08:00", "18:00"],
                    "configuracion_atencion": {
                        "modo": 2, # Rebalse
                        "jsonSerie": [{"IdSerie": 1, "Prioridad": 2, "Alterna": 1},{"IdSerie": 2, "Prioridad": 2, "Alterna": 1}]
                    }
                }
            ]
        },
        "3": {
            "configuraciones": [
                {
                    "inicio_fin": ["08:00", "18:00"],
                    "configuracion_atencion": {
                        "modo": 0, # Alternancia
                        "jsonSerie": [{"IdSerie": 1, "Prioridad": 1, "Alterna": 5},{"IdSerie": 2, "Prioridad": 2, "Alterna": 3}]
                    }
                }
            ]
        }
    },
    "series": [
        {
            "idSerie": 1,
            "serie": "Atencion General",
            "tiempoMaximoEspera": 300,
            "SLA_objetivo": 0.5,
            "hI1": "07070707070707",
            "mI1": "00000000000000",
            "hF1": "12121212121212",
            "mF1": "00000000000000",
            "hI2": "12121212121212",
            "mI2": "00000000000000",
            "hF2": "22222222222221",
            "mF2": "00000000000000"
        },{
            "idSerie": 2,
            "serie": "Atencion Prioritaria",
            "tiempoMaximoEspera": 300,
            "SLA_objetivo": 0.8,
        },
        {
            "idSerie": 5,
            "serie": "Atencion Interesante",
            "tiempoMaximoEspera": 600,
            "SLA_objetivo": 0.35,
        },
        {
            "idSerie": 10,
            "serie": "Caja",
            "tiempoMaximoEspera": 300,
            "SLA_objetivo": 0.60,
        },
        {
            "idSerie": 11,
            "serie": "Atencion Interesante",
            "tiempoMaximoEspera": 600,
            "SLA_objetivo": 0.25,
        },
        {
            "idSerie": 12,
            "serie": "Caja",
            "tiempoMaximoEspera": 500,
            "SLA_objetivo": 0.35,
        },
        {
            "idSerie": 14,
            "serie": "Caja",
            "tiempoMaximoEspera": 300,
            "SLA_objetivo": 0.70,
        },
        {
            "idSerie": 17,
            "serie": "Caja",
            "tiempoMaximoEspera": 300,
            "SLA_objetivo": 0.60,
        }
    ]
}
"""
Ejemplo de Body para la API del simulador.  
Lo uso para probar `Input_Simulador(**sim_in)` (está en `modelos_datos.api.simulador`)
"""

sim_out = {
    "resumen" : {
        "T_Esp" : 1200,  # cummean de todas las atenciones
        "T_Ate" : 300,   # cummean de todas las atenciones
        "n_Eje" : 14,    # escritorios x 2, asumiendo dos turnos al dia
        "n_Ate" : 667,   # total de atenciones
        "sla_glob" : 0.9 # Weighted_avg por Series (tienen peso), podemos asumir que es igual
    },
    # "metadata" : {
    #     "sesion" : 1006,        # Identificador randong-gen
    #     "fecha" : "2023-01-01", # Viene desde el BODY
    #     "IdOficina" : 2         # Viene desde el BODY
    # },
    "sla_instantaneo": [
    {"FH": "2023-10-31T08:00:00", "IdSerie": 1, "n_Ate" : 10 ,"SLA": 0.97, "T_Ate" : 10, "T_Esp" : 8}, {"FH": "2023-10-31T08:00:00", "IdSerie": 2, "n_Ate" : 10 ,"SLA": 0.97, "T_Ate" : 2, "T_Esp" : 8},
    {"FH": "2023-10-31T09:00:00", "IdSerie": 1, "n_Ate" : 15 ,"SLA": 0.83, "T_Ate" : 15, "T_Esp" : 3}, {"FH": "2023-10-31T09:00:00", "IdSerie": 2, "n_Ate" : 15 ,"SLA": 0.83, "T_Ate" : 4, "T_Esp" : 3},
    {"FH": "2023-10-31T10:00:00", "IdSerie": 1, "n_Ate" : 21 ,"SLA": 0.81, "T_Ate" : 21, "T_Esp" : 1}, {"FH": "2023-10-31T10:00:00", "IdSerie": 2, "n_Ate" : 21 ,"SLA": 0.81, "T_Ate" : 8, "T_Esp" : 1},
    {"FH": "2023-10-31T11:00:00", "IdSerie": 1, "n_Ate" : 29 ,"SLA": 0.79, "T_Ate" : 29, "T_Esp" : 9}, {"FH": "2023-10-31T11:00:00", "IdSerie": 2, "n_Ate" : 29 ,"SLA": 0.79, "T_Ate" : 1, "T_Esp" : 9},
    {"FH": "2023-10-31T12:00:00", "IdSerie": 1, "n_Ate" : 35 ,"SLA": 0.52, "T_Ate" : 35, "T_Esp" : 2}, {"FH": "2023-10-31T12:00:00", "IdSerie": 2, "n_Ate" : 35 ,"SLA": 0.52, "T_Ate" : 5, "T_Esp" : 2},
    {"FH": "2023-10-31T13:00:00", "IdSerie": 1, "n_Ate" : 26 ,"SLA": 0.84, "T_Ate" : 26, "T_Esp" : 4}, {"FH": "2023-10-31T13:00:00", "IdSerie": 2, "n_Ate" : 26 ,"SLA": 0.84, "T_Ate" : 2, "T_Esp" : 4},
    {"FH": "2023-10-31T14:00:00", "IdSerie": 1, "n_Ate" : 29 ,"SLA": 0.87, "T_Ate" : 29, "T_Esp" : 7}, {"FH": "2023-10-31T14:00:00", "IdSerie": 2, "n_Ate" : 29 ,"SLA": 0.87, "T_Ate" : 8, "T_Esp" : 7},
    {"FH": "2023-10-31T15:00:00", "IdSerie": 1, "n_Ate" : 44 ,"SLA": 0.67, "T_Ate" : 44, "T_Esp" : 7}, {"FH": "2023-10-31T15:00:00", "IdSerie": 2, "n_Ate" : 44 ,"SLA": 0.67, "T_Ate" : 1, "T_Esp" : 7},
    {"FH": "2023-10-31T16:00:00", "IdSerie": 1, "n_Ate" : 44 ,"SLA": 0.47, "T_Ate" : 44, "T_Esp" : 7}, {"FH": "2023-10-31T16:00:00", "IdSerie": 2, "n_Ate" : 44 ,"SLA": 0.47, "T_Ate" : 1, "T_Esp" : 7},
    {"FH": "2023-10-31T17:00:00", "IdSerie": 1, "n_Ate" : 38 ,"SLA": 0.47, "T_Ate" : 38, "T_Esp" : 7}, {"FH": "2023-10-31T17:00:00", "IdSerie": 2, "n_Ate" : 38 ,"SLA": 0.47, "T_Ate" : 1, "T_Esp" : 7},
    {"FH": "2023-10-31T18:00:00", "IdSerie": 1, "n_Ate" : 18 ,"SLA": 0.57, "T_Ate" : 18, "T_Esp" : 7}, {"FH": "2023-10-31T18:00:00", "IdSerie": 2, "n_Ate" : 18 ,"SLA": 0.57, "T_Ate" : 1, "T_Esp" : 7},
    {"FH": "2023-10-31T19:00:00", "IdSerie": 1, "n_Ate" : 11 ,"SLA": 0.57, "T_Ate" : 11, "T_Esp" : 7}, {"FH": "2023-10-31T19:00:00", "IdSerie": 2, "n_Ate" : 11 ,"SLA": 0.57, "T_Ate" : 1, "T_Esp" : 7}
    ],
    "sla_cumulativo" : [
        {"IdSerie": 1, "SLA": 0.78, "n_Ate" : 300, "T_Ate" : 2, "T_Esp" : 8},
        {"IdSerie": 2, "SLA": 0.65, "n_Ate" : 300, "T_Ate" : 2, "T_Esp" : 8}
    ],
    "uso_escritorios" : [
    {"FH": "2023-10-31T08:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T08:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T09:00:00", "IdEsc": 1, "n_Ate" : 15 , "T_Ate" : 43 , "T_Inact" :   8, "T_Disp" :  9}, {"FH": "2023-10-31T09:00:00", "IdEsc": 2, "n_Ate" : 15 , "T_Ate" : 43 , "T_Inact" :   8, "T_Disp" :  9},
    {"FH": "2023-10-31T10:00:00", "IdEsc": 1, "n_Ate" : 21 , "T_Ate" : 33 , "T_Inact" :  15, "T_Disp" : 12}, {"FH": "2023-10-31T10:00:00", "IdEsc": 2, "n_Ate" : 11 , "T_Ate" : 33 , "T_Inact" :  15, "T_Disp" : 12},
    {"FH": "2023-10-31T11:00:00", "IdEsc": 1, "n_Ate" : 29 , "T_Ate" : 43 , "T_Inact" :  11, "T_Disp" :  6}, {"FH": "2023-10-31T11:00:00", "IdEsc": 2, "n_Ate" : 19 , "T_Ate" : 43 , "T_Inact" :  11, "T_Disp" :  6},
    {"FH": "2023-10-31T12:00:00", "IdEsc": 1, "n_Ate" : 21 , "T_Ate" : 33 , "T_Inact" :  15, "T_Disp" : 12}, {"FH": "2023-10-31T12:00:00", "IdEsc": 2, "n_Ate" : 11 , "T_Ate" : 33 , "T_Inact" :  15, "T_Disp" : 12},
    {"FH": "2023-10-31T13:00:00", "IdEsc": 1, "n_Ate" : 15 , "T_Ate" : 43 , "T_Inact" :   8, "T_Disp" :  9}, {"FH": "2023-10-31T13:00:00", "IdEsc": 2, "n_Ate" : 15 , "T_Ate" : 43 , "T_Inact" :   8, "T_Disp" :  9},
    {"FH": "2023-10-31T14:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T14:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T15:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T15:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T16:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T16:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T17:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T17:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T18:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T18:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26},
    {"FH": "2023-10-31T19:00:00", "IdEsc": 1, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}, {"FH": "2023-10-31T19:00:00", "IdEsc": 2, "n_Ate" : 10 , "T_Ate" : 23 , "T_Inact" :  11, "T_Disp" : 26}
    ]
}
"""
Salida de la API del simulador, en un formato relativamente facil de graficar para Luis et al. 
el servicio de consulta historica devuelve este mismo formato de salida serializado
"""

best_workforce = {
    "mejor_run": 22,
    "planificacion": {
        "1": {
            "configuraciones": [
                {
                    "inicio_fin": ["08:00", "18:00"],
                    "configuracion_atencion": {
                        "modo": 1,
                        "jsonSerie": [
                            {"IdSerie": 1, "Prioridad": 1, "Alterna": 1},
                            {"IdSerie": 2, "Prioridad": 2, "Alterna": 1},
                        ],
                    },
                }
            ]
        },
        "2": {
            "configuraciones": [
                {
                    "inicio_fin": ["08:00:00", "18:00:00"],
                    "configuracion_atencion": {
                        "modo": 2,
                        "jsonSerie": [
                            {"IdSerie": 2, "Prioridad": 1, "Alterna": 1},
                            {"IdSerie": 1, "Prioridad": 2, "Alterna": 1},
                        ],
                    },
                }
            ]
        },
    },
    "simulacion": sim_out,
}
"""
Ejemplo muy simple de como deberia ser el mejor Workforce, post-API de Workforce Management
"""

planificacion = {
 '1': [{'inicio': '13:00:00	', 'termino': '10:07:40',
   'propiedades': {'skills': [10],
    'configuracion_atencion': 'FIFO'}}],
 '2': [{'inicio': '08:40:00', 'termino': '18:40:00',
   'propiedades': {'skills': [5, 10, 11, 12],
    'configuracion_atencion': 'Rebalse'}}],
 '3': [{'inicio': '08:40:00', 'termino': '16:00:00',
   'propiedades': {'skills': [10], 
    'configuracion_atencion': 'FIFO'}}],
 '4': [{'inicio': '08:40:00', 'termino': '18:40:00',
   'propiedades': {'skills': [10, 12],
    'configuracion_atencion': 'FIFO'}}],
 '5': [{'inicio': '08:40:00', 'termino': '14:40:00',
   'propiedades': {'skills': [10, 12],
    'configuracion_atencion': 'FIFO'}}],
 '6': [{'inicio': '08:40:00', 'termino': '18:40:00',
   'propiedades': {'skills': [10, 11, 12],
    'configuracion_atencion': 'Alternancia'}}],
 '7': [{'inicio': '10:30:00', 'termino': '14:40:00',
   'propiedades': {'skills': [10],
    'configuracion_atencion': 'FIFO'}}],
 '8': [{'inicio': '08:40:00', 'termino': '12:40:00',
   'propiedades': {'skills': [5, 10, 11, 12],
    'configuracion_atencion': 'Rebalse'}}],
 '9': [{'inicio': '08:40:00', 'termino': '15:10:00',
   'propiedades': {'skills': [5, 10],
    'configuracion_atencion': 'FIFO'}}],
 '10': [{'inicio': '08:40:00', 'termino': '18:40:00',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO'}}],
 '11': [{'inicio': '08:40:00', 'termino': '14:40:00',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO'}}],
 '12': [{'inicio': '08:40:00', 'termino': '14:40:00',
   'propiedades': {'skills': [14, 17],
    'configuracion_atencion': 'FIFO'}}],
}
"""
Una planificacion para el simulador. No es igual a la planificacion de la API
Tomada de `Fonasa 02` el `2023-09-12`
"""
#%%
import optuna
from itertools import chain, combinations
import math
from src.simulador_v02 import *  
from src.gymnasium_utils import *  

un_dia   = forecast().un_mes().un_dia().df.sort_values(by='FH_Emi', inplace=False)

sucursal = 2
un_dia   = un_dia[un_dia.IdOficina == sucursal] #9 2
skills   = obtener_skills(un_dia)
series   = sorted(list({val for sublist in skills.values() for val in sublist}))
SLAs     = [(0.84, 10), (0.34, 25), (0.8, 33)]
niveles_servicio_x_serie = {s:random.choice(SLAs) for s in series}
modos_atenciones=["Alternancia", "FIFO", "Rebalse"]
max_escritorios = len(skills.keys())

prioridades = prioridad_x_serie(niveles_servicio_x_serie,2,1) 

action_space = get_action_space(
    skills     = skills, 
    config_set = modos_atenciones, 
    time_start = str(un_dia.FH_Emi.min().time()), 
    time_end   = str(un_dia.FH_Emi.max().time())
)

# plan = convert_output(action_space.sample(), skills_set = series, config_set = modos_atenciones)

# registros_SLA =   simular(plan, niveles_servicio_x_serie, un_dia, prioridades)  

 
from scipy.stats import gmean

#gmean(registros_SLA .drop("hora", axis=1).iloc[-1].dropna())

# Function to generate all non-empty subsets of a given list
def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))

# Optuna multi-objective function
def objective(trial, skills):
    try:
        bool_vector = [trial.suggest_categorical(f'escritorio_{i}', [True, False]) for i in range(len(skills.keys()))]  
        modos_atenciones = ["Alternancia", "FIFO", "Rebalse"]
        series = sorted(list({val for sublist in skills.values() for val in sublist}))
        str_dict = {i: trial.suggest_categorical('modo atención', modos_atenciones) for i in range(len(skills.keys()))}
        subset_dict = {i: trial.suggest_categorical('series', non_empty_subsets(series)) for _ in range(len(skills.keys()))}

        # Storing user attributes
        trial.set_user_attr('bool_vector', bool_vector)
        trial.set_user_attr('str_dict', str_dict)
        trial.set_user_attr('subset_dict', subset_dict)

        SLAs = [(0.84, 10), (0.34, 25), (0.8, 33)]
        niveles_servicio_x_serie = {s: random.choice(SLAs) for s in series}
        prioridades = prioridad_x_serie(niveles_servicio_x_serie, 2, 1) 
        
        action_space = get_action_space(
            skills={k: v for k, v, mask in zip(skills.keys(), skills.values(), bool_vector) if mask},
            config_set=modos_atenciones, 
            time_start=str(un_dia.FH_Emi.min().time()), 
            time_end=str(un_dia.FH_Emi.max().time())
        )
        
        plan = convert_output(action_space.sample(), skills_set=series, config_set=modos_atenciones)
        registros_SLA = simular(plan, niveles_servicio_x_serie, un_dia, prioridades)
        
        return gmean(registros_SLA.drop("hora", axis=1).iloc[-1].dropna()), sum(bool_vector)

    except Exception as e:
        print(f"An exception occurred: {e}")
        raise optuna.TrialPruned()


study = optuna.multi_objective.create_study(directions=['maximize', 'minimize'])
study.optimize(lambda trial: objective(trial, 
                                       skills = skills,
                                       #max_escritorios          = max_escritorios, 
                                       #series                   = series, 
                                       #modos_atenciones         = modos_atenciones,
                                       #niveles_servicio_x_serie = niveles_servicio_x_serie
                                       ), 
                                       n_trials                 = 20)

# Results: the first Pareto front (i.e., the best trade-offs between the two objectives)
pareto_front_trials = study.get_pareto_front_trials()
for i, trial in enumerate(pareto_front_trials):
    print(f"Trial {i+1}")
    print(f"Nivel de servicio global (maximize): {trial.values[0]}")
    print(f"Número de escritorios (minimize): {trial.values[1]}")
    print(f"Configuración: {trial.params}")


# %%

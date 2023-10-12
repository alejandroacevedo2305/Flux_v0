#%%
import optuna
from itertools import chain, combinations
import math

# Function to generate all non-empty subsets of a given list
def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))

# Function to perform mock transformations and compute a single objective value
def mock_transform(bool_vector, str_dict, subset_dict):
    filtered_str_dict = {k: v for k, v in str_dict.items() if bool_vector[k]}
    filtered_subset_dict = {k: v for k, v in subset_dict.items() if bool_vector[k]}
    
    transformed_str_dict = {k: v[::-1] for k, v in filtered_str_dict.items()}
    transformed_subset_dict = {k: sum(v) for k, v in filtered_subset_dict.items()}
    
    objective_value = sum(len(v) for v in transformed_str_dict.values()) + sum(v for v in transformed_subset_dict.values())
    
    return math.sqrt(objective_value)

# Optuna multi-objective function
def objective(trial, max_escritorios, series ,modos_atenciones):
    bool_vector = [trial.suggest_categorical(f'bool_{i}', [True, False]) for i in range(max_escritorios)]
    str_dict = {i: trial.suggest_categorical(f'str_{i}', modos_atenciones) for i in range(max_escritorios)}
    subset_dict = {i: trial.suggest_categorical(f'subset_{i}', non_empty_subsets(series)) for i in range(max_escritorios)}

    trial.set_user_attr('bool_vector', bool_vector)
    trial.set_user_attr('str_dict', str_dict)
    trial.set_user_attr('subset_dict', subset_dict)

    return mock_transform(bool_vector, str_dict, subset_dict), sum(bool_vector), sum(bool_vector)

study = optuna.multi_objective.create_study(directions=['maximize', 'minimize', 'minimize'])
study.optimize(lambda trial: objective(trial, 
                                       max_escritorios  = 3, 
                                       series           = [1, 11, 7, 9], 
                                       modos_atenciones = ["FIFO", "alternancia", "rebalse"]), 
                                       n_trials         = 10)

# Results: the first Pareto front (i.e., the best trade-offs between the two objectives)
pareto_front_trials = study.get_pareto_front_trials()
for i, trial in enumerate(pareto_front_trials):
    print(f"Trial {i+1}")
    print(f"Objective 1 (maximize): {trial.values[0]}")
    print(f"Objective 2 (minimize): {trial.values[1]}")
    print(f"Objective 3 (minimize): {trial.values[2]}")

    print(f"Parameters: {trial.params}")


# %%

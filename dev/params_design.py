#%%
import optuna

# Objective function to minimize
def objective(trial):
    # Define the length of the Boolean vector
    n = 10
    
    # Initialize the Boolean vector with Optuna's suggest_categorical() method
    bool_vector = [trial.suggest_categorical(f'bool_{i}', [True, False]) for i in range(n)]
    
    # Count the number of True entries in the Boolean vector
    count_true = sum(bool_vector)
    
    return count_true

# Create a study object and specify the direction is 'minimize'.
study = optuna.create_study(direction='minimize')

# Optimize the study, the objective function is passed in as the first argument
study.optimize(objective, n_trials=100)

# Results
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print(f'Value: {trial.value}')
print(f'Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')




# %%

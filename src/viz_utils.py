#%%
# Import standard libraries
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Import custom modules
from src.simulador_v02 import *  
from src.gymnasium_utils import *  
from src.datos_utils import *
from src.viz_utils import *
from src.optuna_utils import *

# Import optimization library
import optuna

# Compute count and average of 'espera' for given intervals
def sla_x_serie(df, interval='1H', corte=45, factor_conversion_T_esp:int=60):
    # Reset index and adjust data types
    df = df.reset_index(drop=False)
    df['FH_Emi'] = pd.to_datetime(df['FH_Emi'])  # Convert to datetime
    # More data type adjustments
    df['IdSerie'] = df['IdSerie'].astype(str)  # Convert to string
    df['espera'] = df['espera'].astype(float)  # Convert to float
    df['espera'] /= factor_conversion_T_esp
    # Set index for resampling
    df.set_index('FH_Emi', inplace=True)
    # Compute counts and averages
    df_count = df.resample(interval).size().reset_index(name='Count')
    def percentage_below_threshold(x):
        return (x < corte).mean() * 100
    df_percentage = df.resample(interval)['espera'].apply(percentage_below_threshold).reset_index(name='espera')
    return df_count, df_percentage

# Plotting function for a single subplot
def plot_count_and_avg(df_count, df_avg, ax1, color_sla='navy', serie='', tipo_SLA='SLA'):
    # Generate x-labels and plot bars and lines
    x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(df_count['FH_Emi'].dt.strftime('%H:%M:%S'), (df_count['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S'))]
    bars = ax1.bar(x_labels, df_count['Count'], alpha=0.6, label='Demanda')
    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(x_labels, df_avg['espera'], color=color_sla, marker='o', label=f'{tipo_SLA}')
    # Set labels and legends
    ax1.set_xlabel('')
    ax1.set_ylabel('Demanda (#)', color='black')
    ax2.set_ylabel(f'{tipo_SLA} (%)', color='black')    
    ax2.set_ylim([0, 105])
    ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
    ax1.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor", size =7)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 1.22))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 1.22))
    ax1.set_title(f"Serie {serie}", y=1.15) 
    # Add grid and background color
    ax1.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax2.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax1.set_facecolor((0.75, 0.75, 0.75, 0.85))  
    ax2.set_facecolor((0.75, 0.75, 0.75, 0.85))  

# Plotting function for all subplots
def plot_all_reports(df_pairs, tipo_SLA, color_sla, n_rows=3, n_cols=2, heigth=10, width=10):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, heigth))  
    axs = axs.ravel()  
    for i, ((df_count, df_avg), serie) in enumerate(df_pairs):
        plot_count_and_avg(df_count, df_avg, axs[i], serie=serie, tipo_SLA=tipo_SLA, color_sla=color_sla)
    fig.subplots_adjust(hspace=1, wspace=.45)  
    plt.show()

########################################################################
##----------------------Datos históricos de un día----------------
########################################################################

dataset = DatasetTTP.desde_csv_atenciones("data/fonasa_monjitas.csv.gz")
el_dia_real = dataset.un_dia("2023-05-15").sort_values(by='FH_Emi', inplace=False)
skills = obtener_skills(el_dia_real)
series = sorted(list({val for sublist in skills.values() for val in sublist}))
registros_atenciones = pd.DataFrame()
tabla_atenciones = el_dia_real[['FH_Emi', 'IdSerie', 'T_Esp']]
tabla_atenciones.columns = ['FH_Emi', 'IdSerie', 'espera']
registros_atenciones = tabla_atenciones.copy()
registros_atenciones['IdSerie'] = registros_atenciones['IdSerie'].astype(int)
registros_x_serie = [registros_atenciones[registros_atenciones.IdSerie == s] for s in series]
df_pairs = [(sla_x_serie(r_x_s, '1H', corte=20), s) for r_x_s, s in zip(registros_x_serie, series)]

# Plotting
plot_all_reports([df_pairs[idx] for idx in [0, 1, 2, 3, 4, 5]], tipo_SLA="SLA histórico", color_sla="navy", n_rows=3, n_cols=2, heigth=10, width=12)

#%%
""" 
The code produce a plot like the one I attached. I need the function "plot_all_reports" to take two arguments 
like [df_pairs[idx] for idx in [0,1,2,3,4,5]] and within each subplot plot both lines. 
Make other two parameters to set the legend label and color of the second line. also add a argument to put a main title
the function must be like this:

plot_all_reports_two_lines( df_pairs_1 = [df_pairs[idx] for idx in [0,1,2,3,4,5]], 
                            df_pairs_2 = [df_pairs[idx] for idx in [5,4,3,2,1,0]], 
                            label_1="SLA histórico", label_2="SLA IA", 
                            color_1="navy", color_2="purple", 
                            n_rows=3, n_cols=2, height=10, width=12, main_title = 'main_title'):

"""
#%%
# Modified plotting function for a single subplot
def plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, ax1, label_1, label_2, color_1, color_2, serie=''):
    # Debugging statement:
    print(f'Type of df_count_1: {type(df_count_1)}')
    # Ensure df_count_1 is a dataframe
    if not isinstance(df_count_1, pd.DataFrame):
        raise TypeError("df_count_1 is not a DataFrame!")

    # Extract start and end times
    start_times = df_count_1['FH_Emi'].dt.strftime('%H:%M:%S')
    end_times = (df_count_1['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S')

    # Create the x_labels using the extracted start and end times
    x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(start_times, end_times)]
    
    bars = ax1.bar(x_labels, df_count_1['Count'], alpha=0.6, label='Demanda')
    # ... Rest of the function


    # Create a second y-axis
    ax2 = ax1.twinx()
    ##############
        # Ensure the data lengths match before plotting
    min_len_1 = min(len(x_labels), len(df_avg_1['espera']))
    min_len_2 = min(len(x_labels), len(df_avg_2['espera']))
    
    # Use the minimum length to slice the data and plot
    ax2.plot(x_labels[:min_len_1], df_avg_1['espera'][:min_len_1], color=color_1, marker='o', label=label_1)
    ax2.plot(x_labels[:min_len_2], df_avg_2['espera'][:min_len_2], color=color_2, marker='o', linestyle='--', label=label_2)

    ############
    # ax2 = ax1.twinx()
    # ax2.plot(x_labels, df_avg_1['espera'], color=color_1, marker='o', label=label_1)
    # ax2.plot(x_labels, df_avg_2['espera'], color=color_2, marker='o', linestyle='--', label=label_2)

    # Set labels and legends
    ax1.set_xlabel('')
    ax1.set_ylabel('Demanda (#)', color='black')
    ax2.set_ylabel('SLA (%)', color='black')    
    ax2.set_ylim([0, 105])
    ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
    ax1.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor", size =7)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 1.22))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 1.22))
    ax1.set_title(f"Serie {serie}", y=1.15) 

    # Add grid and background color
    ax1.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax2.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  
    ax1.set_facecolor((0.75, 0.75, 0.75, 0.85))  
    ax2.set_facecolor((0.75, 0.75, 0.75, 0.85))  

# Plotting function for all subplots with two lines
def plot_all_reports_two_lines(df_pairs_1, df_pairs_2, label_1, label_2, color_1, color_2, n_rows, n_cols, height, width, main_title):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    axs = axs.ravel()  
    for i, (pair_1, pair_2) in enumerate(zip(df_pairs_1, df_pairs_2)):
        # Unpacking the tuple correctly
        (df_count_1, df_avg_1), _ = pair_1
        (df_count_2, df_avg_2), _ = pair_2
        
        plot_count_and_avg_two_lines(df_count_1, df_avg_1, df_count_2, df_avg_2, axs[i], label_1, label_2, color_1, color_2, serie='')
# ... [rest of the function]

    fig.subplots_adjust(hspace=1, wspace=.45)  
    fig.suptitle(main_title, y=1.07, fontsize=16)
    plt.show()


# Usage:
# (You can call the function using the signature you provided)
plot_all_reports_two_lines(df_pairs_1 = [df_pairs[idx] for idx in [0,1,2,3,4,5]], 
                           df_pairs_2 = [df_pairs[idx] for idx in [5,4,3,2,1,0]], 
                           label_1="SLA histórico", label_2="SLA IA", 
                           color_1="navy", color_2="purple", 
                           n_rows=3, n_cols=2, height=10, width=12, main_title='main_title')



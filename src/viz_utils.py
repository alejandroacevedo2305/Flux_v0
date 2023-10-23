#%%

import matplotlib.pyplot as plt
import pandas as pd

def sla_x_serie(df, interval='1H', corte=45, factor_conversion_T_esp:int=60):
    df = df.reset_index(drop=False)
    df['FH_Emi'] = pd.to_datetime(df['FH_Emi'])  # Convert to datetime
    df['IdSerie'] = df['IdSerie'].astype(str)  # Ensuring IdSerie is string
    df['espera'] = df['espera'].astype(float)  # Convert to float
    df['espera'] = df['espera']/factor_conversion_T_esp  
    # Set FH_Emi as the index for resampling
    df.set_index('FH_Emi', inplace=True)
    # First DataFrame: Count of events in each interval
    df_count = df.resample(interval).size().reset_index(name='Count')
    # Second DataFrame: Percentage of "espera" values below the threshold
    def percentage_below_threshold(x):
        return (x < corte).mean() * 100
    df_percentage = df.resample(interval)['espera'].apply(percentage_below_threshold).reset_index(name='espera')
    
    return df_count, df_percentage

def plot_count_and_avg(df_count, df_avg, ax1, color_sla:str='navy', serie:str="_", tipo_SLA:str="SLA"):
    x_labels = [f"{start_time} - {end_time}" for start_time, end_time in zip(df_count['FH_Emi'].dt.strftime('%H:%M:%S'), (df_count['FH_Emi'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M:%S'))]
    bars = ax1.bar(x_labels, df_count['Count'], alpha=0.6, label='Demanda')
    ax2 = ax1.twinx()
    ax2.plot(x_labels, df_avg['espera'], color=color_sla, marker='o', label=f'{tipo_SLA}')
    ax1.set_xlabel('')
    ax1.set_ylabel('Demanda (#)', color='black')
    ax2.set_ylabel(f'{tipo_SLA} (%)', color='black')    
    ax2.set_ylim([0, 105])
    ax1.set_xticks([rect.get_x() + rect.get_width() / 2 for rect in bars])
    ax1.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor", size =7)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.35, 1.22))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 1.22))
    ax1.set_title(f"Serie {serie}", y=1.15) 

    # New code to add a grey grid and light grey background color
    ax1.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  # Adding grey grid to ax1
    ax2.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.35)  # Adding grey grid to ax2
    ax1.set_facecolor((0.75, 0.75, 0.75, 0.85))  # Adding light grey background color to ax1
    ax2.set_facecolor((0.75, 0.75, 0.75, 0.85))  # Adding light grey background color to ax2

    
def plot_all_reports(df_pairs, tipo_SLA, color_sla, n_rows:int=3, n_cols:int=2, heigth:float=10, width:float=10):
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, heigth))  # 2x3 grid of subplots
    axs = axs.ravel()  # Flatten the grid to easily loop through it
    for i, ((df_count, df_avg), serie) in enumerate(df_pairs):
        plot_count_and_avg(df_count, df_avg, axs[i], serie = serie, tipo_SLA= tipo_SLA, color_sla=color_sla)
    fig.subplots_adjust(hspace=1, wspace=.45)  # Adjusts the vertical space between subplots. Default is 0.2.
    #plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
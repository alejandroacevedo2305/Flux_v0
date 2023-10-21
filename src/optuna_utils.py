#%%
from itertools import chain, combinations
from scipy import stats
import numpy as np
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

def calculate_geometric_mean(series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return stats.gmean(series)  
def extract_skills_length(data):
    result = {}
    
    # Initialize a variable to store the sum of all lengths.
    total_length = 0
    
    # Iterate over keys and values in the input dictionary.
    for key, entries in data.items():
        # Initialize an empty list to store the lengths for this key.
        lengths_for_key = []
        
        # Iterate over each entry which is a dictionary.
        for entry in entries:
            # Access the 'skills' field, and calculate its length.
            skills_length = len(entry['propiedades']['skills'])
            
            # Add the current length to the total_length.
            total_length += skills_length
            
            # Append this length to the list for this key.
            lengths_for_key.append(skills_length)
        
        # Store the list of lengths in the result dictionary, using the same key.
        result[key] = lengths_for_key
    
    # Return the result dictionary and the total sum of all lengths.
    return total_length #, result
def non_empty_subsets(lst):
    return list(chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1)))
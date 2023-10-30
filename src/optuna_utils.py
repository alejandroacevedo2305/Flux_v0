#%%
from itertools import chain, combinations
from scipy import stats
import numpy as np
import pandas as pd
import optuna
from scipy.stats.mstats import gmean
import random

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
def calcular_optimo_max_min(multi_obj):
    return multi_obj[0]/(multi_obj[1]) # max_SLA/(min_n_espera)
def calcular_optimo(multi_obj):
    return multi_obj[0]/(multi_obj[1]+multi_obj[2]) # max_SLA/(min_Esc + min_Skills)
def extract_max_value_keys(input_dict):
    output_dict = {}  # Initialize an empty dictionary to store the result
    # Loop through each item in the input dictionary
    for workforce, values_dict in input_dict.items():
        max_key = max(values_dict, key=values_dict.get)  # Find the key with the maximum value in values_dict
        max_value = values_dict[max_key]  # Get the maximum value
        output_dict[workforce] = (max_key, max_value)  # Add the key and value to the output dictionary
    return output_dict  # Return the output dictionary
def calculate_geometric_mean(series, weights=None):
    series = series.dropna()
    return np.nan if series.empty else gmean(series, weights=weights)
def plan_unico(lst_of_dicts):
    new_list = []    
    # Initialize a counter for the new globally unique keys
    global_counter = 0    
    # Loop through each dictionary in the original list
    for dct in lst_of_dicts:        
        # Initialize an empty dictionary to hold the key-value pairs of the original dictionary but with new keys
        new_dct = {}        
        # Loop through each key-value pair in the original dictionary
        for key, value in dct.items():            
            # Assign the value to a new key in the new dictionary
            new_dct[global_counter] = value            
            # Increment the global counter for the next key
            global_counter += 1        
        # Append the newly created dictionary to the list
        new_list.append(new_dct)
        
    return {str(k): v for d in new_list for k, v in d.items()}


def get_time_intervals(df, n, porcentaje_actividad:float=1):
    assert porcentaje_actividad <= 1 
    assert porcentaje_actividad > 0 

    # Step 1: Find the minimum and maximum times from the FH_Emi column
    min_time = df['FH_Emi'].min()
    max_time = df['FH_Emi'].max()    
    # Step 2: Calculate the total time span
    total_span = max_time - min_time    
    # Step 3: Divide this span by n to get the length of each interval
    interval_length = total_span / n    
    # Step 4: Create the intervals
    intervals = [(min_time + i*interval_length, min_time + (i+1)*interval_length) for i in range(n)]    
    # New Step: Adjust the start time of each interval based on the percentage input
    adjusted_intervals = [(start_time + 1 * (1 - porcentaje_actividad) * (end_time - start_time), end_time) for start_time, end_time in intervals]
    # Step 5: Format the intervals as requested
    formatted_intervals = [(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')) for start_time, end_time in adjusted_intervals]
    
    return formatted_intervals
def partition_dataframe_by_time_intervals(df, intervals):
    partitions = []    
    # Loop over each time interval to create a partition
    for start, end in intervals:
        # Convert the time strings to Pandas time objects
        start_time = pd.to_datetime(start).time()
        end_time = pd.to_datetime(end).time()        
        # Create a mask for filtering the DataFrame based on the time interval
        mask = (df['FH_Emi'].dt.time >= start_time) & (df['FH_Emi'].dt.time <= end_time)        
        # Apply the mask to the DataFrame to get the partition and append to the list
        partitions.append(df[mask])        
    return partitions

def get_random_non_empty_subset(lst):
    """
    Get a random non-empty subset from a list.

    Parameters:
    lst (list): The list from which to generate a subset.

    Returns:
    list: A random non-empty subset of the input list.
    """
    # Step 1: Check that the input list is not empty
    if not lst:
        return "Input list is empty, cannot generate a subset."
    
    # Step 2: Generate a random integer for the size of the subset.
    # Since we want a non-empty subset, we select a size between 1 and len(lst).
    subset_size = random.randint(1, len(lst))
    
    # Step 3: Randomly select 'subset_size' unique elements from the list
    return random.sample(lst, subset_size)

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


def extract_min_value_keys(input_dict):
    output_dict = {}  # Initialize an empty dictionary to store the result
    # Loop through each item in the input dictionary
    for workforce, values_dict in input_dict.items():
        max_key = min(values_dict, key=values_dict.get)  # Find the key with the maximum value in values_dict
        max_value = values_dict[max_key]  # Get the maximum value
        output_dict[workforce] = (max_key, max_value)  # Add the key and value to the output dictionary
    return output_dict  # Return the output dictionary

if __name__ == "__main__":
    #  
    pass



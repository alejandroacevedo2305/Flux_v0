#%%
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from itertools import count, islice
import numpy as np
import logging
import random
from gymnasium.spaces import Discrete, Box, MultiBinary
from gymnasium import Space
import gymnasium as gym
import numpy as np

def time_to_seconds(time_str):
    """Convert HH:MM:SS time format to seconds from the start of the day."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s
class SegmentSpace(Space):
    def __init__(self, low, high, dtype=np.float32):
        super(SegmentSpace, self).__init__(shape=(2,), dtype=dtype)
        self.low = np.array([low, low], dtype=dtype)
        self.high = np.array([high, high], dtype=dtype)
        self.inner_space = Box(low=low, high=high, shape=(1,), dtype=dtype)
        
    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        point1 = self.inner_space.sample()[0]
        point2 = self.inner_space.sample()[0]
        
        # Ensure that point1 and point2 are different
        while point1 == point2:
            point2 = self.inner_space.sample()[0]
        segment = np.sort([point1, point2])
        return segment.astype(self.dtype)
    
    def contains(self, x):
        return self.inner_space.contains(x[0]) and self.inner_space.contains(x[1]) and x[0] < x[1]
      
class MultiBinaryNonEmpty(MultiBinary):
    def sample(self):
        sample = super().sample()
        if np.all(sample == 0):  # If all zeros
            idx = np.random.randint(0, len(sample))  # Pick a random index
            sample[idx] = 1  # Set that bit to 1 to ensure non-emptiness
        return sample

def convert_output(raw_output, skills_set, config_set):
    # Create a regular dictionary to hold the converted data
    converted_output = {}    
    for key, entries in raw_output.items():
        converted_entries = []        
        for entry in entries:
            # Convert skills using a boolean mask
            actual_skills = [skills_set[i] 
                             for i, is_present in enumerate(entry['propiedades']['skills']) if is_present]            
            # Convert configuracion_atencion
            actual_config = config_set[entry['propiedades']['configuracion_atencion']]            
            # Convert inicio and termino to time format
            actual_inicio =  '08:00:00'#(int(entry['tramo'][0]))
            actual_termino = '14:00:00'#seconds_to_time(int(entry['tramo'][1]))
            # Create the converted entry
            converted_entry = {
                'inicio': actual_inicio,
                'termino': actual_termino,
                'propiedades': {
                    'skills': actual_skills,
                    'configuracion_atencion': actual_config
                }
            }            
            converted_entries.append(converted_entry)        
        converted_output[key] = converted_entries    
    return converted_output
def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
def get_action_space(skills, config_set, time_start, time_end):  # sourcery skip: inline-immediately-returned-variable
    time_start_seconds = time_to_seconds(time_start)
    time_end_seconds = time_to_seconds(time_end)
    return gym.spaces.Dict({escritorio: gym.spaces.Tuple([
                                gym.spaces.Dict({
                                #'tramo':gym.spaces.Box(low= time_start_seconds, high=time_end_seconds), #SegmentSpace(low=time_start_seconds, high=time_end_seconds, dtype=int),
                                'propiedades': gym.spaces.Dict({
                                                            'skills': MultiBinaryNonEmpty(len(list_series)),  # Create a binary mask for skills
                                                            'configuracion_atencion': Discrete(len(config_set))  # 3 different configurations
                                })
                                })                                    
                                for _ in range(1)]) for escritorio,list_series in skills.items() })
# %%

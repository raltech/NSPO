import numpy as np
import brian2 as b2
from objective_func import stimulate, f
from experiment_params import *

def random_exploration(t_max):
    score_history = []
    for t in range(t_max):
        stim_pattern = np.random.uniform(-10, 10, num_neurons) * b2.nA
        statemon, spikemon, ratemon = stimulate(synapse_w, stim_pattern)
        score_history.append(f(np.array(spikemon.count), target_pattern))
        # print(np.array(spikemon.count), target_pattern, f(np.array(spikemon.count), target_pattern))
    return stim_pattern, score_history
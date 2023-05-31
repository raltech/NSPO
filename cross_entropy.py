import numpy as np
import brian2 as b2
from objective_func import stimulate, f
from experiment_params import *

def cross_entropy_method(t_max, sigma=1.0):
    # Initialize parameters
    num_samples = 10
    elite_ratio = 0.2
    num_elite = int(num_samples * elite_ratio)
    mean = np.array([0] * num_neurons)
    cov = np.eye(num_neurons)

    score_history = []

    for t in range(t_max):
        # Generate candidate solutions
        samples = np.random.multivariate_normal(mean, cov, num_samples)

        print("Generation: ", t)
        print(samples)

        # Evaluate objective function and constraints
        scores = []
        for x in samples:
            stim_pattern = x * b2.nA
            statemon, spikemon, ratemon = stimulate(synapse_w, stim_pattern)
            scores.append(f(np.array(spikemon.count), target_pattern))
            print(np.array(spikemon.count), target_pattern, f(np.array(spikemon.count), target_pattern))
  
        # Update probability distribution based on best solutions
        elite_indices = np.argsort(scores)[:num_elite]
        elite_samples = samples[elite_indices]
        mean = np.mean(elite_samples, axis=0)
        elite_samples = (elite_samples - mean) / np.std(elite_samples, axis=0)
        cov = np.cov(elite_samples, rowvar=False)
        cov = np.diag(np.diag(cov)) * sigma + cov
        score_history.append(np.mean(scores))

    # import pdb; pdb.set_trace()
    x_best = mean
    return x_best, score_history
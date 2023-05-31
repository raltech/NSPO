import numpy as np
import random
import brian2 as b2
from objective_func import stimulate, f
from experiment_params import *

def genetic_algorithm(t_max, m, k, crossover_rate, mutation_rate, sigma):
    population = np.random.normal(0, sigma, (m, num_neurons))
    score_history = []
    for t in range(t_max):
        print("Generation: ", t)
        print(population)
        parents, done, values = truncate_selection(target_pattern, population, k)
        score_history.append(np.mean(values))
        if done:
            return parents[0], score_history

        population = []
        while len(population) < m:
            if np.random.rand() < crossover_rate:
                if len(parents) < 2:
                    x, y = parents[0], parents[0]
                else:
                    x, y = random.sample(parents, 2)
                child = single_point_crossover(x, y)
                child = gaussian_mutate(child, mutation_rate, sigma)
                population.append(child)
        population = np.array(population)
    best, done, values = truncate_selection(target_pattern, population, 1) 
    score_history.append(np.mean(values))
    return best, score_history

# select top k individuals
def truncate_selection(target_pattern, population, k):
    done = False
    values = []
    for individual in population:
        stim_pattern = individual * b2.nA
        statemon, spikemon, ratemon = stimulate(synapse_w, stim_pattern)
        values.append(f(np.array(spikemon.count), target_pattern))
        print(np.array(spikemon.count), target_pattern, f(np.array(spikemon.count), target_pattern))
    combined = list(zip(values, population))
    # import pdb; pdb.set_trace()
    sorted_combined = sorted(combined, key=lambda x: x[0])
    population = [item[1] for item in sorted_combined]
    if np.min(values) == 0:
        done = True
    return population[:k], done, values

def single_point_crossover(x, y):
    point = np.random.randint(len(x))
    # import pdb; pdb.set_trace()
    return np.concatenate((x[:point], y[point:]))

def gaussian_mutate(x, mutation_rate, sigma):
    for idx in range(len(x)):
        if np.random.rand() < mutation_rate:
            x[idx] += np.random.normal(0, sigma)
    return x
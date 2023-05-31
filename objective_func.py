import numpy as np
from brian2 import *
from experiment_params import *

def f(spike_pattern, target_pattern):
    return np.sum(np.abs(spike_pattern - target_pattern))
    # return np.sum(spike_pattern != target_pattern)
    

def stimulate(synapse_w, stim_pattern):
    # Your setup code
    # start_scope()
    # Constants for the HH model
    # area = 20000*umetre**2
    area = 0.004*mmetre**2 # (https://www.nature.com/articles/nature26159)
    Cm = 1*ufarad*cm**-2 * area
    g_kd = 30*msiemens*cm**-2 * area  # Maximum permeability for Potassium
    g_na = 100*msiemens*cm**-2 * area  # Maximum permeability for Sodium
    gl = 5e-5*siemens*cm**-2 * area  # Leak conductance
    EK = -77*mV  # Potassium potential
    ENa = 55*mV  # Sodium potential
    El = -65*mV  # Leak potential
    VT = -63*mV  # Threshold potential

    duration = 20*ms
    stim_start_time = 2*ms
    stim_end_time = 3*ms
    num_samples = int(duration/defaultclock.dt)
    tmp = zeros((num_samples,num_neurons)) * nA  # Ensure the correct units
    for i in range(num_neurons):
        I_arr = zeros(num_samples) * nA
        start_sample_idx = int(stim_start_time/defaultclock.dt)
        end_sample_idx =  int(stim_end_time/defaultclock.dt)
        I_arr[start_sample_idx:end_sample_idx] = stim_pattern[i]
        tmp[:,i] = I_arr
    I_recorded = TimedArray(tmp, dt=defaultclock.dt)


    # HH Equations
    eqs = '''
    dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
    dm/dt = 0.32*(mV**-1)*4*mV/exprel((13.*mV-v+VT)/(4*mV))/ms*(1-m)-0.28*(mV**-1)*5*mV/exprel((v-VT-40.*mV)/(5*mV))/ms*m : 1
    dn/dt = 0.032*(mV**-1)*5*mV/exprel((15.*mV-v+VT)/(5*mV))/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
    dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
    I = I_recorded(t,i) : amp
    '''

    # The reset clause has to be removed as the model is not a threshold-based model
    # G = NeuronGroup(num_neurons, eqs, threshold='v>-40*mV', refractory='v>-40*mV', method='exponential_euler')
    G = NeuronGroup(num_neurons, eqs, threshold='v>-40*mV', refractory=5*ms, method='exponential_euler')
    G.v = El

    # Synapses
    S = Synapses(G, G, 'w : volt', on_pre='v_post += w')
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                S.connect(i=i, j=j)
                S.w[i, j] = synapse_w[i][j] * mV  # assuming weights are given in mV
            
    statemon = StateMonitor(G, variables=True, record=True)
    spikemon = SpikeMonitor(G)
    ratemon = PopulationRateMonitor(G)

    run(duration)

    # for i in range(num_neurons):
    #     # print("Neuron ", i, " firing rate: ", spikemon.count[i]/duration)
    #     print((stim_end_time - stim_start_time)*stim_pattern[i]/area)

    return statemon, spikemon, ratemon
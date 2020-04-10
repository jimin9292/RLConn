
# coding: utf-8

###################
#UTILITY FUNCTIONS#
###################

import json
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

from RLConn import neural_params as n_params
from RLConn import network_sim

def load_Json(filename):

    with open(filename) as content:

        content = json.load(content)

    return content

def construct_dyn_inputmat(t0, tf, dt, input_type, neuron_indices, normalized_amps = False, freqs = False, noise_amplitudes = False, step_time_interval = False):

    timepoints = np.arange(t0, tf, dt)
    input_mat = np.zeros((len(timepoints) + 1, n_params.default['N']))

    if input_type == 'sinusoidal':
        
        amps = np.asarray(normalized_amps) / 2.

        for i in range(len(neuron_indices)):

            for j in range(len(timepoints)):
                
                input_mat[j, neuron_indices[i]] = amps[i] * np.sin(freqs[i] * timepoints[j]) + amps[i]

    elif input_type == 'noisy':

        for k in range(len(neuron_indices)):

            noise = 10**(-2)*np.random.normal(0, noise_amplitudes[k], len(input_mat))
            input_mat[:, neuron_indices[k]] = normalized_amps[k] + noise

    #elif input_type == 'step':

    #    for k in range(len(neuron_indices)):

    #        step_start = step_time_interval[k, 0]
    #        step_end = step_time_interval[k, 1]

    #        step_start_ind = step_start / dt
    #        step_end_ind = step_end / dt
    #        step_start_ind = int(step_start_ind)
    #        step_end_ind = int(step_end_ind)

    #        input_mat[step_start_ind:step_end_ind, neuron_indices[k]] = normalized_amps[k]

    return input_mat

def redblue(m):

    m1 = m * 0.5
    r = np.divide(np.arange(0, m1)[:, np.newaxis], np.max([m1-1,1]))
    g = r
    r = np.vstack([r, np.ones((int(m1), 1))])
    g = np.vstack([g, np.flipud(g)])
    b = np.flipud(r)
    x = np.linspace(0, 1, m)[:, np.newaxis]

    red = np.hstack([x, r, r])
    green = np.hstack([x, g, g])
    blue = np.hstack([x, b, b])

    red_tuple = tuple(map(tuple, red))
    green_tuple = tuple(map(tuple, green))
    blue_tuple = tuple(map(tuple, blue))

    cdict = {
    	'red': red_tuple,
        'green': green_tuple,
        'blue': blue_tuple
        }

    return cdict

def project_v_onto_u(v, u):
    
    factor = np.divide(np.dot(u, v), np.power(np.linalg.norm(u), 2))
    projected = factor * u
    
    return projected

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), spstats.sem(a)
    h = se * spstats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h

def continuous_transition_scaler(old, new, t, rate, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def compute_action_combinations(action_2_delweight_space, num_modifiable_weights):

    import itertools

    actions = np.asarray(action_2_stim_space)
    action_combs = np.asarray([p for p in itertools.product(actions, repeat=num_neurons)])

    return action_combs

def convert_syn_2_vec(M):

    M_vec = np.matrix.flatten(M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1))

    return M_vec

def convert_gap_2_vec(M):

    M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1)

def compute_score(Gg, Gs, E, 
                    input_vec, ablation_mask, 
                    tf, t_delta, cutoff_1, cutoff_2,
                    plot_result = True):

    # Construct network dict

    network_dict = {

    "gap" : Gg,
    "syn" : Gs,
    "directionality" : E

    }

    # Initialize model with the network dict

    network_sim.initialize_params_neural()
    network_sim.initialize_connectivity(network_dict)

    # Simulate network with given input_vec and ablation mask

    network_result_dict = network_sim.run_network_constinput_RL(0, tf, t_delta, 
                                                               input_vec=input_vec,
                                                               ablation_mask=ablation_mask)

    plt.plot(network_result_dict['v_solution'][100:, :])

    # Obtain test modes using SVD 

    v_solution_truncated = network_result_dict['v_solution'][100:, :]
    u,s,v = np.linalg.svd(v_solution_truncated.T)

    m1_test = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 0]
    m2_test = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 1]

    # Compute the error

    m1_target = n_params.m1_target
    m2_target = n_params.m2_target

    m1_diff = np.subtract(m1_target, m1_test)
    m2_diff = np.subtract(m2_target, m2_test)

    m_joined = np.vstack([m1_diff, m2_diff])
    errors = np.sqrt(np.power(m_joined, 2).sum(axis = 0))

    mean_error = np.mean(errors)
    sum_error = np.sum(errors)

    # Plot the target vs test

    if plot_result == True:

        plt.figure(figsize=(5.5,5))

        plt.scatter(m1_target, m2_target, s = 0.75, color = 'black')
        plt.scatter(m1_test, m2_test, s = 0.75, color = 'red')

        plt.ylim(-25, 25)
        plt.xlim(-25, 25)

    return mean_error, sum_error













# coding: utf-8

###################
#UTILITY FUNCTIONS#
###################

import json
import numpy as np
import scipy.stats as spstats
import itertools
import matplotlib.pyplot as plt

from RLConn import neural_params as n_params
from RLConn import network_sim
from RLConn import problem_definitions as problems

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

def compute_action_combinations(del_W_space, num_modifiable_weights):

    actions = np.asarray(del_W_space)
    action_combs = np.asarray([p for p in itertools.product(actions, repeat=num_modifiable_weights)])

    return action_combs

def compute_possible_pairs(num_neurons):

    return list(itertools.combinations(range(num_neurons), 2))

def convert_conn_2_vec(M):

    M_vec = np.matrix.flatten(M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1))

    return M_vec

def update_weight_gap(Gg, neuron_from, neuron_to, action_bidirectional, weight_min, weight_max):

    updated_Gg = Gg.copy()

    new_weight = Gg[neuron_from, neuron_to] + action_bidirectional

    new_weight = filter_weight(new_weight, weight_min, weight_max)

    updated_Gg[neuron_from, neuron_to] = new_weight
    updated_Gg[neuron_to, neuron_from] = new_weight

    return updated_Gg   

def update_weight_syn(Gs, neuron_from, neuron_to, action_outgoing, action_incoming, weight_min, weight_max, update_method):

    updated_Gs = Gs.copy()

    if update_method == 'both':

        new_weight_outgoing = Gs[neuron_from, neuron_to] + action_outgoing
        new_weight_incoming = Gs[neuron_to, neuron_from] + action_incoming

        new_weight_outgoing = filter_weight(new_weight_outgoing, weight_min, weight_max)
        new_weight_incoming = filter_weight(new_weight_incoming, weight_min, weight_max)

        updated_Gs[neuron_from, neuron_to] = new_weight_outgoing
        updated_Gs[neuron_to, neuron_from] = new_weight_incoming

        return updated_Gs

    elif update_method == 'outgoing':

        new_weight_outgoing = Gs[neuron_from, neuron_to] + action_outgoing

        new_weight_outgoing = filter_weight(new_weight_outgoing, weight_min, weight_max)

        updated_Gs[neuron_from, neuron_to] = new_weight_outgoing

        return updated_Gs

    elif update_method == 'incoming':

        new_weight_incoming = Gs[neuron_to, neuron_from] + action_incoming

        new_weight_incoming = filter_weight(new_weight_incoming, weight_min, weight_max)

        updated_Gs[neuron_to, neuron_from] = new_weight_incoming

        return updated_Gs

def filter_weight(new_weight, weight_min, weight_max):

    if new_weight < weight_min:

        return weight_min

    elif new_weight > weight_max:

        return weight_max

    else:

        return new_weight

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def compute_problem_score(Gg, Gs, problem_definition, verbose=True):
    """
    Example usage:
    Gg, Gs = your_model.produce()
    mean_error, sum_error = compute_score(Gg, Gs, problems.FOUR_NEURON_OSCILLATION)
    :return mean_error, sum_error
    """
    return compute_score(
        Gg = Gg,
        Gs = Gs,
        E = problem_definition.directionality,
        input_vec = problem_definition.input_vec,
        # We are not doing any ablation.
        ablation_mask = np.ones(problem_definition.N),
        # How long the simulation will run for. tf stands for time_final.
        tf = 7,
        t_delta = 0.01,
        # Time window to calculate error metric across.
        cutoff_1 = 100,
        cutoff_2 = 600,
        m1_target = problem_definition.m1_target,
        m2_target = problem_definition.m2_target,
        plot_result=True,
        verbose=verbose)

def compute_score(Gg, Gs, E, 
                    input_vec, ablation_mask, 
                    tf, t_delta, cutoff_1, cutoff_2,
                    m1_target, m2_target,
                    plot_result = True,
                    verbose = True):

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
                                                               ablation_mask=ablation_mask,
                                                               verbose=verbose)

    #plt.plot(network_result_dict['v_solution'][100:, :])

    # Obtain test modes using SVD 

    v_solution_truncated = network_result_dict['v_solution'][100:, :]
    u,s,v = np.linalg.svd(v_solution_truncated.T)

    m1_test = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 0]
    m2_test = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 1]

    # Compute the error

    target_dist_ref = np.linalg.norm(np.vstack([m1_target, m2_target]))
    target_vel_ref = np.linalg.norm(np.vstack([np.diff(m1_target), np.diff(m2_target)]))
    target_volume = np.sqrt(np.power(np.vstack([m1_target, m2_target]), 2).sum(axis = 0)).sum()
    test_volume = np.sqrt(np.power(np.vstack([m1_test, m2_test]), 2).sum(axis = 0)).sum()

    m1_diff_dist = np.subtract(m1_target, m1_test)
    m2_diff_dist = np.subtract(m2_target, m2_test)

    #m1_diff_vel = np.subtract(np.diff(m1_target), np.diff(m1_test))
    #m2_diff_vel = np.subtract(np.diff(m2_target), np.diff(m2_test))

    #m_joined_dist = np.vstack([m1_diff_dist, m2_diff_dist])
    #errors_dist = np.sqrt(np.power(m_joined_dist, 2).sum(axis = 0))

    #m_joined_vel = np.vstack([m1_diff_vel, m2_diff_vel])
    #errors_vel = np.sqrt(np.power(m_joined_vel, 2).sum(axis = 0))

    #mean_error_dist = np.mean(errors_dist)
    #sum_error_dist = np.sum(errors_dist)

    volume_ratio = test_volume / target_volume
    volume_deviation_err = np.cosh(7 * (-1 + volume_ratio)) - 1

    error_frobenius_dist = np.linalg.norm(np.subtract(np.vstack([m1_target, m2_target]), np.vstack([m1_test, m2_test])))
    error_frobenius_vel = np.linalg.norm(np.subtract(np.vstack([np.diff(m1_target), np.diff(m2_target)]), np.vstack([np.diff(m1_test), np.diff(m2_test)])))
    error_frobenius = (1/3) * (error_frobenius_dist / target_dist_ref) + (1/3) * (error_frobenius_vel / target_vel_ref) + (1/3) * volume_deviation_err

    #mean_error_vel = np.mean(errors_vel)
    #sum_error_vel = np.sum(errors_vel)

    error_dist_flattened = np.hstack([m1_diff_dist, m2_diff_dist])
    #error_vel_flattened = np.hstack([m1_diff_vel, m2_diff_vel])

    # Plot the target vs test

    if plot_result == True:

        plt.figure(figsize=(5.5,5))

        plt.scatter(m1_target, m2_target, s = 0.75, color = 'black')
        plt.scatter(m1_test, m2_test, s = 0.75, color = 'red')

        plt.ylim(-45, 45)
        plt.xlim(-45, 45)

    return error_dist_flattened, error_frobenius, m1_test, m2_test

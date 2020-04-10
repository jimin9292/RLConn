
# coding: utf-8

###################
#UTILITY FUNCTIONS#
###################

import json
import numpy as np
import scipy.stats as spstats

from RLConn import neural_params as n_params

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

def compute_mean_velocity(x, y):
    
    # directional vectors
    
    x_pos_components = x[:, 8] - x[:, 16]
    y_pos_components = y[:, 8] - y[:, 16]
    positional_vecs = np.vstack([x_pos_components, y_pos_components])[:, :-1]
    
    # velocity vectors
    
    x_vel_components = np.diff(x[:, 12])
    y_vel_components = np.diff(y[:, 12])
    velocity_vecs = np.vstack([x_vel_components, y_vel_components])
    
    projected_vel_norms = np.zeros(len(positional_vecs[0, :]))
    
    for k in range(len(positional_vecs[0, :])):
        
        projected = project_v_onto_u(velocity_vecs[:, k], positional_vecs[:, k])
        projected_vel_norms[k] = np.linalg.norm(projected) * np.sign(np.dot(projected, positional_vecs[:, k]))
        
    mean_velocity = np.mean(projected_vel_norms)
    
    return projected_vel_norms, mean_velocity

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), spstats.sem(a)
    h = se * spstats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h

def compute_chemotaxis_index(inmask_mat, neuron_ind, ref_signal):
    
    stim_integral = np.sum(inmask_mat[:, neuron_ind])
    CI = (stim_integral - ref_signal) / ref_signal
    
    return CI

def continuous_transition_scaler(old, new, t, rate, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def compute_segment_angles(x, y):
    
    phi_segments = []

    for k in range(len(x)):

        segment_vecs = np.vstack([np.diff(x[k][::4]), np.diff(y[k][::4])]).T
        
        angles_list = []
        
        for j in range(len(segment_vecs) - 1):

            v0 = segment_vecs[j]
            v1 = segment_vecs[j+1]

            angles_list.append(np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))))
        
        angles_vec = np.asarray(angles_list)
        phi_segments.append(angles_vec)
        
        #print(k)
        
    return np.vstack(phi_segments)

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
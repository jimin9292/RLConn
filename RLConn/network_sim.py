
# coding: utf-8

import os

import numpy as np
import scipy.io as sio
from scipy import integrate, sparse, linalg, interpolate

from RLConn import neural_params as n_params
from RLConn import sys_paths as paths
from RLConn import utils
from RLConn import control_dqn as ccd

import matplotlib.pyplot as plt

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

batchsize = 3
update_frequency = 10

n_actions = 10
n_features =  10 * batchsize # 18 for single gradient sensor, 3 controllable neurons and batchsize of 3

action_2_delweight_space = [5e-4, 0, -5e-4]
#action_2_delweight_multiple = utils.compute_action_combinations()

RL = ccd.DeepQNetwork(n_actions, n_features)

# Default
# lr = 0.01
# memory_size = 5000
# batch size = 64
# learn every 10 steps
# n_features = 20
# n_actions = 3
# reward function = diff norm tanh
# epsilon = greedy
# epsilon increment = 0.002 (500 learnings)

def train_network(epochs = False):

    if epochs != False:

        for k in range(epochs):

            if k == 1:

                observation, newest_err_vec = compute_batch_state()

            else:

                observation_, newest_err_vec_ = compute_batch_state()

                reward = compute_reward(delta_norm, reward_type = 'delta_norm_tanh')
                reward_list.append(reward)

                RL.store_transition(observation, action, reward, observation_)


                observation = observation_.copy()
                newest_err_vec = newest_err_vec_.copy()

                action = RL.choose_action(observation)
                action_stim = action_2_stim_space_multiple[action]

            if k > batchsize and k % update_frequency == 0:
          
                    RL.learn()
                    print("Epoch: " + str(k))
                    print("Collected reward: " + str())

    print("score: " + str(np.sum(reward_list)))

    trained_network_dict = {

    "Gg": 'tbh',
    "Gs": 'tbh',
    "Emat_mask": 'tbh',
    "raw_v_solution": traj,
    "v_threshold": vthmat,
    "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1),
    "lowdim_traj" : 'tbh',
    "error" : 'tbh',
    "reward": 'tbh'

    }

    return trained_network_dict



def run_network_constinput_RL(t_start, t_final, t_delta, input_vec, ablation_mask, \
    custom_initcond = False, ablation_type = "all"):

    np.random.seed(10)

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    t0 = t_start
    tf = t_final
    dt = t_delta

    params_obj_neural['simulation_type'] = 'constant_input'

    nsteps = int(np.floor((tf - t0)/dt) + 1)
    params_obj_neural['inmask'] = input_vec
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Calculate V_threshold """

    params_obj_neural['vth'] = EffVth_rhs(params_obj_neural['inmask'])

    """ Set initial condition """

    if custom_initcond == False:

        initcond = 10**(-4)*np.random.normal(0, 0.94, 2*params_obj_neural['N'])
        #print(initcond)

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configuring the ODE Solver """
    r = integrate.ode(membrane_voltageRHS_constinput, compute_jacobian_constinput).set_integrator('vode', atol = 1e-3, min_step = dt*1e-6, method = 'bdf')
    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat = np.tile(params_obj_neural['vth'], (nsteps, 1))

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]

        k += 1

        if k in progress_milestones:

            print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

def generate_random_network(N, n_inhibitory, max_degree):

    np.random.seed(10)

    # Synaptic

    Gs = np.random.randint(0, max_degree, (N,N))
    np.fill_diagonal(Gs, 0)

    # Electrical

    Gg = np.random.randint(0, max_degree, (N,N))
    Gg_symm = (Gg + Gg.T)/2
    np.fill_diagonal(Gg_symm, 0)
    Gg = Gg_symm.astype('int')

    # Directionality

    inhibitory_inds = np.random.choice(np.arange(N), n_inhibitory)
    E_vec = np.zeros(N)
    E_vec[inhibitory_inds] = 1
    E_Mat = np.tile(E_vec, (N, 1))

    network_dict = {

    "gap" : Gg,
    "syn" : Gs,
    "directionality" : E_Mat

    }

    return network_dict

def initialize_params_neural(custom_params = False):

    global params_obj_neural

    if custom_params == False:

        params_obj_neural = n_params.default
        print('Using the default neural parameters')

    else:

        assert type(custom_params) == dict, "Custom neural parameters should be of dictionary format"

        if validate_custom_neural_params(custom_params) == True:

            params_obj_neural = custom_params
            print('Accepted the custom neural parameters')

def validate_custom_neural_params(custom_params):

    # TODO: Also check for dimensions

    key_checker = []

    for key in n_params.default.keys():
        
        key_checker.append(key in custom_params)

    all_keys_present = np.sum(key_checker) == n_params.default['init_key_counts']
    
    assert np.sum(key_checker) == n_params.default['init_key_counts'], "Provided dictionary is incomplete"

    return all_keys_present

def initialize_connectivity(custom_connectivity_dict = False):

    # To be executed after load_params_neural
    # custom_connectivity_dict should be of dict format with keys - 'gap', 'syn', 'directionality'
    # TODO: Check validity of custom connectomes

    assert 'params_obj_neural' in globals(), "Neural parameters must be initialized before initializing the connectivity"

    if custom_connectivity_dict == False:

        params_obj_neural['Gg_Static'] = n_params.Gg_Static
        params_obj_neural['Gs_Static'] = n_params.Gs_Static
        EMat_mask = n_params.EMat_mask
        params_obj_neural['N'] = len(n_params.Gg_Static)
        print('Using the default connectivity')

    else:

        assert type(custom_connectivity_dict) == dict, "Custom connectivity should be of dictionary format"

        params_obj_neural['Gg_Static'] = custom_connectivity_dict['gap']
        params_obj_neural['Gs_Static'] = custom_connectivity_dict['syn']
        EMat_mask = custom_connectivity_dict['directionality']
        params_obj_neural['N'] = len(custom_connectivity_dict['gap'])
        print('Accepted the custom connectivity')

    params_obj_neural['EMat'] = params_obj_neural['E_rev'] * EMat_mask
    params_obj_neural['mask_Healthy'] = np.ones(params_obj_neural['N'], dtype = 'bool')

def EffVth(Gg, Gs):

    Gcmat = np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    EcVec = np.multiply(params_obj_neural['Ec'], np.ones((params_obj_neural['N'], 1)))

    M1 = -Gcmat
    b1 = np.multiply(params_obj_neural['Gc'], EcVec)

    Ggap = np.multiply(params_obj_neural['ggap'], Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(params_obj_neural['gsyn'], Gs)
    s_eq = round((params_obj_neural['ar']/(params_obj_neural['ar'] + 2 * params_obj_neural['ad'])), 4)
    sjmat = np.multiply(s_eq, np.ones((params_obj_neural['N'], params_obj_neural['N'])))
    S_eq = np.multiply(s_eq, np.ones((params_obj_neural['N'], 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()

    #b3 = np.dot(Gs_ij, np.multiply(s_eq, params_obj_neural['E']))
    b3 = np.dot(np.multiply(Gs_ij, params_obj_neural['EMat']), s_eq * np.ones((params_obj_neural['N'], 1)))

    M = M1 + M2 + M3

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, params_obj_neural['N'])

    params_obj_neural['LL'] = LL
    params_obj_neural['UU'] = UU
    params_obj_neural['bb'] = bb

def EffVth_rhs(inmask):

    InputMask = np.multiply(params_obj_neural['iext'], inmask)
    b = np.subtract(params_obj_neural['bb'], InputMask)

    vth = linalg.solve_triangular(params_obj_neural['UU'], linalg.solve_triangular(params_obj_neural['LL'], b, lower = True, check_finite=False), check_finite=False)

    return vth

def modify_Connectome(ablation_mask, ablation_type):

    # ablation_type can be 'all': ablate both synaptic and gap junctions, 'syn': Synaptic only and 'gap': Gap junctions only

    if np.sum(ablation_mask) == params_obj_neural['N']:

        apply_Mat = np.ones((params_obj_neural['N'], params_obj_neural['N']))

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

        print("All neurons are healthy")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

    else:

        apply_Col = np.tile(ablation_mask, (params_obj_neural['N'], 1))
        apply_Row = np.transpose(apply_Col)

        apply_Mat = np.multiply(apply_Col, apply_Row)

        if ablation_type == "all":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            print("Ablating both Gap and Syn")

        elif ablation_type == "syn":

            params_obj_neural['Gg_Dynamic'] = params_obj_neural['Gg_Static'].copy()
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            print("Ablating only Syn")

        elif ablation_type == "gap":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = params_obj_neural['Gs_Static'].copy()

            print("Ablating only Gap")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])


def modify_edges(neurons_from, neurons_to, conn_type):

    apply_Mat = np.ones((params_obj_neural['N'],params_obj_neural['N']), dtype = 'bool')
    apply_Mat_Identity = np.ones((params_obj_neural['N'],params_obj_neural['N']), dtype = 'bool')

    for k in range(len(neurons_from)):

        neuron_from_ind = []
        neurons_target_inds = []

        neuron_from = neurons_from[k]
        neurons_target = neurons_to[k]

        neuron_from_ind.append(neuron_names.index(neuron_from))

        for neuron_target in neurons_target:

            neurons_target_inds.append(neuron_names.index(neuron_target))

        if conn_type == 'syn':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0

        elif conn_type == 'gap':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0
            apply_Mat[neuron_from_ind, neurons_target_inds] = 0

    if conn_type == 'syn':

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat_Identity)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

    elif conn_type == 'gap':

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat_Identity)

    EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

def add_gap_junctions(neuron_pairs_mat, gap_weights_vec):

    """ neuron_pairs_mat is (N, 2) numpy.array form where N is the total number of pairs. 
        Each element should be of type float denoting the index number of neuron"""

    """ gap_weights_vec is (N,) numpy.array form where N is the total number of pair.
        Each element should be of type float denoting the gap weights to be added for each pair"""

    """ This function should be executed after modify_Connectome """

    num_pairs = len(neuron_pairs_mat)

    for k in range(num_pairs):

        neuron_ind_1 = neuron_pairs_mat[k, 0]
        neuron_ind_2 - neuron_pairs_mat[k, 1]

        params_obj_neural['Gg_Dynamic'][neuron_ind_1, neuron_ind_2] = params_obj_neural['Gg_Dynamic'][neuron_ind_1, neuron_ind_2] + gap_weights_vec[k]
        params_obj_neural['Gg_Dynamic'][neuron_ind_2, neuron_ind_1] = params_obj_neural['Gg_Dynamic'][neuron_ind_2, neuron_ind_1] + gap_weights_vec[k]

    EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

def voltage_filter(v_vec, vmax, scaler):
    
    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))
    
    return filtered

def produce_lowdim_traj(v_solution, dim_num):

    u,s,v = np.linalg.svd(v_solution)

    u_subbed = u[:, :dim_num]
    s_subbed = s[:dim_num]
    s_subbed_mat = np.tile(s_subbed, (len(v_solution[0, :]), 1))
    projected = np.dot(v_solution.T, u_subbed)
    weighted_projected = np.multiply(s_subbed_mat, projected)

    return projected, weighted_projected

###################################################################################

def compute_batch_state():

    return 'function to be written'

def compute_reward(reward_param, reward_type = 'asymptotic'):

    if reward_type == 'asymptotic':

        reward = np.reciprocal(np.exp(positive_reward_droprate * reward_param))

    elif reward_type == 'leaky':

        if reward_param < target_radius:

            reward = np.reciprocal(np.exp(positive_reward_droprate * reward_param))

        else:

            reward = -negative_reward_coeff * (reward_param - target_radius)

    elif reward_type == 'delta_norm_tanh':

        reward = -np.tanh(delta_norm_const * reward_param)

    elif reward_type == 'proximity':

        if reward_param < target_radius:

            reward = 1

        else:

            reward = -1

    elif reward_type == 'attraction':

        gradient_strength = reward_param[0]

        if gradient_strength < 1:

            reward = gradient_strength

        else:

            reward = 1

    return reward

def membrane_voltageRHS_constinput(t, y):

    Vvec, SVec = np.split(y, 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth'])))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def compute_jacobian_constinput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth']))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J














"""
A list of connectome problem definitions to make sure that all approaches are being scored the same
way.
"""

import numpy as np
from RLConn import network_sim
from RLConn import neural_params as n_params
from RLConn import connectome_utils
from RLConn import stephen_utils as connectomes

class ProblemDefinition():
    def __init__(self, N, m1_target, m2_target, directionality, input_vec,
                 tf = 7, cutoff_1 = 100, cutoff_2 = 600, init_Gg = None, init_Gs = None):
        self.N = N
        # The top 2 modes of the gold dynamics to compare against.
        self.m1_target = m1_target
        self.m2_target = m2_target
        self.directionality = directionality
        self.input_vec = input_vec
        self.tf = tf
        self.cutoff_1 = cutoff_1
        self.cutoff_2 = cutoff_2

        # Suggested initial connectome to try.
        self.init_Gg = init_Gg
        self.init_Gs = init_Gs

np.random.seed(10)
network_dict = connectome_utils.generate_random_network(
    N = 4,
    n_inhibitory = 1,
    max_degree = 10
)

# Stimulus just to the second neuron.
input_vec = np.zeros(4)
input_vec[1] = 0.068

# DEFINITION 1: 4 neurons that show oscillations.
FOUR_NEURON_OSCILLATION = ProblemDefinition(
    N = 4,
    m1_target = n_params.m1_target,
    m2_target = n_params.m2_target,
    # TODO: Add initial guesses from noised-up true parameters. Punted to final phase.
    directionality =  network_dict['directionality'],
    input_vec =  input_vec
)

def get_three_neuron_oscillation_definition():
    N = 3
    Gg = np.array([[0, 8, 5],
                   [8, 0, 2],
                   [5, 2, 0]])
    Gs = np.array([[0, 2, 8],
                   [7, 0, 3],
                   [7, 7, 0]])
    is_inhibitory = np.array([1, 0, 0])
    input_vec = [0, 0.03, 0]
    # We are not doing any ablation.
    ablation_mask = np.ones(N)
    t_delta = 0.01

    # Run for 10 seconds.
    tf = 10
    # Note that these cutoff offsets are actually number of dt's _after_
    # the 100 timestep truncation in compute_score.
    cutoff_1 = 400
    cutoff_2 = 900

    network_dict = {
        "gap": Gg,
        "syn": Gs,
        "directionality": is_inhibitory
    }

    # Initialize model with the network dict

    network_sim.initialize_params_neural()
    network_sim.initialize_connectivity(network_dict)

    # Simulate network with given input_vec and ablation mask

    network_result_dict = network_sim.run_network_constinput_RL(0, tf, t_delta,
                                                                input_vec=input_vec,
                                                                ablation_mask=ablation_mask,
                                                                verbose=False)
    # Obtain test modes using SVD
    v_solution_truncated = network_result_dict['v_solution'][100:, :]
    u, s, v = np.linalg.svd(v_solution_truncated.T)
    projected = np.dot(v_solution_truncated, u)
    m1 = projected[cutoff_1:cutoff_2, 0]
    m2 = projected[cutoff_1:cutoff_2, 1]

    # Set initial guess to be around the true parameter values, but with noise.
    np.random.seed(2)
    init_compact_vec = [8, 5, 2, 7, 7, 7, 2, 8, 3]
    init_compact_vec += np.random.rand(len(init_compact_vec))
    init_Gg, init_Gs = connectomes.compact_to_model_param(init_compact_vec, N)

    return ProblemDefinition(
        N = 3,
        m1_target = m1,
        m2_target = m2,
        directionality =  np.array([1, 0 ,0]),
        input_vec =  np.array([0, 0.03, 0]),
        tf = tf,
        cutoff_1 = cutoff_1,
        cutoff_2 = cutoff_2,
        init_Gg = init_Gg,
        init_Gs = init_Gs,
    )

def get_four_neuron_oscillation_definition():
    np.random.seed(10)
    N = 4
    network = connectome_utils.generate_random_network(N, 1, 10)
    Gg = network['gap']
    Gs = network['syn']
    is_inhibitory = network['directionality'][0]
    input_vec = [0.068, 0, 0, 0]

    # We are not doing any ablation.
    ablation_mask = np.ones(N)
    t_delta = 0.01

    # Run for 10 seconds.
    tf = 10
    # Note that these cutoff offsets are actually number of dt's _after_
    # the 100 timestep truncation in compute_score.
    cutoff_1 = 150
    cutoff_2 = 650

    network_dict = {
        "gap": Gg,
        "syn": Gs,
        "directionality": is_inhibitory
    }

    # Initialize model with the network dict

    network_sim.initialize_params_neural()
    network_sim.initialize_connectivity(network_dict)

    # Simulate network with given input_vec and ablation mask

    network_result_dict = network_sim.run_network_constinput_RL(0, tf, t_delta,
                                                                input_vec=input_vec,
                                                                ablation_mask=ablation_mask,
                                                                verbose=False)
    # Obtain test modes using SVD
    v_solution_truncated = network_result_dict['v_solution'][100:, :]
    u, s, v = np.linalg.svd(v_solution_truncated.T)
    top_mode = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 0]

    # Set initial guess to be around the true parameter values, but with noise.
    np.random.seed(2)
    init_compact_vec = connectomes.model_to_compact_param(Gg, Gs, N);
    init_compact_vec = init_compact_vec.astype(float);
    init_compact_vec += np.random.rand(len(init_compact_vec))
    init_Gg, init_Gs = connectomes.compact_to_model_param(init_compact_vec, N)

    return ProblemDefinition(
        N = N,
        m1_target = top_mode,
        m2_target = None,
        directionality =  is_inhibitory,
        input_vec =  input_vec,
        tf = tf,
        cutoff_1 = cutoff_1,
        cutoff_2 = cutoff_2,
        init_Gg = init_Gg,
        init_Gs = init_Gs,
    )

def get_five_neuron_oscillation_definition():
    np.random.seed(10)
    N = 5
    np.random.seed(3)
    network = connectome_utils.generate_random_network(N, 1, 10)
    Gg = network['gap']
    Gs = network['syn']
    is_inhibitory = network['directionality'][0]
    input_vec = [0.068, 0, 0, 0, 0]

    # We are not doing any ablation.
    ablation_mask = np.ones(N)
    t_delta = 0.01

    # Run for 15 seconds.
    tf = 15
    # Note that these cutoff offsets are actually number of dt's _after_
    # the 100 timestep truncation in compute_score.
    cutoff_1 = 650
    cutoff_2 = 900

    network_dict = {
        "gap": Gg,
        "syn": Gs,
        "directionality": is_inhibitory
    }

    # Initialize model with the network dict

    network_sim.initialize_params_neural()
    network_sim.initialize_connectivity(network_dict)

    # Simulate network with given input_vec and ablation mask

    network_result_dict = network_sim.run_network_constinput_RL(0, tf, t_delta,
                                                                input_vec=input_vec,
                                                                ablation_mask=ablation_mask,
                                                                verbose=False)
    # Obtain test modes using SVD
    v_solution_truncated = network_result_dict['v_solution'][100:, :]
    u, s, v = np.linalg.svd(v_solution_truncated.T)
    top_mode = np.dot(v_solution_truncated, u)[cutoff_1:cutoff_2, 0]

    # Set initial guess to be around the true parameter values, but with noise.
    np.random.seed(2)
    init_compact_vec = connectomes.model_to_compact_param(Gg, Gs, N);
    init_compact_vec = init_compact_vec.astype(float);
    init_compact_vec += np.random.rand(len(init_compact_vec))
    init_Gg, init_Gs = connectomes.compact_to_model_param(init_compact_vec, N)

    return ProblemDefinition(
        N = N,
        m1_target = top_mode,
        m2_target = None,
        directionality =  is_inhibitory,
        input_vec =  input_vec,
        tf = tf,
        cutoff_1 = cutoff_1,
        cutoff_2 = cutoff_2,
        init_Gg = init_Gg,
        init_Gs = init_Gs,
    )
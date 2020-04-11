"""
A list of connectome problem definitions to make sure that all approaches are being scored the same
way.
"""

import numpy as np
from RLConn import neural_params as n_params
from RLConn import connectome_utils

class ProblemDefinition():
    def __init__(self, N, m1_target, m2_target, directionality, input_vec):
        self.N = N
        # The top 2 modes of the gold dynamics to compare against.
        self.m1_target = m1_target
        self.m2_target = m2_target
        self.directionality = directionality
        self.input_vec = input_vec

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

# Stimulus just to the second neuron.
# DEFINITION 2: 3 neurons that show oscillations.
THREE_NEURON_OSCILLATION = ProblemDefinition(
    N = 3,
    m1_target = n_params.m1_target,
    m2_target = n_params.m2_target,
    # TODO: Add initial guesses from noised-up true parameters. Punted to final phase.
    directionality =  np.array([1, 0 ,0]),
    input_vec =  np.array([0, 0.03, 0])
)

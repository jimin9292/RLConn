import numpy as np

def generate_random_network(N, n_inhibitory, max_degree):
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

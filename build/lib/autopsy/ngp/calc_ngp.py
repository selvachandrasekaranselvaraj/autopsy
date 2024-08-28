import numpy as np
from autopsy.ngp.ngp_fft import ngp_fft

def calc_ngp(atom_positions):
    '''
    Calculate the Nearest Neighbor Geometry Parameter (NGP) for a given set of atomic positions.

    :param atom_positions: numpy array, atomic positions with shape (n_frames, n_atoms, 3)
    :return: numpy array, normalized NGP values
    '''
    r_sum = np.sum(atom_positions, axis=1)
    n_atoms = np.shape(atom_positions)[1]
    ngp = ngp_fft(r_sum)
    return ngp / n_atoms

import numpy as np
from autopsy.msd.fft import fft

def calc_msd_total(atom_positions):
    """
    Calculates the total mean squared displacement (MSD) by summing the positions of all atoms.

    :param atom_positions: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    n_atoms = np.shape(atom_positions)[1]
    
    # Sum the positions of all atoms along the second axis (axis=1)
    r_sum = np.sum(atom_positions, axis=1)
    
    # Calculate the total MSD using the provided `fft` function
    msd = fft(r_sum)
    
    # Normalize MSD by the number of atoms
    return np.array(msd) / n_atoms

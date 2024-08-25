import numpy as np
from autopsy.cross_msd.fft_cross import fft_cross

def calc_msd_cross(atom_type1_positions, atom_type2_positions):
    """
    Computes the cross mean square displacement (MSD) between cation and anion positions.

    :param atom_type1_positions: array[float], cation atom positions over time
    :param atom_type2_positions: array[float], anion atom positions over time
    :return: msd: array[float], cross mean-squared displacement over time
    """      
    # Calculate the sum of cation and anion positions along axis 1
    r_1 = np.sum(atom_type1_positions, axis=1)
    r_2 = np.sum(atom_type2_positions, axis=1)
    
    # Compute the cross mean square displacement using the provided `fft_cross` function
    msd = fft_cross(np.array(r_1), np.array(r_2))
    
    return np.array(msd)

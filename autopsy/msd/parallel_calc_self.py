import numpy as np
from tqdm import tqdm
from autopsy.msd.fft import fft

def calc_msd_self(atom_positions):
    """
    Calculates the mean squared displacement (MSD) for each atom and averages over all atoms.

    :param atom_positions: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    Lii_self = np.zeros(np.shape(atom_positions)[0])
    n_atoms = np.shape(atom_positions)[1]
    
    # Loop through each atom and calculate MSD using FFT
    for atom_num in tqdm(range(n_atoms)):
        r = atom_positions[:, atom_num, :]
        msd_temp = fft(np.array(r))
        Lii_self += msd_temp
    
    # Average MSD over all atoms
    msd = np.array(Lii_self) / n_atoms
    
    return msd


import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from autopsy.msd.fft import fft

def calculate_msd_for_atom(args):
    atom_positions, atom_num = args
    r = atom_positions[:, atom_num, :]
    msd_temp = fft(np.array(r))
    return msd_temp

def calc_msd_self(atom_positions):
    """
    Calculates the mean squared displacement (MSD) for each atom and averages over all atoms.

    :param atom_positions: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    Lii_self = np.zeros(np.shape(atom_positions)[0])
    n_atoms = np.shape(atom_positions)[1]

    # Create a list of arguments for each atom
    args_list = [(atom_positions, atom_num) for atom_num in range(n_atoms)]

    # Use ProcessPoolExecutor to parallelize the calculations
    with ProcessPoolExecutor(max_workers=128) as executor:
        results = list(tqdm(executor.map(calculate_msd_for_atom, args_list), total=n_atoms, disable=False))

    # Sum up the results
    for msd_temp in results:
        Lii_self += msd_temp

    # Average MSD over all atoms
    msd = np.array(Lii_self) / n_atoms

    return msd



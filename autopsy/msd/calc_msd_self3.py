# Simple usage:
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from autopsy.msd.fft import fft

def calc_msd_self(atom_positions, n_jobs=-1):
    """Simplified fastest version."""
    n_frames, n_atoms, _ = atom_positions.shape
    
    # Use float32 to save memory
    if atom_positions.dtype != np.float32:
        atom_positions = atom_positions.astype(np.float32)
    
    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(fft)(atom_positions[:, i, :])
        for i in tqdm(range(n_atoms), desc="MSD calculation")
    )
    
    return np.sum(results, axis=0) / n_atoms

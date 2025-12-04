import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from autopsy.msd.fft import fft
import multiprocessing

def calculate_msd_for_atoms_batch(args):
    """Calculate MSD for a batch of atoms."""
    atom_positions, atom_indices = args
    batch_msd = np.zeros(atom_positions.shape[0])
    
    for atom_idx in atom_indices:
        r = atom_positions[:, atom_idx, :]
        msd_temp = fft(r)
        batch_msd += msd_temp
    
    return batch_msd, len(atom_indices)

def calc_msd_self(atom_positions, n_proc=None):
    """
    Optimized MSD calculation with batch processing.
    """
    n_frames, n_atoms, _ = atom_positions.shape
    
    if n_proc is None:
        n_proc = multiprocessing.cpu_count()
    
    # Create batches of atoms
    batch_size = max(1, n_atoms // (n_proc * 2))
    atom_batches = []
    
    for i in range(0, n_atoms, batch_size):
        atom_batches.append((atom_positions, list(range(i, min(i + batch_size, n_atoms)))))
    
    msd = np.zeros(n_frames)
    total_atoms_processed = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=n_proc) as executor:
        results = list(tqdm(
            executor.map(calculate_msd_for_atoms_batch, atom_batches),
            total=len(atom_batches),
            desc="Processing atom batches"
        ))
    
    # Combine results
    for batch_msd, n_batch_atoms in results:
        msd += batch_msd
        total_atoms_processed += n_batch_atoms
    
    # Safety check and normalization
    if total_atoms_processed != n_atoms:
        print(f"Warning: Processed {total_atoms_processed} atoms out of {n_atoms}")
    
    msd /= total_atoms_processed
    
    return msd

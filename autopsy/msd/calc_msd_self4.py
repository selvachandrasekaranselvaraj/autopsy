import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from autopsy.msd.fft import fft
import os
import tempfile

def _worker_function(args):
    """Worker function for parallel processing."""
    mmap_path, shape, dtype, frame_start, frame_end, atom_idx = args
    
    # Open memory-mapped array
    arr = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)
    
    # Extract positions for this atom
    r = arr[frame_start:frame_end, atom_idx, :].copy()
    msd_temp = fft(r)
    
    del arr
    return msd_temp[:frame_end-frame_start]

def calc_msd_self(atom_positions, n_proc=None, chunk_frames=2000):
    """
    Simple memory-constrained version using multiprocessing.Pool.
    """
    n_frames, n_atoms, _ = atom_positions.shape
    
    if n_proc is None:
        n_proc = max(1, mp.cpu_count() - 1)
    
    # Convert to float32
    positions = atom_positions.astype(np.float32, copy=False)
    
    # Create temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
    tmp_path = tmp_file.name
    tmp_file.close()
    
    # Write to memory-mapped file
    mmap_arr = np.memmap(tmp_path, dtype=np.float32, 
                         mode='w+', shape=positions.shape)
    mmap_arr[:] = positions[:]
    mmap_arr.flush()
    del mmap_arr
    
    try:
        msd = np.zeros(n_frames, dtype=np.float64)
        
        # Process in frame chunks
        for frame_start in range(0, n_frames, chunk_frames):
            frame_end = min(frame_start + chunk_frames, n_frames)
            chunk_size = frame_end - frame_start
            
            print(f"Processing frames {frame_start}-{frame_end}...")
            
            # Prepare tasks for this chunk
            tasks = []
            for atom_idx in range(n_atoms):
                tasks.append((tmp_path, positions.shape, np.float32, 
                             frame_start, frame_end, atom_idx))
            
            # Process in parallel
            with mp.Pool(processes=n_proc) as pool:
                results = list(tqdm(
                    pool.imap(_worker_function, tasks, chunksize=10),
                    total=n_atoms,
                    desc=f"Atoms for frames {frame_start}-{frame_end}"
                ))
            
            # Sum results
            chunk_msd = np.sum(results, axis=0) / n_atoms
            msd[frame_start:frame_end] = chunk_msd
        
        return msd
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

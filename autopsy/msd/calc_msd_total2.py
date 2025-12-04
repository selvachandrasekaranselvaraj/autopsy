import numpy as np
import h5py
from joblib import Parallel, delayed
import tempfile
import os
from tqdm.auto import tqdm
from autopsy.msd.fft import fft

def calc_msd_total(atom_positions, n_jobs=-2, chunk_frames=1000, 
                            use_hdf5=True, cache_dir=None):
    """
    Parallel calculation of total MSD using HDF5 for memory efficiency.
    
    Parameters:
    -----------
    atom_positions : np.ndarray or str
        Either a numpy array or path to HDF5 file
    n_jobs : int
        Number of parallel jobs (-2 = all but one CPU)
    chunk_frames : int
        Number of frames to process at once
    use_hdf5 : bool
        Whether to use HDF5 for disk caching
    cache_dir : str
        Directory for temporary files (None = system temp)
    
    Returns:
    --------
    msd : np.ndarray
        Total mean-squared displacement
    """
    # If input is a file path, open it
    if isinstance(atom_positions, str):
        with h5py.File(atom_positions, 'r') as f:
            n_frames, n_atoms, _ = f['positions'].shape
        hdf5_path = atom_positions
        need_cleanup = False
    else:
        # Convert input array to HDF5 for efficient access
        n_frames, n_atoms, _ = atom_positions.shape
        
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        
        hdf5_path = os.path.join(cache_dir, f'msd_total_{os.getpid()}.h5')
        need_cleanup = True
        
        # Create HDF5 file with chunked storage
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('positions', data=atom_positions,
                            chunks=(min(500, n_frames), 1, 3),
                            compression='gzip')
    
    print(f"Processing total MSD: {n_frames:,} frames × {n_atoms:,} atoms")
    
    try:
        # Process in chunks of frames to manage memory
        msd_total = np.zeros(n_frames, dtype=np.float64)
        
        # Function to process a chunk of frames
        def process_frame_chunk(frame_start, frame_end):
            """Process a chunk of frames to compute summed positions."""
            with h5py.File(hdf5_path, 'r') as f:
                # Read chunk for all atoms
                chunk = f['positions'][frame_start:frame_end, :, :]
                # Sum positions across all atoms
                r_sum_chunk = np.sum(chunk, axis=1, dtype=np.float64)
            
            return r_sum_chunk, frame_start, frame_end
        
        # Create frame chunk tasks
        frame_chunks = [(i, min(i + chunk_frames, n_frames)) 
                       for i in range(0, n_frames, chunk_frames)]
        
        print(f"Processing {len(frame_chunks)} frame chunks...")
        
        # Process frame chunks in parallel
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(process_frame_chunk)(start, end)
            for start, end in tqdm(frame_chunks, desc="Frame chunks")
        )
        
        # Combine summed positions from all chunks
        r_sum_full = np.zeros((n_frames, 3), dtype=np.float64)
        for r_sum_chunk, start, end in results:
            r_sum_full[start:end] = r_sum_chunk
        
        # Now compute FFT on the full summed trajectory
        print("Computing FFT on summed positions...")
        msd = fft(r_sum_full)
        
        return msd / n_atoms
        
    finally:
        # Clean up temporary HDF5 file if we created it
        if need_cleanup and os.path.exists(hdf5_path):
            os.remove(hdf5_path)

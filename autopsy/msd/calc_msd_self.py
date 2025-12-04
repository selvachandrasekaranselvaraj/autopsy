from joblib import Parallel, delayed, Memory
import numpy as np
from tqdm.auto import tqdm
from autopsy.msd.fft import fft
import tempfile
import os
import gc

# Optional: Enable disk caching to avoid recomputation
cachedir = tempfile.mkdtemp()
memory = Memory(cachedir, verbose=0)

def calc_msd_self(atom_positions, n_jobs=-1, 
                               use_float32=True, frame_chunk=2000,
                               atom_chunk=50, use_disk_cache=False):
    """
    Optimized Joblib version for massive datasets (20k+ frames, 10k+ atoms).
    
    Parameters:
    -----------
    atom_positions : np.ndarray
        Shape (n_frames, n_atoms, 3)
    n_jobs : int
        -1: all CPUs, -2: all but one, -3: all but two, etc.
    use_float32 : bool
        Use float32 to save memory (highly recommended)
    frame_chunk : int
        Number of frames to process at once
    atom_chunk : int
        Number of atoms per parallel task
    use_disk_cache : bool
        Cache intermediate results to disk
    
    Returns:
    --------
    msd : np.ndarray
        Mean-squared displacement
    """
    n_frames, n_atoms, _ = atom_positions.shape
    
    print(f"Processing: {n_frames:,} frames × {n_atoms:,} atoms")
    print(f"Total positions: {n_frames * n_atoms:,}")
    
    # Convert to float32 to save memory
    if use_float32 and atom_positions.dtype != np.float32:
        print("Converting to float32 for memory efficiency...")
        positions = atom_positions.astype(np.float32, copy=False)
        mem_saving = 1 - (positions.nbytes / atom_positions.nbytes)
        print(f"Memory reduced by {mem_saving:.1%}")
    else:
        positions = atom_positions
    
    # Write to memory-mapped file for efficient partial reading
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
    tmp_path = tmp_file.name
    tmp_file.close()
    
    np.save(tmp_path, positions)
    print(f"Data cached to: {tmp_path}")
    
    # Function to process a batch of atoms for specific frames
    @memory.cache if use_disk_cache else (lambda func: func)
    def process_atom_batch(atom_indices, frame_slice):
        """Process a batch of atoms for specific frames."""
        frame_start, frame_end = frame_slice
        chunk_size = frame_end - frame_start
        
        # Memory-map the file for efficient reading
        data = np.load(tmp_path, mmap_mode='r')
        
        batch_msd = np.zeros(chunk_size, dtype=np.float32)
        
        for atom_idx in atom_indices:
            # Read only needed frames for this atom
            r = data[frame_start:frame_end, atom_idx, :].copy()
            msd_temp = fft(r)
            batch_msd += msd_temp[:chunk_size].astype(np.float32)
        
        # Explicit cleanup
        del data
        gc.collect()
        
        return batch_msd, len(atom_indices)
    
    try:
        msd_total = np.zeros(n_frames, dtype=np.float64)
        atoms_processed = 0
        
        # Process in chunks of frames
        for frame_start in tqdm(range(0, n_frames, frame_chunk),
                               desc="Frame chunks"):
            frame_end = min(frame_start + frame_chunk, n_frames)
            
            # Split atoms into batches for parallel processing
            atom_batches = []
            for i in range(0, n_atoms, atom_chunk):
                atom_batch = list(range(i, min(i + atom_chunk, n_atoms)))
                atom_batches.append(atom_batch)
            
            print(f"  Processing {len(atom_batches)} atom batches for "
                  f"frames {frame_start}-{frame_end}...")
            
            # Process atom batches in parallel
            results = Parallel(n_jobs=n_jobs, backend='loky',
                             prefer='processes', verbose=0)(
                delayed(process_atom_batch)(batch, (frame_start, frame_end))
                for batch in tqdm(atom_batches, desc="Atom batches",
                                 leave=False, position=1)
            )
            
            # Combine results for this frame chunk
            chunk_msd = np.zeros(frame_end - frame_start, dtype=np.float64)
            chunk_atoms = 0
            
            for batch_msd, batch_size in results:
                chunk_msd += batch_msd.astype(np.float64)
                chunk_atoms += batch_size
            
            # Store results
            msd_total[frame_start:frame_end] = chunk_msd / chunk_atoms
            atoms_processed += chunk_atoms
            
            # Progress update
            print(f"  Completed {frame_end}/{n_frames} frames "
                  f"({frame_end/n_frames*100:.1f}%)")
        
        print(f"\nProcessed all {atoms_processed:,} atoms successfully")
        
        return msd_total
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

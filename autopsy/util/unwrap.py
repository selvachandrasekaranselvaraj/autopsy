import numpy as np
from concurrent.futures import ProcessPoolExecutor
from autopsy.util.find_n_proc import find_optimal_n_proc
from ase.io import read

def unwrap_frame(args):
    """
    Unwrap positions for a single frame using the first frame as a reference.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the current frame's wrapped positions, cell matrix, 
        and the reference frame's wrapped positions and cell matrix inverse.
    
    Returns
    -------
    unwrapped_positions : numpy.ndarray
        Array of unwrapped positions for the frame.
    """
    wrapped_positions_i, wrapped_positions_f, cell_matrix_current, reference_positions, cell_matrix_inv_0 = args

    # Convert positions to fractional coordinates with respect to the first frame
    fractional_positions_i = np.dot(wrapped_positions_i, cell_matrix_inv_0.T)
    fractional_positions_f = np.dot(wrapped_positions_f, cell_matrix_inv_0.T)
    reference_fractional_positions = np.dot(reference_positions, cell_matrix_inv_0.T)
    
    # Calculate the displacement in fractional coordinates
    displacement = fractional_positions_i - fractional_positions_f #reference_fractional_positions
    
    # Unwrap fractional coordinates
    displacement -= np.round(displacement)
    
    # Convert back to Cartesian coordinates
    unwrapped_positions = reference_positions + np.dot(displacement, cell_matrix_current)
    
    return unwrapped_positions

def unwrap(data):
    """
    Unwrap positions for all frames in a trajectory using the first frame as a reference.
    Parallelized version.
    
    Parameters
    ----------
    data : list of ASE Atoms objects
        List of ASE Atoms objects containing wrapped positions for all frames.
    
    Returns
    -------
    list of ASE Atoms objects
        List of ASE Atoms objects with unwrapped positions for all frames.
    """
    num_cores = find_optimal_n_proc()
    positions = np.array([d.positions for d in data])
    cell_matrix = np.array([np.array(d.cell) for d in data])
    
    n_frames, n_atoms, _ = positions.shape
    unwrapped_positions = np.zeros_like(positions)
    
    # Use the first frame as the reference
    unwrapped_positions[0] = positions[0]
    
    # Compute the inverse cell matrix for the first frame
    cell_matrix_inv_0 = np.linalg.inv(cell_matrix[0])
    
    # Prepare arguments for parallel processing
    args_list = [(positions[i-1], positions[i], cell_matrix[i], positions[0], cell_matrix_inv_0) for i in range(1, n_frames)]
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(unwrap_frame, args_list)
    
    # Collect the results
    for i, result in enumerate(results, start=1):
        data[i].positions = result
    
    return data

# Example usage:
# Assuming you have a list of ASE Atoms objects `atoms_list`
# atoms_list = read('trajectory_file.xyz', index=':')

# unwrapped_atoms_list = unwrap_trajectory_parallel(atoms_list)


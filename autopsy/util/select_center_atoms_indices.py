import numpy as np

def select_center_atoms_indices(positions, cell):
    '''
    Sorts atomic indices based on their distance from the center of the cell and within a specified z-range.

    :param positions: numpy array, atomic positions in the trajectory.
    :param cell: numpy array, cell dimensions.
    :return: numpy 1D array of first four center atoms indices.

    '''
    # Calculate the center of the cell
    cell_center = cell * 0.5

    # Calculate the positions relative to the center of the cell
    pos_c = positions[0] - cell_center

    # Calculate the distance of each atom from the center
    r_c = np.linalg.norm(pos_c, axis=1)

    # Sort atomic indices based on distance
    indices_sort = np.argsort(r_c)

    # Optionally, you can use a distance cutoff (e.g., r_c < 5) and get indices within that cutoff
    # indices = np.where(r_c < 5)[0]

    return indices_sort

import numpy as np

def sort_atomic_indices(positions, cell):
    '''
    Sorts atomic indices based on their distance from the center of the cell.

    :param positions: numpy array, atomic positions in the trajectory.
    :param cell: numpy array, cell dimensions.

    :return: numpy array, sorted atomic indices.
    '''
    # Calculate the center of the cell
    cell_center = cell * 0.5

    # Calculate the positions relative to the center of the cell
    pos_c = positions[0] - cell_center

    # Calculate the distance of each atom from the center
    r_c = np.linalg.norm(pos_c, axis=1)

    # Sort atomic indices based on distance
    indices = np.argsort(r_c)

    # Optionally, you can use a distance cutoff (e.g., r_c < 5) and get indices within that cutoff
    # indices = np.where(r_c < 5)[0]

    return indices

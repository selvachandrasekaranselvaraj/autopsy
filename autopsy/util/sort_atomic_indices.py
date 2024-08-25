import numpy as np

def sort_atomic_indices(positions, cell, zmin, zmax):
    '''
    Sorts atomic indices based on their distance from the center of the cell and within a specified z-range.

    :param positions: numpy array, atomic positions in the trajectory.
    :param cell: numpy array, cell dimensions.
    :param zmin: float, minimum value of z-axis for selection.
    :param zmax: float, maximum value of z-axis for selection.
    :return: numpy array, sorted atomic indices within the specified z-range.

    '''
    p = positions

    if zmin == None and zmax == None:
        indices = np.where((p[0, :, 2] >= min(p[0, :, 2])) & (p[0, :, 2] <= max(p[0, :, 2])))[0]
    else:
        if float(zmin) == 0.0:
            zmin = min(p[0, :, 2])
        indices = np.where((p[0, :, 2] >= float(zmin)) & (p[0, :, 2] <= float(zmax)))[0]

    return indices


    def indices_(p, i):
        exclude_boundary_value = 0.01
        pos = p[0, :, i]
        return np.where((pos > exclude_boundary_value) & (pos < cell[i] - exclude_boundary_value))[0]

    # Find common indices in indices_xyz_
    indices_list = [indices_(p, i) for i in range(3)]
    common_indices = np.intersect1d(np.intersect1d(indices_list[0], indices_list[1]), indices_list[2])

    last_frame_pos = positions[-1]  # Extract positions in the last frame
    z_pos = last_frame_pos[:, 2]  # Extract z-axis values

    # Select indices of atoms that are in the range of [zmin, zmax]
    indices_z = np.where((z_pos > zmin) & (z_pos < zmax))[0]

    return np.intersect1d(common_indices, indices_z)

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

    return indices

import numpy as np
from numba import njit

@njit
def custom_norm(arr, axis=-1):
    if arr.ndim == 3 and axis == -1:
        # Special case for 3D array and axis=-1
        return np.sqrt(np.sum(arr**2, axis=(arr.ndim-1)))
    else:
        return np.sqrt(np.sum(arr**2, axis=axis))

#@njit
#def custom_norm(arr, axis=-1):
#    return np.sqrt(np.sum(arr**2, axis=axis))

@njit
def dis_1d(traj, lag_time):
    return traj[lag_time:] - traj[:-lag_time]

def calculate_displacements(traj, lag_time):
    if traj.shape[-1] == 3:
        # Calculate Euclidean norm of displacements for 3D trajectories
        displacements = custom_norm(traj[lag_time:] - traj[:-lag_time], axis=-1)
    elif traj.shape[-1] == 1:
        # Calculate 1D displacements for 1D trajectories
        displacements = dis_1d(traj, lag_time) 

    return displacements.reshape(-1)  # Explicitly reshape to 1D array

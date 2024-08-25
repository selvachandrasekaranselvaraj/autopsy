import numpy as np
import pandas as pd
from tqdm import tqdm
from autopsy.util.calc_r import calculate_displacements

def calc_vhf(traj, atom):
    '''
    Calculate Van Hove function (VHF) for a given atom in a trajectory.

    :param traj: numpy array, trajectory data with shape (n_frames, n_particles, n_dimensions)
    :param atom: str, atomic symbol
    :return: tuple of numpy arrays, containing mean displacements and VHF for x, y, z directions
    '''
    n_frames, n_particles, n_dimensions = traj.shape
    bins = np.linspace(0, 3.0, 200)  # Adjust the range and number of bins as needed
    
    if n_frames > 5000:
        max_lag_times = 5000
        x = int(np.floor(n_frames / max_lag_times))
        lag_times = np.arange(1, n_frames, x)  # Specify lag times of interest
    else:
        lag_times = np.arange(1, n_frames)  # Specify lag times of interest
    

    n_bins = len(bins) - 1
    n_lag_times = len(lag_times)

    van_hove = np.zeros((n_bins, n_lag_times))
    van_hove_x = np.zeros((n_bins, n_lag_times))
    van_hove_y = np.zeros((n_bins, n_lag_times))
    van_hove_z = np.zeros((n_bins, n_lag_times))
    max_bin_value = 0.0
    dis = np.zeros((n_lag_times))
    dis_x = np.zeros((n_lag_times))
    dis_y = np.zeros((n_lag_times))
    dis_z = np.zeros((n_lag_times))

    for i, lag_time in tqdm(enumerate(lag_times), disable=False):
        r = calculate_displacements(traj, lag_time)
        dis[i] = np.mean(r)
        hist, _ = np.histogram(r, bins=bins, density=True)

        rx = calculate_displacements(traj[:, :, 0].reshape(n_frames, n_particles, 1), lag_time)
        dis_x[i] = np.mean(np.abs(rx))
        hist_x, _ = np.histogram(rx, bins=bins, density=True)

        ry = calculate_displacements(traj[:, :, 1].reshape(n_frames, n_particles, 1), lag_time)
        dis_y[i] = np.mean(np.abs(ry))
        hist_y, _ = np.histogram(ry, bins=bins, density=True)

        rz = calculate_displacements(traj[:, :, 2].reshape(n_frames, n_particles, 1), lag_time)
        dis_z[i] = np.mean(np.abs(rz))
        hist_z, _ = np.histogram(rz, bins=bins, density=True)

        max_value_index = np.max(np.where(hist > max(hist) * 0.05))
        max_bin_value_ = bins[max_value_index]
        if max_bin_value_ > max_bin_value:
            max_bin_value = max_bin_value_

        van_hove[:, i] += hist / (n_particles * (n_particles - 1))
        van_hove_x[:, i] += hist_x / (n_particles * (n_particles - 1))
        van_hove_y[:, i] += hist_y / (n_particles * (n_particles - 1))
        van_hove_z[:, i] += hist_z / (n_particles * (n_particles - 1))

    # Create a Pandas Excel writer
    with pd.ExcelWriter('van_hove_data.xlsx') as writer:
        # Create DataFrames
        df_van_hove = pd.DataFrame(van_hove)
        df_van_hove_x = pd.DataFrame(van_hove_x)
        df_van_hove_y = pd.DataFrame(van_hove_y)
        df_van_hove_z = pd.DataFrame(van_hove_z)

        # Save DataFrames to separate sheets
        df_van_hove.to_excel(writer, sheet_name=f'{atom}_vhf_xyz', float_format='%10.4f', index=False)
        df_van_hove_x.to_excel(writer, sheet_name=f'{atom}_vhf_x', float_format='%10.4f', index=False)
        df_van_hove_y.to_excel(writer, sheet_name=f'{atom}_vhf_y', float_format='%10.4f', index=False)
        df_van_hove_z.to_excel(writer, sheet_name=f'{atom}_vhf_z', float_format='%10.4f', index=False)

    return dis, dis_x, dis_y, dis_z

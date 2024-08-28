import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from autopsy.util.calc_r import calculate_displacements

def calculate_vhf_for_lag_time(args):
    traj, lag_time, bins, n_frames, n_particles = args  # Unpack the arguments
    r = calculate_displacements(traj.reshape(n_frames, n_particles, 3), lag_time)
    hist, _ = np.histogram(r, bins=bins, density=True)

    rx = calculate_displacements(traj[:, :, 0].reshape(n_frames, n_particles, 1), lag_time)
    hist_x, _ = np.histogram(rx, bins=bins, density=True)

    ry = calculate_displacements(traj[:, :, 1].reshape(n_frames, n_particles, 1), lag_time)
    hist_y, _ = np.histogram(ry, bins=bins, density=True)

    rz = calculate_displacements(traj[:, :, 2].reshape(n_frames, n_particles, 1), lag_time)
    hist_z, _ = np.histogram(rz, bins=bins, density=True)

    max_value_index = np.max(np.where(np.abs(hist) > max(np.abs(hist)) * 0.05))
    max_bin_value_ = bins[max_value_index]

    return hist, hist_x, hist_y, hist_z, max_bin_value_

def calc_vhf(traj, atom):
    n_frames, n_particles, n_dimensions = traj.shape
    bins = np.linspace(0, 5.0, 300)  # Adjust the range and number of bins as needed

    if n_frames > 3000:
        max_lag_times = 3000
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

    with ProcessPoolExecutor(max_workers=16) as executor:
        args_list = [(traj, lag_time, bins, n_frames, n_particles) for lag_time in lag_times]
        results = list(tqdm(executor.map(calculate_vhf_for_lag_time, args_list), total=n_lag_times, disable=False))

    for i, (hist, hist_x, hist_y, hist_z, max_bin_value_) in enumerate(results):
        dis[i] = np.mean(hist)
        dis_x[i] = np.mean(np.abs(hist_x))
        dis_y[i] = np.mean(np.abs(hist_y))
        dis_z[i] = np.mean(np.abs(hist_z))

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


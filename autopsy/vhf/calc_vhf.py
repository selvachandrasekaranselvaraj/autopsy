import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from autopsy.util.calc_r import calculate_displacements
import multiprocessing
import time


# Function for parallel CSV writing
def write_csv(args):
    df, filename = args
    df.to_csv(filename, float_format='%10.4f', index=False)


# Function for calculating Van Hove functions for a specific lag time
def calculate_vhf_for_lag_time(args):
    traj, lag_time, bins, n_frames, n_particles, max_bin_value = args
    r = calculate_displacements(traj.reshape(n_frames, n_particles, 3),
                                lag_time)
    hist, _ = np.histogram(r, bins=bins, density=True)
    dis_ = [np.mean(r[:, i])
            for i in range(4)]  # displacement of four center atoms
    vhf_ = hist / (n_particles * (n_particles - 1))

    rx = calculate_displacements(
        traj[:, :, 0].reshape(n_frames, n_particles, 1), lag_time)
    hist_x, _ = np.histogram(rx, bins=bins, density=True)
    dis_x = [np.mean(rx[:, i, :]) for i in range(4)]  # np.mean(np.abs(hist_x))
    vhf_x = hist_x / (n_particles * (n_particles - 1))

    ry = calculate_displacements(
        traj[:, :, 1].reshape(n_frames, n_particles, 1), lag_time)
    hist_y, _ = np.histogram(ry, bins=bins, density=True)
    dis_y = [np.mean(ry[:, i, :]) for i in range(4)]  # np.mean(np.abs(hist_y))
    vhf_y = hist_y / (n_particles * (n_particles - 1))

    rz = calculate_displacements(
        traj[:, :, 2].reshape(n_frames, n_particles, 1), lag_time)
    hist_z, _ = np.histogram(rz, bins=bins, density=True)
    dis_z = [np.mean(rz[:, i, :]) for i in range(4)]  # np.mean(np.abs(hist_z))
    vhf_z = hist_z / (n_particles * (n_particles - 1))

    max_value_index = np.max(np.where(np.abs(hist) > max(np.abs(hist)) * 0.05))
    max_bin_value_ = bins[max_value_index]
    dis = [dis_, dis_x, dis_y, dis_z]
    vhf = [vhf_, vhf_x, vhf_y, vhf_z]
    if max_bin_value < max_bin_value_:
        max_bin_value = max_bin_value_

    return dis, vhf, max_bin_value


# Main function for calculating Van Hove functions
def calc_vhf(traj, atom, n_proc):
    n_frames, n_particles, n_dimensions = traj.shape
    bins = np.linspace(0, 5.0,
                       300)  # Adjust the range and number of bins as needed

    if n_frames > 2000:
        max_lag_times = 1500
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

    args_list = [(traj, lag_time, bins, n_frames, n_particles, max_bin_value)
                 for lag_time in lag_times]

    # Use ThreadPoolExecutor for better exception handling
    with ThreadPoolExecutor(max_workers=n_proc) as pool:
        futures = [
            pool.submit(calculate_vhf_for_lag_time, args) for args in args_list
        ]

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")

    dis = np.zeros((4, n_lag_times))
    dis_x = np.zeros((4, n_lag_times))
    dis_y = np.zeros((4, n_lag_times))
    dis_z = np.zeros((4, n_lag_times))

    for i, (dis_, vhf_, max_bin_value) in enumerate(results):
        for j in range(4):  # four atoms
            dis[j, i] = dis_[0][j]
            dis_x[j, i] = dis_[1][j]  # np.mean(np.abs(hist_x))
            dis_y[j, i] = dis_[2][j]  # np.mean(np.abs(hist_y))
            dis_z[j, i] = dis_[3][j]  # np.mean(np.abs(hist_z))

        van_hove[:, i] += vhf_[0]  # hist / (n_particles * (n_particles - 1))
        van_hove_x[:,
                   i] += vhf_[1]  # hist_x / (n_particles * (n_particles - 1))
        van_hove_y[:,
                   i] += vhf_[2]  # hist_y / (n_particles * (n_particles - 1))
        van_hove_z[:,
                   i] += vhf_[3]  # hist_z / (n_particles * (n_particles - 1))

    # Time the parallel CSV writing
    time_i = time.time()

    # Create DataFrames
    df_van_hove = pd.DataFrame(van_hove)
    df_van_hove_x = pd.DataFrame(van_hove_x)
    df_van_hove_y = pd.DataFrame(van_hove_y)
    df_van_hove_z = pd.DataFrame(van_hove_z)

    # Define CSV filenames
    csv_filenames = [
        f'{atom}_vhf_xyz.csv',
        f'{atom}_vhf_x.csv',
        f'{atom}_vhf_y.csv',
        f'{atom}_vhf_z.csv',
    ]

    # Use ThreadPoolExecutor for parallel CSV writing
    df_ = [df_van_hove, df_van_hove_x, df_van_hove_y, df_van_hove_z]
    args_list = [(df, filename) for df, filename in zip(df_, csv_filenames)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit CSV writing tasks
        futures = [executor.submit(write_csv, args) for args in args_list]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")

    # Print the time taken
    print(f"Time for VHF parallel csv: {time.time() - time_i}")

    return dis, dis_x, dis_y, dis_z
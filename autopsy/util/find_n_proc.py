import multiprocessing

def find_optimal_n_proc():
    # Get the number of available CPU cores
    num_cores = multiprocessing.cpu_count()

    # Adjust n_proc based on the number of CPU cores
    # You might need to fine-tune this based on your specific use case
    if num_cores <= 2:
        return int(num_cores)
    elif num_cores <= 4:
        return int(num_cores - 1)
    else:
        return int(num_cores - 1)


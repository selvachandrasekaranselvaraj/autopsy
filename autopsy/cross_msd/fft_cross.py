import numpy as np
from autopsy.cross_msd.cross_corr import cross_corr

def fft_cross(r, k):
    """
    Calculates "MSD" (cross-correlations) using the fast Fourier transform.

    :param r: array[float], positions of atom type 1 over time
    :param k: array[float], positions of atom type 2 over time
    :return: msd: array[float], "MSD" over time
    """
    # Calculate the length of the data sets
    N = len(r)
    
    # Compute the element-wise product of r and k, and sum along axis 1
    D = np.multiply(r, k).sum(axis=1)
    
    # Append zero to the end of D
    D = np.append(D, 0) 
    
    # Compute the cross-correlations S2 and S3 using the cross_corr function
    S2 = sum([cross_corr(r[:, i], k[:, i]) for i in range(r.shape[1])])
    S3 = sum([cross_corr(k[:, i], r[:, i]) for i in range(k.shape[1])])
    
    # Initialize Q as twice the sum of D
    Q = 2 * D.sum()
    
    # Initialize S1 as an array of zeros
    S1 = np.zeros(N)
    
    # Calculate the cross-correlation function
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N-m)
    
    # Calculate the "MSD" by subtracting S2 and S3 from S1
    msd = S1 - S2 - S3
    
    return msd

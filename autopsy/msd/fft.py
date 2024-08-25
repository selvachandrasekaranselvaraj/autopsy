import numpy as np
from autopsy.msd.autocorrFFT import autocorrFFT

def fft(r):
    """
    Computes the mean square displacement (MSD) using the fast Fourier transform.

    :param r: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    N = len(r)
    
    # Calculate the squared sum of atom positions along axis 1
    D = np.square(r).sum(axis=1)
    
    # Append 0 to D for boundary condition
    D = np.append(D, 0)
    
    # Calculate the second order autocorrelation using the provided `autocorrFFT` function
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    
    # Calculate the first order autocorrelation
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        S1[m] = Q / (N - m)
    
    # Compute the mean-squared displacement
    msd = S1 - 2 * S2
    
    return msd  # This makes avarage of x, y, and z directions

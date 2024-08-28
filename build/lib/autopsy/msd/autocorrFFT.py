import numpy as np

def autocorrFFT(x):
    """
    Calculates the autocorrelation function using the fast Fourier transform.

    :param x: array[float], function on which to compute autocorrelation function
    :return: acf: array[float], autocorrelation function
    """
    N = len(x)
    
    # Apply the fast Fourier transform
    F = np.fft.fft(x, n=2 * N)  
    PSD = F * F.conjugate()
    
    # Apply the inverse Fourier transform
    res = np.fft.ifft(PSD)
    res = (res[:N]).real   
    
    # Calculate the autocorrelation function
    n = N * np.ones(N) - np.arange(0, N) 
    acf = res / n
    
    return acf

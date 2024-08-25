import numpy as np

def cross_corr(x, y):
    """
    Calculates the cross-correlation function of x and y using the fast Fourier transform.

    :param x: array[float], data set 1
    :param y: array[float], data set 2
    :return: cf: array[float], cross-correlation function
    """   
    # Calculate the length of the data set
    N = len(x)
    
    # Apply fast Fourier transform to x and y with zero-padding to the next power of 2
    F1 = np.fft.fft(x, n=2**(N*2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2**(N*2 - 1).bit_length())
    
    # Compute the power spectral density (PSD) using the complex conjugate of F2
    PSD = F1 * F2.conjugate()
    
    # Apply the inverse fast Fourier transform to obtain the cross-correlation function
    res = np.fft.ifft(PSD)
    
    # Take the real part of the result and truncate to the original length
    res = (res[:N]).real   
    
    # Create the lag vector
    n = N * np.ones(N) - np.arange(0, N)
    
    # Calculate the cross-correlation function
    cf = res / n
    
    return cf

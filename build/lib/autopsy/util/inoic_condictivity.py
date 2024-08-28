def ionic_conductivity(slope, temperature):
    '''
    Calculate ionic conductivity using the Nernst-Einstein equation.

    :param slope: float, slope of the linear regression (cm^2/Vs)
    :param temperature: float, temperature in Kelvin
    :return: float, ionic conductivity in mS/cm
    '''
    # Constants
    q = 1.60217663e-19  # elementary charge in Coulombs
    kB = 1.380649e-23  # Boltzmann constant in J/K
    T = temperature  # temperature in Kelvin

    # Parameters
    n = 1.4048758558035422e+22  # Li/cm^3 (example value, replace with the actual concentration)
    D = slope / 6  # cm/s (example value, replace with the actual diffusion coefficient)

    # Calculate ionic conductivity using the Nernst-Einstein equation (converted to mS/cm)
    sigma = (n * q**2 * D) / (kB * T) * 1000

    # Uncomment the following line to print D and sigma for debugging
    # print(D, sigma)

    return sigma

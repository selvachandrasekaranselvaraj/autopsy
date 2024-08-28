def calc_ngp(atom_positions):
    r_sum = np.sum(atom_positions, axis = 1)
    n_atoms = np.shape(atom_positions)[1]
    ngp = ngp_fft(r_sum)
    return ngp/n_atoms
def ngp_fft(r):
    """
    Computes ngp using the fast Fourier transform.
    
    :param r: array[float], atom positions over time
    :return: msd: array[float], mean-squared displacement over time
    """
    N=len(r)
    D=np.square(r).sum(axis=1) 
    
    D=np.append(D,0) 
    D_squre=D*D
    #D_squre=np.power(r, 4).sum(axis=1)
    #D_squre=np.append(D_squre,0)
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    S2_squre=sum([autocorrFFT(r[:, i]*r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    Q_squre=2*D_squre.sum()
    S1=np.zeros(N)
    S1_squre=np.zeros(N)
    for m in tqdm(range(N)):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
        
        Q_squre=Q_squre-D_squre[m-1]-D_squre[N-m]
        S1_squre[m]=Q_squre/(N-m)  
    print(D.shape, S2.shape, Q, S1.shape)
    msd_t = S1-2*S2
    m4d_t = S1_squre-2*S2_squre
    ngp_t = ((3*m4d_t)/(5*msd_t*msd_t))-1
    return ngp_t/N
from Config import *
import numpy as np
from numpy import linalg as LA
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load DayMet dataset
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_snapshots_pod():

    fname = '../Data/Daymet_total_tmax.npy'

    data_total = np.load(fname)
    data_train = data_total[0*365:11*365] # 2000-2010
    data_valid = data_total[11*365:15*365] # 2011-2014
    data_test = data_total[15*365:] # 2016 

    
    for i in range(np.shape(data_train)[0]):
        data_train[i] = gaussian_filter(data_train[i],sigma=1)

    for i in range(np.shape(data_valid)[0]):
        data_valid[i] = gaussian_filter(data_valid[i],sigma=1)

    for i in range(np.shape(data_test)[0]):
        data_test[i] = gaussian_filter(data_test[i],sigma=1)

    dim_0 = np.shape(data_train)[0]
    dim_0_v = np.shape(data_valid)[0]
    dim_0_t = np.shape(data_test)[0]

    dim_1 = np.shape(data_train)[1]
    dim_2 = np.shape(data_train)[2]

    # Get rid of oceanic points with mask
    mask = np.zeros(shape=(dim_1,dim_2),dtype='bool')

    for i in range(dim_1):
        for j in range(dim_2):
            if data_train[0,i,j] > -1000:
                mask[i,j] = 1

    mask = mask.flatten()
    data_train = data_train.reshape(dim_0,dim_1*dim_2)
    data_valid = data_valid.reshape(dim_0_v,dim_1*dim_2)
    data_test = data_test.reshape(dim_0_t,dim_1*dim_2)

    data_train = data_train[:,mask]
    data_valid = data_valid[:,mask]
    data_test = data_test[:,mask]

    np.save('../Data/mask',mask)
    tmax_mean = np.mean(data_train,axis=0)

    data_train = (data_train-tmax_mean)
    data_valid = (data_valid-tmax_mean)
    data_test = (data_test-tmax_mean)

    np.save('../Latent_Space/POD_Snapshot_Mean.npy',tmax_mean)

    return np.transpose(data_train), np.transpose(data_valid), np.transpose(data_test)


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load prepared coefficients
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_coefficients_pod():
    phi = np.load('../Latent_Space/POD_Modes.npy')
    cf = np.load('../Latent_Space/POD_Coeffs_train.npy')
    cf_v = np.load('../Latent_Space/POD_Coeffs_valid.npy')
    cf_t = np.load('../Latent_Space/POD_Coeffs_test.npy')
    smean = np.load('../Latent_Space/POD_Snapshot_Mean.npy')

    # Do truncation
    phi = phi[:,0:num_modes] # Columns are modes
    cf = cf[0:num_modes,:] #Columns are time, rows are modal coefficients
    cf_v = cf_v[0:num_modes,:] #Columns are time, rows are modal coefficients
    cf_t = cf_t[0:num_modes,:] #Columns are time, rows are modal coefficients

    # Lowess filtering
    arr_len = np.shape(cf)[0]
    for i in range(np.shape(cf)[1]):
        cf[:,i] = lowess(cf[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)

    arr_len = np.shape(cf_v)[0]
    for i in range(np.shape(cf_v)[1]):
        cf_v[:,i] = lowess(cf_v[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)

    arr_len = np.shape(cf_t)[0]
    for i in range(np.shape(cf_t)[1]):
        cf_t[:,i] = lowess(cf_t[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)


    return phi, cf, cf_v, cf_t, smean


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate POD basis
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_pod_bases(snapshot_matrix_train,snapshot_matrix_valid,snapshot_matrix_test,num_modes): #Mean removed
    '''
    Takes input of a snapshot matrix and computes POD bases
    Outputs truncated POD bases and coefficients
    '''
    new_mat = np.matmul(np.transpose(snapshot_matrix_train),snapshot_matrix_train)

    w,v = LA.eig(new_mat)

    # Bases
    phi = np.real(np.matmul(snapshot_matrix_train,v))
    trange = np.arange(np.shape(snapshot_matrix_train)[1])
    phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

    coefficient_matrix = np.matmul(np.transpose(phi),snapshot_matrix_train)
    coefficient_matrix_valid = np.matmul(np.transpose(phi),snapshot_matrix_valid)
    coefficient_matrix_test = np.matmul(np.transpose(phi),snapshot_matrix_test)

    # Output amount of energy retained
    print('Amount of energy retained:',np.sum(w[:num_modes])/np.sum(w))

    np.save('../Latent_Space/POD_Modes.npy',phi)
    np.save('../Latent_Space/POD_Coeffs_train.npy',coefficient_matrix)
    np.save('../Latent_Space/POD_Coeffs_valid.npy',coefficient_matrix_valid)
    np.save('../Latent_Space/POD_Coeffs_test.npy',coefficient_matrix_test)

    return phi, coefficient_matrix, coefficient_matrix_valid, coefficient_matrix_test


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Plot POD modes
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def plot_pod_modes(phi,mode_num):
    plt.figure()
    plt.plot(phi[:,mode_num])
    plt.show()


if __name__ == '__main__':
    pass

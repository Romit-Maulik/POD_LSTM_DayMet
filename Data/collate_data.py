import numpy as np

for i in range(2000,2018):
    fname = 'daymet_v3_prcp_'+str(i)+'_na_prcp.npy'

    if i == 2000:
        data = np.load(fname)
    else:
        temp_data = np.load(fname)
        data = np.concatenate((data,temp_data),axis=0)

np.save('Daymet_total_prcp.npy',data)
print(np.shape(data))

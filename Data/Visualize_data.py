import numpy as np
import matplotlib.pyplot as plt 

# Load the precipitation data and mask
year = 15
dataset = np.load('Daymet_total_prcp_old.npy')
mask = np.load('mask_prcp.npy')

valid_list = []

for i in range(np.shape(dataset)[0]):
    if np.sum(dataset[i]) < -1.0:
        valid_list.append(i)
    
dataset_valid = dataset[valid_list]

np.save('Daymet_total_prcp.npy',dataset_valid)

fig, ax = plt.subplots(ncols=3)
cx0 = ax[0].imshow(dataset_valid[100],vmin=-20,vmax=40)
cx1 = ax[1].imshow(dataset_valid[101],vmin=-20,vmax=40)
cx2 = ax[2].imshow(dataset_valid[102],vmin=-20,vmax=40)

fig.colorbar(cx0,ax=ax[0],fraction=0.046, pad=0.04)
fig.colorbar(cx1,ax=ax[1],fraction=0.046, pad=0.04)
fig.colorbar(cx2,ax=ax[2],fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('Viz_check.png')
plt.close()
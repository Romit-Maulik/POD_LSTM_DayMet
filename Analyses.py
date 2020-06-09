from Config import *
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Visualize DayMet dataset
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def visualize_predictions(cf_pred,cf_true,sm,phi,mode):

    #Visualization of state predictions for 4 modes
    fig,ax = plt.subplots(nrows=4)
    mode_num = 0
    ax[0].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[0].plot(cf_pred[mode_num,:-window_len],label='POD-LSTM')

    mode_num = 1
    ax[1].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[1].plot(cf_pred[mode_num,:-window_len],label='POD-LSTM')

    mode_num = 2
    ax[2].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[2].plot(cf_pred[mode_num,:-window_len],label='POD-LSTM')

    mode_num = 3
    ax[3].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[3].plot(cf_pred[mode_num,:-window_len],label='POD-LSTM')

    plt.legend()
    plt.title('Forecast ' + mode)
    plt.tight_layout()
    plt.savefig('./Visualization/Coefficients/Coefficients_'+mode+'.png')
    plt.close()

    if field_viz:
        if mode == 'train':
            tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
        elif mode == 'valid':
            tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
        elif mode == 'test':
            tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[15*365:] # 2016-
   
        mask = np.load('./Data/mask.npy')
        dim_0 = np.shape(tmax)[0]
        dim_1 = np.shape(tmax)[1]
        dim_2 = np.shape(tmax)[2]

        tmax = tmax.reshape(dim_0,dim_1*dim_2)

        # Reconstruct
        prediction = sm+np.transpose(np.matmul(phi,cf_pred))
        true_pod = sm+np.transpose(np.matmul(phi,cf_true))

        tmax_pod = np.copy(tmax)
        tmax_pred = np.copy(tmax)

        tmax_pod[:,mask] = true_pod[:,:]
        tmax_pred[:,mask] = prediction[:,:]

        tmax = tmax.reshape(dim_0,dim_1,dim_2)
        tmax_pod = tmax_pod.reshape(dim_0,dim_1,dim_2)
        tmax_pred = tmax_pred.reshape(dim_0,dim_1,dim_2)

        pnum = 0
        for t in range(window_len,dim_0,30):
            fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,12))
            cx = ax[0].imshow(tmax[t,:,:],vmin=-20,vmax=40)#sst_pod_lstm
            ax[0].set_title('True')

            ax[1].imshow(tmax_pod[t,:,:],vmin=-20,vmax=40)#sst_pod_lstm
            ax[1].set_title('Projected true')

            ax[2].imshow(tmax_pred[t,:,:],vmin=-20,vmax=40)#sst_pod_lstm
            ax[2].set_title('Predicted')

            fig.colorbar(cx, ax = ax[0],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[1],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[2],fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig('./Visualization/Contours/Plot_'+mode+'_'+"{0:0>4}".format(pnum)+'.png')
            pnum = pnum + 1
            plt.close()


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Perform bias analyses
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def analyze_predictions(cf_pred,cf_true,sm,phi,mode):

    if mode == 'train':
        tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
    elif mode == 'valid':
        tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
    elif mode == 'test':
        tmax = np.load('./Data/Daymet_total.npy',allow_pickle=True)[15*365:] # 2016-

    mask = np.load('./Data/mask.npy')
    dim_0 = np.shape(tmax)[0]
    dim_1 = np.shape(tmax)[1]
    dim_2 = np.shape(tmax)[2]

    tmax = tmax.reshape(dim_0,dim_1*dim_2)

    # Reconstruct
    prediction = sm+np.transpose(np.matmul(phi,cf_pred))
    true_pod = sm+np.transpose(np.matmul(phi,cf_true))

    tmax_pod = np.copy(tmax)
    tmax_pred = np.copy(tmax)

    tmax_pod[:,mask] = true_pod[:,:]
    tmax_pred[:,mask] = prediction[:,:]

    tmax = tmax.reshape(dim_0,dim_1,dim_2)
    tmax_pod = tmax_pod.reshape(dim_0,dim_1,dim_2)
    tmax_pred = tmax_pred.reshape(dim_0,dim_1,dim_2)

    # import subregions
    import os
    files = os.listdir('./Analyses/region_masks')

    for file in files:
        if file.endswith('.nc'):
            region_name = file.split('.')[0]
            print('Region:',region_name)

            # load masks
            region_mask = np.load('./Analyses/region_masks/'+str(region_name)+'_mask.npy')
            # Get data points
            true_data_list = []
            for i in range(np.shape(tmax)[0]):
                region_temps = tmax[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                true_data_list.append(region_temps)
            true_data = np.asarray(true_data_list)

            # Get data points
            pod_data_list = []
            for i in range(np.shape(tmax)[0]):
                region_temps = tmax_pod[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                pod_data_list.append(region_temps)
            pod_data = np.asarray(pod_data_list)

            # Get data points
            pred_data_list = []
            for i in range(np.shape(tmax)[0]):
                region_temps = tmax_pred[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                pred_data_list.append(region_temps)
            pred_data = np.asarray(pred_data_list)

            
            day_list = [60,120,180,240,300,360]
            binwidth = 1

            for day in day_list:
                # PDFs of regions
                if len(true_data[day]) != 0: # For any purely oceanic regions
                    plt.figure(figsize=(6,6))
                    plt.hist(true_data[day],bins=np.arange(min(true_data[day]), max(true_data[day]) + binwidth, binwidth),label='True',alpha = 0.5)
                    plt.hist(pod_data[day],bins=np.arange(min(true_data[day]), max(true_data[day]) + binwidth, binwidth),label='Projected',alpha=0.5)
                    plt.hist(pred_data[day],bins=np.arange(min(true_data[day]), max(true_data[day]) + binwidth, binwidth),label='Predicted',alpha=0.5)
                    plt.legend()
                    plt.xlim((-50,60))
                    plt.savefig('./Analyses/pdfs/'+region_name+'_'+str(day)+'.png')
                    plt.close()

            # 7-day biases of regions
            bias_vals = []
            true_ave_temp_vals = []
            pred_ave_temp_vals = []
            pod_ave_temp_vals = []

            for week in range(0,np.shape(true_data)[0]//7):
                true_ave_temp = 0.0
                pred_ave_temp = 0.0
                pod_ave_temp = 0.0

                for day in range(week*7,(week+1)*7):
                    true_ave_temp = np.mean(true_data[day]) + true_ave_temp
                    pred_ave_temp = np.mean(pred_data[day]) + pred_ave_temp
                    pod_ave_temp = np.mean(pod_data[day]) + pod_ave_temp

                true_ave_temp = true_ave_temp/7.0
                pred_ave_temp = pred_ave_temp/7.0

                true_ave_temp_vals.append(true_ave_temp)
                pred_ave_temp_vals.append(pred_ave_temp)
                pod_ave_temp_vals.append(pod_ave_temp)

                bias_vals.append(true_ave_temp-pred_ave_temp)
                
            plt.figure()
            plt.plot(np.arange(len(bias_vals)),bias_vals,'o')
            plt.xlabel('Week')
            plt.ylabel('Bias in celsius')
            plt.savefig('./Analyses/biases/Bias_'+region_name+'.png')
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(true_ave_temp_vals)),true_ave_temp_vals,'o',label='True')
            plt.plot(np.arange(len(pred_ave_temp_vals)),pred_ave_temp_vals,'o',label='Predicted')
            plt.xlabel('Week')
            plt.ylabel('Weekly average temperatures')
            plt.legend()
            plt.savefig('./Analyses/biases/Temperature_'+region_name+'.png')
            plt.close()


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Subregion mask generator
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def import_subregions():
    # import subregions
    import os
    files = os.listdir('./Analyses/region_masks')

    # Load the full field
    lat_full = np.load('./Data/daymet_v3_tmax_2015_na_lat.npy')
    lon_full = np.load('./Data/daymet_v3_tmax_2015_na_lon.npy')
    data_full = np.load('./Data/daymet_v3_tmax_2014_na_tmax.npy')

    data_full[data_full>-1000] = 100.0

    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0
    
    for file in files:
        if file.endswith('.nc'):
            region_name = file.split('.')[0]
            print('Region:',region_name)
            region = Dataset('./Analyses/region_masks/'+file, 'r')  # Dataset is the class behavior to open the file and create an instance of the ncCDF4 class

            # Data
            lat = region.variables["lat"][:]
            lon = -360.0+region.variables["lon"][:]
            mask = region.variables["mask"][:]

            lat_grid, lon_grid = np.meshgrid(lat,lon)
            lat_grid = lat_grid[mask.T==1]
            lon_grid = lon_grid[mask.T==1]

            full_grid_points = np.concatenate((lat_full.flatten().reshape(-1,1),lon_full.flatten().reshape(-1,1)),axis=-1)
            region_grid_points = np.concatenate((lat_grid.reshape(-1,1),lon_grid.reshape(-1,1)),axis=-1)

            region_mask = in_hull(full_grid_points,np.array(region_grid_points)).reshape(np.shape(data_full[0]))

            # Save regions to check
            plt.figure()
            cx = plt.imshow(data_full[0],vmin=-20,vmax=70)
            plt.imshow(100.0*region_mask.astype('double'),vmin=-20,vmax=70,alpha=0.5)
            # plt.colorbar(cx)
            plt.savefig('./Analyses/region_masks/'+region_name+'.png')
            plt.close()

            # Save masks
            np.save('./Analyses/region_masks/'+str(region_name)+'_mask.npy',region_mask)


if __name__ == '__main__':
    import_subregions()

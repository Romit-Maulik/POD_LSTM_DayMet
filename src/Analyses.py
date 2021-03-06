from Config import *
import numpy as np
import matplotlib.pyplot as plt
from CAE import cae_model
from tensorflow.keras import backend as K
#from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Visualize DayMet dataset
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def visualize_predictions_pod(cf_pred,cf_true,sm,phi,mode):

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
    plt.savefig('../Visualization/POD/Coefficients/Coefficients_'+mode+'.png')
    plt.close()

    if field_viz:

        if mode == 'train':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
        elif mode == 'valid':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
        elif mode == 'test':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[15*365:] # 2016-   

   
        mask = np.load('../Data/mask.npy')
        dim_0 = np.shape(snapshots)[0]
        dim_1 = np.shape(snapshots)[1]
        dim_2 = np.shape(snapshots)[2]

        snapshots = snapshots.reshape(dim_0,dim_1*dim_2)

        # Reconstruct
        prediction = sm+np.transpose(np.matmul(phi,cf_pred))
        true_pod = sm+np.transpose(np.matmul(phi,cf_true))

        snapshots_pod = np.copy(snapshots)
        snapshots_pred = np.copy(snapshots)

        snapshots_pod[:,mask] = true_pod[:,:]
        snapshots_pred[:,mask] = prediction[:,:]

        snapshots = snapshots.reshape(dim_0,dim_1,dim_2)
        snapshots_pod = snapshots_pod.reshape(dim_0,dim_1,dim_2)
        snapshots_pred = snapshots_pred.reshape(dim_0,dim_1,dim_2)

        pnum = 0
        for t in range(window_len,dim_0,30):
            fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,12))
            cx = ax[0].imshow(snapshots[t,:,:],vmin=-20,vmax=40)
            ax[0].set_title('True')

            ax[1].imshow(snapshots_pod[t,:,:],vmin=-20,vmax=40)
            ax[1].set_title('Projected true')

            ax[2].imshow(snapshots_pred[t,:,:],vmin=-20,vmax=40)
            ax[2].set_title('Predicted')

            fig.colorbar(cx, ax = ax[0],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[1],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[2],fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig('../Visualization/POD/Contours/Plot_'+mode+'_'+"{0:0>4}".format(pnum)+'.png')
            pnum = pnum + 1
            plt.close()


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Perform bias analyses
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def analyze_predictions_pod(cf_pred,cf_true,sm,phi,mode):
      
    if mode == 'train':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
    elif mode == 'valid':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
    elif mode == 'test':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[15*365:] # 2016-   

    mask = np.load('../Data/mask.npy')
    dim_0 = np.shape(snapshots)[0]
    dim_1 = np.shape(snapshots)[1]
    dim_2 = np.shape(snapshots)[2]

    snapshots = snapshots.reshape(dim_0,dim_1*dim_2)

    # Reconstruct
    prediction = sm+np.transpose(np.matmul(phi,cf_pred))
    true_pod = sm+np.transpose(np.matmul(phi,cf_true))

    snapshots_pod = np.copy(snapshots)
    snapshots_pred = np.copy(snapshots)

    snapshots_pod[:,mask] = true_pod[:,:]
    snapshots_pred[:,mask] = prediction[:,:]

    snapshots = snapshots.reshape(dim_0,dim_1,dim_2)
    snapshots_pod = snapshots_pod.reshape(dim_0,dim_1,dim_2)
    snapshots_pred = snapshots_pred.reshape(dim_0,dim_1,dim_2)

    # import subregions
    import os
    files = os.listdir('../Analyses/region_masks')

    for file in files:
        if file.endswith('.nc'):
            region_name = file.split('.')[0]
            print('Region:',region_name)

            # load masks
            region_mask = np.load('../Analyses/region_masks/'+str(region_name)+'_mask.npy')
            # Get data points
            true_data_list = []
            for i in range(np.shape(snapshots)[0]):
                region_temps = snapshots[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                true_data_list.append(region_temps)
            true_data = np.asarray(true_data_list)

            # Get data points
            pod_data_list = []
            for i in range(np.shape(snapshots)[0]):
                region_temps = snapshots_pod[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                pod_data_list.append(region_temps)
            pod_data = np.asarray(pod_data_list)

            # Get data points
            pred_data_list = []
            for i in range(np.shape(snapshots)[0]):
                region_temps = snapshots_pred[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                pred_data_list.append(region_temps)
            pred_data = np.asarray(pred_data_list)

            binwidth = 1
            for week in range(0,np.shape(true_data)[0]//7,4):
                
                for day in range(week*7,(week+1)*7):
                    
                    if day == 0:
                        true_arr = true_data[day]
                        pod_arr = pod_data[day]
                        pred_arr = pred_data[day]
                    else:
                        true_arr = np.append(true_arr,true_data[day])
                        pod_arr = np.append(pod_arr,pod_data[day])
                        pred_arr = np.append(pred_arr,pred_data[day])

                try:
    
                    plt.figure(figsize=(6,6))
                    plt.hist(true_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='True',alpha = 0.5)
                    plt.hist(pod_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='Projected',alpha=0.5)
                    plt.hist(pred_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='Predicted',alpha=0.5)
                    plt.legend()
                    plt.xlim((-50,60))
                    plt.savefig('../Analyses/POD/pdfs/'+region_name+'_'+str(week)+'.png')
                    plt.close()
                
                except:
                    print('Issue with region:',region_name)

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
            plt.ylabel('Bias')
            plt.savefig('../Analyses/POD/biases/Bias_'+region_name+'.png')
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(true_ave_temp_vals)),true_ave_temp_vals,'o',label='True')
            plt.plot(np.arange(len(pred_ave_temp_vals)),pred_ave_temp_vals,'o',label='Predicted')
            plt.xlabel('Week')
            plt.ylabel('Weekly average forecasts')
            plt.legend()
            plt.savefig('../Analyses/POD/biases/Forecasts_'+region_name+'.png')
            plt.close()


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Visualize DayMet dataset - CAE
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def visualize_predictions_cae(cf_pred,cf_true,mode):

    #Visualization of state predictions for 4 modes
    fig,ax = plt.subplots(nrows=4)
    mode_num = 0
    ax[0].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[0].plot(cf_pred[mode_num,:-window_len],label='CAE-LSTM')

    mode_num = 1
    ax[1].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[1].plot(cf_pred[mode_num,:-window_len],label='CAE-LSTM')

    mode_num = 2
    ax[2].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[2].plot(cf_pred[mode_num,:-window_len],label='CAE-LSTM')

    mode_num = 3
    ax[3].plot(cf_true[mode_num,:-window_len],label='Truth')
    ax[3].plot(cf_pred[mode_num,:-window_len],label='CAE-LSTM')

    plt.legend()
    plt.title('Forecast ' + mode)
    plt.tight_layout()
    plt.savefig('../Visualization/CAE/Coefficients/Coefficients_'+mode+'.png')
    plt.close()

    if field_viz:

        if mode == 'train':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
        elif mode == 'valid':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
        elif mode == 'test':
            snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[15*365:] # 2016-


        # loading the CAE preprocessor
        from sklearn.externals import joblib
        scaler_filename = "cae_scaler.save"
        preproc = joblib.load(scaler_filename) 
      
        # Load the model
        weights_filepath = "../CAE_Training/cae_best_weights.h5"
        model,_,decoder = cae_model()
        model.load_weights(weights_filepath)

        cf_true = np.transpose(cf_true)
        cf_pred = np.transpose(cf_pred)

        pnum = 0
        for t in range(window_len,np.shape(cf_pred)[0],30):
            fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,12))
            cx = ax[0].imshow(snapshots[t,:,:],vmin=-20,vmax=40)
            ax[0].set_title('True')

            # Reconstruct from latent space
            snapshot_pred = K.eval(decoder(cf_true[t:t+1].reshape(1,2,2,1).astype('float32')))
            snapshot_pred = snapshot_pred[0,512-404:512+404,512-391:512+391,0]
            snapshot_pred = snapshot_pred.reshape(1,808*782)
            snapshot_pred = preproc.inverse_transform(snapshot_pred)
            snapshot_pred = snapshot_pred.reshape(808,782)
            # Plot
            ax[1].imshow(snapshot_pred,vmin=-20,vmax=40)
            ax[1].set_title('Reconstructed')

            # Reconstruct from latent space
            snapshot_pred = K.eval(decoder(cf_pred[t:t+1].reshape(1,2,2,1).astype('float32')))
            snapshot_pred = snapshot_pred[0,512-404:512+404,512-391:512+391,0]
            snapshot_pred = snapshot_pred.reshape(1,808*782)
            snapshot_pred = preproc.inverse_transform(snapshot_pred)
            snapshot_pred = snapshot_pred.reshape(808,782)

            ax[2].imshow(snapshot_pred,vmin=-20,vmax=40)
            ax[2].set_title('Reconstructed+Predicted')

            fig.colorbar(cx, ax = ax[0],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[1],fraction=0.046, pad=0.04)
            fig.colorbar(cx, ax = ax[2],fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig('../Visualization/CAE/Contours/Plot_'+mode+'_'+"{0:0>4}".format(pnum)+'.png')
            pnum = pnum + 1
            plt.close()

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Perform bias analyses
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def analyze_predictions_cae(cf_pred,cf_true,mode):
        
    if mode == 'train':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[0*365:11*365] # 2000-2010
    elif mode == 'valid':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[11*365:15*365] # 2011-2015
    elif mode == 'test':
        snapshots = np.load('../Data/Daymet_total_tmax.npy',allow_pickle=True)[15*365:] # 2016-  

    # loading the CAE preprocessor
    from sklearn.externals import joblib
    scaler_filename = "cae_scaler.save"
    preproc = joblib.load(scaler_filename) 
    
    # Load the model
    weights_filepath = "../CAE_Training/cae_best_weights.h5"
    model,_,decoder = cae_model()
    model.load_weights(weights_filepath)

    # Coefficients in latent space
    cf_true = np.transpose(cf_true)
    cf_pred = np.transpose(cf_pred)

    # import subregions
    import os
    files = os.listdir('../Analyses/region_masks')

    for file in files:
        if file.endswith('.nc'):
            region_name = file.split('.')[0]
            print('Region:',region_name)

            # load masks
            region_mask = np.load('../Analyses/region_masks/'+str(region_name)+'_mask.npy')
            # Get data points
            true_data_list = []
            for i in range(np.shape(snapshots)[0]):
                region_temps = snapshots[i][region_mask]
                region_temps = region_temps[region_temps>-150]
                true_data_list.append(region_temps)
            true_data = np.asarray(true_data_list)

            # Get data points
            dec_data_list = []
            for i in range(np.shape(snapshots)[0]):
                # Reconstruct from latent space
                snapshot_pred = K.eval(decoder(cf_true[i:i+1].reshape(1,2,2,1).astype('float32')))
                snapshot_pred = snapshot_pred[0,512-404:512+404,512-391:512+391,0]
                snapshot_pred = snapshot_pred.reshape(1,808*782)
                snapshot_pred = preproc.inverse_transform(snapshot_pred)
                snapshot_pred = snapshot_pred.reshape(808,782)

                region_temps = snapshot_pred[region_mask]
                region_temps = region_temps[region_temps>-150]
                dec_data_list.append(region_temps)
            dec_data = np.asarray(dec_data_list)

            # Get data points
            pred_data_list = []
            for i in range(np.shape(snapshots)[0]):
                # Reconstruct from latent space
                snapshot_pred = K.eval(decoder(cf_pred[i:i+1].reshape(1,2,2,1).astype('float32')))
                snapshot_pred = snapshot_pred[0,512-404:512+404,512-391:512+391,0]
                snapshot_pred = snapshot_pred.reshape(1,808*782)
                snapshot_pred = preproc.inverse_transform(snapshot_pred)
                snapshot_pred = snapshot_pred.reshape(808,782)


                region_temps = snapshot_pred[region_mask]
                region_temps = region_temps[region_temps>-150]
                pred_data_list.append(region_temps)
            pred_data = np.asarray(pred_data_list)

            binwidth = 1
            for week in range(0,np.shape(true_data)[0]//7,4):
                
                for day in range(week*7,(week+1)*7):
                    
                    if day == 0:
                        true_arr = true_data[day]
                        dec_arr = dec_data[day]
                        pred_arr = pred_data[day]
                    else:
                        true_arr = np.append(true_arr,true_data[day])
                        dec_arr = np.append(dec_arr,dec_data[day])
                        pred_arr = np.append(pred_arr,pred_data[day])

                try:
                    plt.figure(figsize=(6,6))
                    plt.hist(true_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='True',alpha = 0.5)
                    plt.hist(dec_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='Reconstructed',alpha=0.5)
                    plt.hist(pred_arr,bins=np.arange(min(true_arr), max(true_arr) + binwidth, binwidth),label='Predicted',alpha=0.5)
                    plt.legend()
                    plt.xlim((-50,60))
                    plt.savefig('../Analyses/CAE/pdfs/'+region_name+'_'+str(week)+'.png')
                    plt.close()
                
                except:
                    print('Issue with region:',region_name)

            # 7-day biases of regions
            bias_vals = []
            true_ave_temp_vals = []
            pred_ave_temp_vals = []
            dec_ave_temp_vals = []

            for week in range(0,np.shape(true_data)[0]//7):
                true_ave_temp = 0.0
                pred_ave_temp = 0.0
                dec_ave_temp = 0.0

                for day in range(week*7,(week+1)*7):
                    true_ave_temp = np.mean(true_data[day]) + true_ave_temp
                    pred_ave_temp = np.mean(pred_data[day]) + pred_ave_temp
                    dec_ave_temp = np.mean(dec_data[day]) + dec_ave_temp

                true_ave_temp = true_ave_temp/7.0
                pred_ave_temp = pred_ave_temp/7.0

                true_ave_temp_vals.append(true_ave_temp)
                pred_ave_temp_vals.append(pred_ave_temp)
                dec_ave_temp_vals.append(dec_ave_temp)

                bias_vals.append(true_ave_temp-pred_ave_temp)
                
            plt.figure()
            plt.plot(np.arange(len(bias_vals)),bias_vals,'o')
            plt.xlabel('Week')
            plt.ylabel('Bias')
            plt.savefig('../Analyses/CAE/biases/Bias_'+region_name+'.png')
            plt.close()

            plt.figure()
            plt.plot(np.arange(len(true_ave_temp_vals)),true_ave_temp_vals,'o',label='True')
            plt.plot(np.arange(len(pred_ave_temp_vals)),pred_ave_temp_vals,'o',label='Predicted')
            plt.xlabel('Week')
            plt.ylabel('Weekly average forecasts')
            plt.legend()
            plt.savefig('../Analyses/CAE/biases/Forecasts_'+region_name+'.png')
            plt.close()

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Subregion mask generator
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def import_subregions():
    # import subregions
    import os
    files = os.listdir('../Analyses/region_masks')

    # Load the full field
    lat_full = np.load('../Data/lat.npy')
    lon_full = np.load('../Data/lon.npy')
    data_full = np.load('../Data/Daymet_total_tmax.npy')[0:365]

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
            region = Dataset('../Analyses/region_masks/'+file, 'r')  # Dataset is the class behavior to open the file and create an instance of the ncCDF4 class

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
            plt.savefig('../Analyses/region_masks/'+region_name+'.png')
            plt.close()

            # Save masks
            np.save('../Analyses/region_masks/'+str(region_name)+'_mask.npy',region_mask)


if __name__ == '__main__':
    import_subregions()

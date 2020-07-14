import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--geo_data",help="Data sets to use ['tmax', 'prcp']",type=str)
parser.add_argument("--train_mode",help="Retrain POD/CAE/LSTM?",action='store_true')

parser.add_argument("--comp",help="Compression to use ['pod', 'cae']",type=str)
parser.add_argument("--train_space",help="Train compression",action='store_true')
parser.add_argument("--epochs_space",help="Number of epochs for training CAE",type=int)
parser.add_argument("--batch_space",help="Batch size of CAE training",type=int)

parser.add_argument("--train_time",help="Train LSTM",action='store_true')
parser.add_argument("--epochs_time",help="Number of epochs for training LSTM",type=int)
parser.add_argument("--batch_time",help="Batch size of LSTM training",type=int)

parser.add_argument("--pod_modes",help="Number of modes to retain (if POD)",type=int)
parser.add_argument("--viz", help="Visualize reconstruction", action='store_true')
parser.add_argument("--win",help="Length of forecast window (always needed)",type=int)

parser.add_argument("--use_hvd",help="Use horovod (only with CAE now)",action='store_true')

args = parser.parse_args()

# Import and tweak configuration
import Config

# Dataset
if args.geo_data is not None:
    Config.geo_data = args.geo_data
if args.comp is not None:
    Config.compression = args.comp
if args.train_mode is not None:
    Config.train_mode = args.train_mode
if args.use_hvd is not None:
    Config.use_hvd = args.use_hvd

# Deployment modes
if args.train_space is not None:
    Config.space_train_mode = args.train_space # test or train CAE
if args.train_time is not None:
    Config.time_train_mode = args.train_time # test or train LSTM

# Number of epochs to train LSTM
if args.epochs_time is not None:
    Config.num_epochs_time = args.epochs_time
# Number of epochs to train CAE
if args.epochs_space is not None:
    Config.num_epochs_space = args.epochs_space
# Batch size of CAE training
if args.batch_space is not None:
    Config.batchsize_space = args.batch_space
# Batch size of LSTM training
if args.batch_time is not None:
    Config.batchsize_time = args.batch_time

# Number of POD modes
if args.pod_modes is not None:
    Config.num_modes = args.pod_modes

# Window length of forecast
if args.win is not None:
    Config.window_len = args.win

# Field visualization
Config.field_viz = args.viz

# Import the configuration finally
from Config import *

# Import libraries
import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.random.set_seed(5)
tf.keras.backend.set_floatx('float32')

from POD import generate_pod_bases, plot_pod_modes, load_snapshots_pod, load_coefficients_pod
from CAE import generate_cae, load_coefficients_cae, load_snapshots_cae
from ML_Time import lstm_for_dynamics, evaluate_rom_deployment_lstm
from Analyses import visualize_predictions_pod, analyze_predictions_pod
from Analyses import visualize_predictions_cae, analyze_predictions_cae

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Scaling for LSTM
preproc_input = Pipeline([('minmaxscaler', MinMaxScaler())])

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the main file for POD-LSTM assessment
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if args.use_hvd:
        import horovod.tensorflow.keras as hvd
        hvd.init()
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if train_mode:

        if compression == 'pod':
            # Perform POD
            if space_train_mode:
                # Snapshot collection
                sm_train, sm_valid, sm_test = load_snapshots_pod() # Note that columns of a snapshot/state are time always and a state vector is a column vector
                # Eigensolve
                generate_pod_bases(sm_train,sm_valid,sm_test,num_modes)
                phi, cf, cf_v, cf_t, _ = load_coefficients_pod()
            else:
                phi, cf, cf_v, cf_t, _ = load_coefficients_pod()

            if time_train_mode:
                # Train LSTM
                cf = preproc_input.fit_transform(np.transpose(cf))
                cf_v = preproc_input.transform(np.transpose(cf_v)) # Times are rows

                # LSTM network
                model = lstm_for_dynamics(cf,cf_v,num_epochs_time,window_len,time_train_mode)

        elif compression == 'cae':
            # Perform CAE training
            if space_train_mode:
                # Snapshot collection
                zero_pad_train, zero_pad_test, preproc, dim_1, dim_2 = load_snapshots_cae() 
                # Train CAE
                generate_cae(zero_pad_train, zero_pad_test, preproc, dim_1, dim_2, space_train_mode,args.use_hvd)
            else:
                cf, cf_t = load_coefficients_cae()

            # Train LSTM
            if time_train_mode:
                num_points = np.shape(cf)[0]
               
                cf_v = cf[int(0.9*num_points):]
                cf = cf[:int(0.9*num_points)]
                
                cf = preproc_input.fit_transform(cf)
                cf_v = preproc_input.transform(cf_v)

                # LSTM network
                model = lstm_for_dynamics(cf,cf_v,num_epochs_time,window_len,time_train_mode)

    else:
        if compression == 'pod':

            # Load data for testing
            phi, cf, cf_v, cf_t, smean = load_coefficients_pod()

            cf = preproc_input.fit_transform(np.transpose(cf))
            cf_t = preproc_input.transform(np.transpose(cf_t))
            cf_v = preproc_input.transform(np.transpose(cf_v))

            # LSTM network on training data
            model = lstm_for_dynamics(cf,cf_v,num_epochs_time,window_len,train_mode)

            tsteps = np.shape(cf)[0]
            _, lstm = evaluate_rom_deployment_lstm(model,cf,tsteps,num_modes,window_len)

            tsteps = np.shape(cf_v)[0]
            _, lstm_v = evaluate_rom_deployment_lstm(model,cf_v,tsteps,num_modes,window_len)

            tsteps = np.shape(cf_t)[0]
            _, lstm_t = evaluate_rom_deployment_lstm(model,cf_t,tsteps,num_modes,window_len)

            # Metrics
            print('MAE metrics on train data:',mean_absolute_error(cf[:-window_len],lstm[:-window_len]))
            print('MAE metrics on valid data:',mean_absolute_error(cf_v[:-window_len],lstm_v[:-window_len]))
            print('MAE metrics on test data:',mean_absolute_error(cf_t[:-window_len],lstm_t[:-window_len]))

            print('R2 metrics on train data:',r2_score(cf[:-window_len],lstm[:-window_len]))
            print('R2 metrics on valid data:',r2_score(cf_v[:-window_len],lstm_v[:-window_len]))
            print('R2 metrics on test data:',r2_score(cf_t[:-window_len],lstm_t[:-window_len]))

            # Rescale and save
            # Train
            cf = np.transpose(preproc_input.inverse_transform(cf))
            lstm = np.transpose(preproc_input.inverse_transform(lstm))
            np.save('../Latent_Space/POD_Prediction_train_'+str(geo_data)+'.npy',lstm)

            # Valid
            cf_v = np.transpose(preproc_input.inverse_transform(cf_v))
            lstm_v = np.transpose(preproc_input.inverse_transform(lstm_v))
            np.save('../Latent_Space/POD_Prediction_valid_'+str(geo_data)+'.npy',lstm_v)

            # Test
            cf_t = np.transpose(preproc_input.inverse_transform(cf_t))
            lstm_t = np.transpose(preproc_input.inverse_transform(lstm_t))
            np.save('../Latent_Space/POD_Prediction_test_'+str(geo_data)+'.npy',lstm_t)

            # # Visualize train
            # visualize_predictions(lstm,cf,smean,phi,'train')
            # # Visualize valid
            # visualize_predictions(lstm_v,cf_v,smean,phi,'valid')
            
            # Visualize and analyze test
            visualize_predictions_pod(lstm_t,cf_t,smean,phi,'test')
            # analyze_predictions_pod(lstm_t,cf_t,smean,phi,'test')

        elif compression == 'cae':

            cf, cf_t = load_coefficients_cae()
            num_points = np.shape(cf)[0]
            # LSTM network
            cf_v = cf[int(0.9*num_points):]
            cf = cf[:int(0.9*num_points)]
            
            cf = preproc_input.fit_transform(cf)
            cf_v = preproc_input.transform(cf_v)
            cf_t = preproc_input.transform(cf_t)
            
            # LSTM network on training data
            model = lstm_for_dynamics(cf,cf_v,num_epochs_time,window_len,train_mode)

            tsteps = np.shape(cf)[0]
            _, lstm = evaluate_rom_deployment_lstm(model,cf,tsteps,num_modes,window_len)

            tsteps = np.shape(cf_v)[0]
            _, lstm_v = evaluate_rom_deployment_lstm(model,cf_v,tsteps,num_modes,window_len)

            tsteps = np.shape(cf_t)[0]
            _, lstm_t = evaluate_rom_deployment_lstm(model,cf_t,tsteps,num_modes,window_len)

            # Metrics
            print('MAE metrics on train data:',mean_absolute_error(cf[:-window_len],lstm[:-window_len]))
            print('MAE metrics on valid data:',mean_absolute_error(cf_v[:-window_len],lstm_v[:-window_len]))
            print('MAE metrics on test data:',mean_absolute_error(cf_t[:-window_len],lstm_t[:-window_len]))

            print('R2 metrics on train data:',r2_score(cf[:-window_len],lstm[:-window_len]))
            print('R2 metrics on valid data:',r2_score(cf_v[:-window_len],lstm_v[:-window_len]))
            print('R2 metrics on test data:',r2_score(cf_t[:-window_len],lstm_t[:-window_len]))

            # Rescale and save
            # Train
            cf = np.transpose(preproc_input.inverse_transform(cf))
            lstm = np.transpose(preproc_input.inverse_transform(lstm))
            np.save('../Latent_Space/CAE_Prediction_train_'+str(geo_data)+'.npy',lstm)

            # Valid
            cf_v = np.transpose(preproc_input.inverse_transform(cf_v))
            lstm_v = np.transpose(preproc_input.inverse_transform(lstm_v))
            np.save('../Latent_Space/CAE_Prediction_valid_'+str(geo_data)+'.npy',lstm_v)

            # Test
            cf_t = np.transpose(preproc_input.inverse_transform(cf_t))
            lstm_t = np.transpose(preproc_input.inverse_transform(lstm_t))
            np.save('../Latent_Space/CAE_Prediction_test_'+str(geo_data)+'.npy',lstm_t)

            # Visualize and analyze test
            visualize_predictions_cae(lstm_t,cf_t,'test')
            analyze_predictions_cae(lstm_t,cf_t,'test')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data",help="Data sets to use ['tmax', 'prcp']",type=str)
parser.add_argument("--modes",help="Number of POD modes to retain (always needed)",type=int)
parser.add_argument("--train",help="Train LSTM",action='store_true')
parser.add_argument('--pod', help="Run a POD on the data", action='store_true')
parser.add_argument('--viz', help="Visualize reconstruction", action='store_true')
parser.add_argument("--epochs",help="Number of epochs for training",type=int)
parser.add_argument("--win",help="Length of forecast window (always needed)",type=int)
parser.add_argument("--batch",help="Batch size of LSTM training",type=int)
args = parser.parse_args()

# Import and tweak configuration
import Config

# Dataset
if args.data is not None:
    Config.geo_data = args.data
# Number of modes
if args.modes is not None:
    Config.num_modes = args.modes
# Window length of forecast
if args.win is not None:
    Config.window_len = args.win
# Number of epochs to train
if args.epochs is not None:
    Config.num_epochs = args.epochs
# Batch size of LSTM training
if args.batch is not None:
    Config.batchsize = args.batch

# Deployment mode
Config.train_mode = args.train # test or train
# Field visualization
Config.field_viz = args.viz
# Perform POD?
Config.perform_pod = args.pod

# Import the configuration finally
from Config import *

# Import libraries
import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.random.set_seed(5)

from POD import generate_pod_bases, plot_pod_modes, load_snapshots_pod, load_coefficients_pod
from ML import lstm_for_dynamics, evaluate_rom_deployment_lstm
from Analyses import visualize_predictions, analyze_predictions

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Scaling
preproc_input = Pipeline([('minmaxscaler', MinMaxScaler())])

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the main file for POD-LSTM assessment
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    if train_mode:

        # Perform POD
        if perform_pod:
            # Snapshot collection
            sm_train, sm_valid, sm_test = load_snapshots_pod() # Note that columns of a snapshot/state are time always and a state vector is a column vector
            # Eigensolve
            generate_pod_bases(sm_train,sm_valid,sm_test,num_modes)
            phi, cf, cf_v, cf_t, _ = load_coefficients_pod()
        else:
            phi, cf, cf_v, cf_t, _ = load_coefficients_pod()

        # Need to scale the rows individually to ensure common scale - need to add validation here
        cf = np.transpose(preproc_input.fit_transform(np.transpose(cf)))
        cf_v = np.transpose(preproc_input.transform(np.transpose(cf_v)))

        # LSTM network
        model = lstm_for_dynamics(cf,cf_v,num_epochs,window_len,train_mode)

    else:

        # Load data for testing
        phi, cf, cf_v, cf_t, smean = load_coefficients_pod()

        cf = np.transpose(preproc_input.fit_transform(np.transpose(cf)))
        cf_t = np.transpose(preproc_input.transform(np.transpose(cf_t)))
        cf_v = np.transpose(preproc_input.transform(np.transpose(cf_v)))

        # LSTM network on training data
        model = lstm_for_dynamics(cf,cf_v,num_epochs,window_len,train_mode)

        tsteps = np.shape(cf)[1]
        _, lstm = evaluate_rom_deployment_lstm(model,cf,tsteps,num_modes,window_len)

        tsteps = np.shape(cf_v)[1]
        _, lstm_v = evaluate_rom_deployment_lstm(model,cf_v,tsteps,num_modes,window_len)

        tsteps = np.shape(cf_t)[1]
        _, lstm_t = evaluate_rom_deployment_lstm(model,cf_t,tsteps,num_modes,window_len)

        # Metrics
        print('MAE metrics on train data:',mean_absolute_error(cf[:,:-window_len],lstm[:,:-window_len]))
        print('MAE metrics on valid data:',mean_absolute_error(cf_v[:,:-window_len],lstm_v[:,:-window_len]))
        print('MAE metrics on test data:',mean_absolute_error(cf_t[:,:-window_len],lstm_t[:,:-window_len]))

        print('R2 metrics on train data:',r2_score(cf[:,:-window_len],lstm[:,:-window_len]))
        print('R2 metrics on valid data:',r2_score(cf_v[:,:-window_len],lstm_v[:,:-window_len]))
        print('R2 metrics on test data:',r2_score(cf_t[:,:-window_len],lstm_t[:,:-window_len]))

        # Rescale and save
        # Train
        cf = np.transpose(preproc_input.inverse_transform(np.transpose(cf)))
        lstm = np.transpose(preproc_input.inverse_transform(np.transpose(lstm)))
        np.save('./Coefficients/Prediction_train_'+str(geo_data)+'.npy',lstm)

        # Valid
        cf_v = np.transpose(preproc_input.inverse_transform(np.transpose(cf_v)))
        lstm_v = np.transpose(preproc_input.inverse_transform(np.transpose(lstm_v)))
        np.save('./Coefficients/Prediction_valid_'+str(geo_data)+'.npy',lstm_v)

        # Test
        cf_t = np.transpose(preproc_input.inverse_transform(np.transpose(cf_t)))
        lstm_t = np.transpose(preproc_input.inverse_transform(np.transpose(lstm_t)))
        np.save('./Coefficients/Prediction_test_'+str(geo_data)+'.npy',lstm_t)

        # # Visualize train
        # visualize_predictions(lstm,cf,smean,phi,'train')
        # # Visualize valid
        # visualize_predictions(lstm_v,cf_v,smean,phi,'valid')
        
        # Visualize and analyze test
        visualize_predictions(lstm_t,cf_t,smean,phi,'test')
        analyze_predictions(lstm_t,cf_t,smean,phi,'test')
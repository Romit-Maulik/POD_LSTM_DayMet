from Config import *
import numpy as np
import tensorflow as tf

# Set seeds
np.random.seed(10)
tf.random.set_seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D

from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import horovod.tensorflow.keras as hvd

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load DayMet dataset in CAE compatible form
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_snapshots_cae():
    
    if geo_data == 'tmax':
        fname = '../Data/Daymet_total_tmax.npy'

        data_total = np.load(fname)
        data_train = data_total[0*365:11*365] # 2000-2010
        data_valid = data_total[11*365:15*365] # 2011-2014
        data_test = data_total[15*365:] # 2016
    
    elif geo_data == 'prcp': # Some data missing in between
        fname = '../Data/Daymet_total_prcp.npy'

        data_total = np.load(fname)
        num_snapshots = np.shape(data_total)[0]
        num_train = int(num_snapshots*0.85)
        num_test = int(num_snapshots*0.15)

        data_train = data_total[0:num_train] 
        data_test = data_total[num_train:]

    num_train = np.shape(data_train)[0]
    num_test = np.shape(data_test)[0]

    dim_1 = np.shape(data_train)[1]
    dim_2 = np.shape(data_train)[2]

    data_train = data_train.reshape(num_train,dim_1*dim_2)
    data_test = data_test.reshape(num_test,dim_1*dim_2)

    preproc = MinMaxScaler()
    # preproc = Pipeline([('minmaxscaler', MinMaxScaler())]) #Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
    
    data_train = preproc.fit_transform(data_train)
    data_train = data_train.reshape(num_train,dim_1,dim_2,1) 

    data_test = preproc.transform(data_test)
    data_test = data_test.reshape(num_test,dim_1,dim_2,1)

    # Save CAE scaler
    from sklearn.externals import joblib
    scaler_filename = "cae_scaler.save"
    joblib.dump(preproc, scaler_filename) 

    # Pad zeros
    zero_pad_train = np.zeros(shape=(num_train,1024,1024,1))
    zero_pad_train[:,512-404:512+404,512-391:512+391,0] = data_train[:,:,:,0]

    zero_pad_test = np.zeros(shape=(num_test,1024,1024,1))
    zero_pad_test[:,512-404:512+404,512-391:512+391,0] = data_test[:,:,:,0]

    return zero_pad_train, zero_pad_test, preproc, dim_1, dim_2


def cae_model():
    ## Encoder
    encoder_inputs = Input(shape=(1024,1024,1),name='Field')
    # Encode   
    x = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    enc_l2 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')(enc_l2)
    enc_l3 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')(enc_l3)
    enc_l4 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(enc_l4)
    enc_l5 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')(enc_l5)
    enc_l6 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(5,kernel_size=(3,3),activation=None,padding='same')(enc_l6)
    enc_l7 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(3,kernel_size=(3,3),activation=None,padding='same')(enc_l7)
    enc_l8 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(enc_l8)
    encoded = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    encoder = Model(inputs=encoder_inputs,outputs=encoded)

    ## Decoder
    decoder_inputs = Input(shape=(4,4,1),name='decoded')

    x = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(decoder_inputs)
    dec_l1 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(3,kernel_size=(3,3),activation='relu',padding='same')(dec_l1)
    dec_l2 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(5,kernel_size=(3,3),activation='relu',padding='same')(dec_l2)
    dec_l3 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(10,kernel_size=(3,3),activation='relu',padding='same')(dec_l3)
    dec_l4 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(dec_l4)
    dec_l5 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')(dec_l5)
    dec_l6 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')(dec_l6)
    dec_l7 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')(dec_l7)
    dec_l8 = UpSampling2D(size=(2, 2))(x)

    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(dec_l8)
    decoder = Model(inputs=decoder_inputs,outputs=decoded)

    ## Autoencoder
    ae_outputs = decoder(encoder(encoder_inputs)) 
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')

    return model, encoder, decoder

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate CAE encoding
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_cae(zero_pad_train, zero_pad_test, preproc, dim_1, dim_2, train_mode, hvd_mode=False):
    # Shuffle
    idx_train = np.arange(np.shape(zero_pad_train)[0])
    np.random.shuffle(idx_train)
    zero_pad_train = zero_pad_train[idx_train]

    # Just keeping a few aside for validation - due to memory limitations
    zero_pad_valid = zero_pad_train[-5:]
    zero_pad_train = zero_pad_train[:-5]
    
    idx_test = np.arange(np.shape(zero_pad_test)[0])
    np.random.shuffle(idx_test)
    zero_pad_test = zero_pad_test[idx_test]

    # CNN training stuff
    weights_filepath = "../CAE_Training/cae_best_weights_"+str(geo_data)+".h5"
    lrate = 0.001

    # Get CAE model
    model,encoder,_ = cae_model()

    # design network
    my_adam = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    if hvd_mode:
        my_adam = hvd.DistributedOptimizer(my_adam)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)    
    callbacks_list = [earlystopping]

    if hvd_mode:
        callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
        ]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks_list = callbacks + callbacks_list
            checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
            callbacks_list.append(checkpoint)

        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0
        model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination],experimental_run_tf_function=False)    
    else:
        model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
        callbacks_list.append(checkpoint)

    model.summary()
    num_epochs = num_epochs_space

    # fit network
    if train_mode:

        if hvd_mode:
            train_history = model.fit(x=zero_pad_train, y=zero_pad_train, callbacks=callbacks_list, batch_size=batchsize_space,\
                                 validation_data=(zero_pad_valid,zero_pad_valid))

            if hvd.rank() == 0:
                model.load_weights(weights_filepath)

                idx_train = sorted(range(len(idx_train)), key=lambda k: idx_train[k])
                idx_test = sorted(range(len(idx_test)), key=lambda k: idx_test[k])

                # Rejoin train and valid
                zero_pad_train = np.concatenate((zero_pad_train,zero_pad_valid),axis=0)
                zero_pad_train = zero_pad_train[idx_train]
                zero_pad_test = zero_pad_test[idx_test]

                # Call to save latent space representation
                save_latent_space(model,encoder,zero_pad_train,zero_pad_test,preproc,dim_1, dim_2)            

        else:
            train_history = model.fit(x=zero_pad_train, y=zero_pad_train,epochs=num_epochs, callbacks=callbacks_list, batch_size=batchsize_space,\
                                 validation_data=(zero_pad_valid,zero_pad_valid))

            model.load_weights(weights_filepath)

            idx_train = sorted(range(len(idx_train)), key=lambda k: idx_train[k])
            idx_test = sorted(range(len(idx_test)), key=lambda k: idx_test[k])

            # Rejoin train and valid
            zero_pad_train = np.concatenate((zero_pad_train,zero_pad_valid),axis=0)
            zero_pad_train = zero_pad_train[idx_train]
            zero_pad_test = zero_pad_test[idx_test]

            # Call to save latent space representation
            save_latent_space(model,encoder,zero_pad_train,zero_pad_test,preproc,dim_1, dim_2)
            
    return model

def save_latent_space(model,encoder,zero_pad_train,zero_pad_test,preproc,dim_1, dim_2):
    
    for time in range(0,300,30):
        recoded = model.predict(zero_pad_test[time:time+1,:,:,:])
        true = preproc.inverse_transform(zero_pad_test[time:time+1,512-404:512+404,512-391:512+391,:].reshape(1,dim_1*dim_2)).reshape(dim_1,dim_2)
        recoded = preproc.inverse_transform(recoded[:,512-404:512+404,512-391:512+391,:].reshape(1,dim_1*dim_2)).reshape(dim_1,dim_2)

        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(6,6))
        cs1 = ax[0].imshow(true,label='input',vmin=-20,vmax=40)
        cs2 = ax[1].imshow(recoded,label='decoded',vmin=-20,vmax=40)

        for i in range(2):
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')

        fig.colorbar(cs1,ax=ax[0],fraction=0.046, pad=0.04)
        fig.colorbar(cs2,ax=ax[1],fraction=0.046, pad=0.04)
        ax[0].set_title(r'True $q_1$')
        ax[1].set_title(r'Reconstructed $q_1$')
        plt.subplots_adjust(wspace=0.5,hspace=-0.3)
        plt.tight_layout()
        plt.savefig('../CAE_Training/Reconstructions/Test_Recon_'+str(time)+'.png')


        # Encode the training data to generate time-series information
        encoded_list = []
        for i in range(np.shape(zero_pad_train)[0]):
            temp = K.eval(encoder(zero_pad_train[i:i+1].astype('float32')))
            encoded_list.append(temp.flatten())
        encoded_train = np.asarray(encoded_list)
        np.save("../Latent_Space/CAE_Coefficients_Train_"+str(geo_data)+".npy",encoded_train)

        encoded_list = []
        for i in range(np.shape(zero_pad_test)[0]):
            temp = K.eval(encoder(zero_pad_test[i:i+1].astype('float32')))
            encoded_list.append(temp.flatten())
        encoded_test = np.asarray(encoded_list)
        np.save("../Latent_Space/CAE_Coefficients_Test_"+str(geo_data)+".npy",encoded_test)


#-------------------------------------------------------------------------------------------------
# Load prepared coefficients
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_coefficients_cae():
    cf = np.load('../Latent_Space/CAE_Coefficients_Train_'+str(geo_data)+'.npy')
    cf_t = np.load('../Latent_Space/CAE_Coefficients_Test_'+str(geo_data)+'.npy')

    # Lowess filtering
    arr_len = np.shape(cf)[0]
    for i in range(np.shape(cf)[1]):
        cf[:,i] = lowess(cf[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)

    arr_len = np.shape(cf_t)[0]
    for i in range(np.shape(cf_t)[1]):
        cf_t[:,i] = lowess(cf_t[:,i], np.arange(arr_len), frac=0.3, return_sorted=False)


    return cf, cf_t

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Metrics
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def new_r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=0)
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=0)), axis=0)
    output_scores =  1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.keras.backend.mean(output_scores)
    return r2

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

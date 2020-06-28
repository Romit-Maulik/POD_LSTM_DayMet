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

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Load DayMet dataset in CAE compatible form
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def load_snapshots_cae():
    
    if geo_data == 'tmax':
        fname = '../Data/Daymet_total_tmax.npy'

        data_total = np.load(fname)
        data_train = data_total[0*365:11*365:4] # 2000-2010
        data_valid = data_total[11*365:15*365:4] # 2011-2014
        data_test = data_total[15*365::4] # 2016
    
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

    preproc = Pipeline([('minmaxscaler', MinMaxScaler())]) #Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
    
    data_train = preproc.fit_transform(data_train)
    data_train = data_train.reshape(num_train,dim_1,dim_2,1) 

    data_test = preproc.transform(data_test)
    data_test = data_test.reshape(num_test,dim_1,dim_2,1)

    # Pad zeros
    zero_pad_train = np.zeros(shape=(num_train,896,896,1))
    zero_pad_train[:,448-404:448+404,448-391:448+391,0] = data_train[:,:,:,0]

    zero_pad_test = np.zeros(shape=(num_test,896,896,1))
    zero_pad_test[:,448-404:448+404,448-391:448+391,0] = data_test[:,:,:,0]

    return zero_pad_train, zero_pad_test, preproc, dim_1, dim_2


#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# Generate CAE encoding
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def generate_cae(zero_pad_train, zero_pad_test, preproc, dim_1, dim_2, train_mode):
    # Shuffle
    idx_train = np.arange(np.shape(zero_pad_train)[0])
    np.random.shuffle(idx_train)
    zero_pad_train = zero_pad_train[idx_train]

    # Just keeping a few aside for validation - due to memory limitations
    zero_pad_valid = np.copy(zero_pad_train[-10:])
    zero_pad_train = np.copy(zero_pad_train[:-10])
    
    idx_test = np.arange(np.shape(zero_pad_test)[0])
    np.random.shuffle(idx_test)
    zero_pad_test = zero_pad_test[idx_test]
    #zero_pad_train_shuffled = np.copy(zero_pad_train)
    #zero_pad_test_shuffled = np.copy(zero_pad_test)
    #np.random.shuffle(zero_pad_train_shuffled)
    #np.random.shuffle(zero_pad_test_shuffled)

    #train_valid_dataset = tf.data.Dataset.from_tensor_slices((zero_pad_train, zero_pad_train))
    #train_dataset = train_valid_dataset.take(len(idx_train))
    #valid_dataset = train_valid_dataset.skip(len(idx_train))
   
    # CNN training stuff
    weights_filepath = "../CAE_Training/cae_best_weights_"+str(geo_data)+".h5"
    lrate = 0.001

    ## Encoder
    encoder_inputs = Input(shape=(896,896,1),name='Field')
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

    x = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(enc_l7)
    encoded = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    encoder = Model(inputs=encoder_inputs,outputs=encoded)

    ## Decoder
    decoder_inputs = Input(shape=(7,7,1),name='decoded')

    x = Conv2D(2,kernel_size=(3,3),activation=None,padding='same')(decoder_inputs)
    dec_l1 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(5,kernel_size=(3,3),activation='relu',padding='same')(dec_l1)
    dec_l2 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(10,kernel_size=(3,3),activation='relu',padding='same')(dec_l2)
    dec_l3 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(dec_l3)
    dec_l4 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')(dec_l4)
    dec_l5 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')(dec_l5)
    dec_l6 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')(dec_l6)
    dec_l7 = UpSampling2D(size=(2, 2))(x)

    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(dec_l7)
    decoder = Model(inputs=decoder_inputs,outputs=decoded)

    
    ## Autoencoder
    ae_outputs = decoder(encoder(encoder_inputs)) 
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='CAE')

    # design network
    my_adam = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]

    model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])    
    model.summary()

    num_epochs = num_epochs_space
    

    # fit network
    if train_mode:
        train_history = model.fit(x=zero_pad_train, y=zero_pad_train,epochs=num_epochs, callbacks=callbacks_list, batch_size=batchsize_space,\
                                 validation_data=(zero_pad_valid,zero_pad_valid))

        model.load_weights(weights_filepath)

        idx_train = sorted(range(len(idx_train)), key=lambda k: idx_train[k])
        idx_test = sorted(range(len(idx_test)), key=lambda k: idx_test[k])

        # Rejoin train and valid
        zero_pad_train = np.concatenate((zero_pad_train,zero_pad_valid),axis=0)

        zero_pad_train = zero_pad_train[idx_train]
        zero_pad_test = zero_pad_test[idx_test]
        
        for time in range(0,10):
            recoded = model.predict(zero_pad_test[time:time+1,:,:,:])
            true = preproc.inverse_transform(zero_pad_test[time:time+1,448-404:448+404,448-391:448+391,:].reshape(1,dim_1*dim_2)).reshape(dim_1,dim_2)
            recoded = preproc.inverse_transform(recoded[:,448-404:448+404,448-391:448+391,:].reshape(1,dim_1*dim_2)).reshape(dim_1,dim_2)

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

    model.load_weights(weights_filepath)

    return model

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

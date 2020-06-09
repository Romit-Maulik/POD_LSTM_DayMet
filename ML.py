from Config import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Add, LSTM, dot, concatenate, Activation, Dropout, Bidirectional
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# LSTM architecture
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def lstm_for_dynamics(cf_trunc,cf_trunc_v,num_epochs,seq_num,train_mode):
    features = np.transpose(cf_trunc)
    features_v = np.transpose(cf_trunc_v) # Validation
    states = np.copy(features[:,:]) #Rows are time, Columns are state values
    states_v = np.copy(features_v[:,:]) #Rows are time, Columns are state values

    # Need to make batches of 10 input sequences and 1 output
    # Training
    total_size = np.shape(features)[0]-2*seq_num + 1
    input_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1]))
    output_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1]))

    for t in range(total_size):
        input_seq[t,:,:] = states[None,t:t+seq_num,:]
        output_seq[t,:,:] = states[None,t+seq_num:t+2*seq_num,:]

    # Validation
    total_size = np.shape(features_v)[0]-2*seq_num + 1
    input_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))
    output_seq_v = np.zeros(shape=(total_size,seq_num,np.shape(states_v)[1]))

    for t in range(total_size):
        input_seq_v[t,:,:] = states_v[None,t:t+seq_num,:]
        output_seq_v[t,:,:] = states_v[None,t+seq_num:t+2*seq_num,:]

    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    input_seq = input_seq[idx,:,:]
    output_seq = output_seq[idx,:,:]
   
    lstm_inputs = Input(shape=(seq_num, np.shape(states)[1],))
    l1 = Bidirectional(LSTM(50,return_sequences=True))(lstm_inputs)
    l1 = Dropout(0.1)(l1,training=False)
    l1 = Bidirectional(LSTM(50,return_sequences=True))(l1)
    l1 = Dropout(0.1)(l1,training=False)
    l1 = Bidirectional(LSTM(50,return_sequences=True))(l1)
    l1 = Dropout(0.1)(l1,training=False)
    l1 = Bidirectional(LSTM(50,return_sequences=True))(l1)
    l1 = Dropout(0.1)(l1,training=False)
    op = Dense(np.shape(states)[1], activation='linear', name='output')(l1)
    model = Model(inputs=lstm_inputs, outputs=op)

    # design network
    my_adam = optimizers.Adam(lr=0.0005, decay=0.0)

    filepath = "./Training/best_weights_lstm.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
    callbacks_list = [checkpoint,earlystopping]
    
    # fit network
    model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])
    model.summary()

    if train_mode:
        train_history = model.fit(input_seq, \
                                output_seq, \
                                epochs=num_epochs, \
                                batch_size=batchsize, \
                                callbacks=callbacks_list, \
                                validation_data=(input_seq_v, output_seq_v))
        
        np.save('./Training/Train_Loss.npy',train_history.history['loss'])
        np.save('./Training/Val_Loss.npy',train_history.history['val_loss'])

    model.load_weights(filepath)

    return model

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# LSTM forecast
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def evaluate_rom_deployment_lstm(model,dataset,tsteps,num_modes,seq_num):

    # Make the initial condition from the first seq_num columns of the dataset
    features = np.transpose(dataset)  
    input_states = np.copy(features)

    state_tracker = np.zeros(shape=(1,tsteps,np.shape(features)[1]),dtype='double')
    state_tracker[0,0:seq_num,:] = input_states[0:seq_num,:]

    total_size = np.shape(features)[0]-seq_num + 1

    for t in range(seq_num,total_size,seq_num):
        lstm_input = np.expand_dims(input_states[t-seq_num:t,:],0)
        output_state = model.predict(lstm_input)
        state_tracker[0,t:t+seq_num,:] = output_state[0,:,:]

    return np.transpose(output_state), np.transpose(state_tracker[0,:,:])

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
    
'''In this file we run Seq2seq autoencoder with 2 variants of loss functions 
(mse & weighted mse) and save the models to files'''


seed_value= 0
import numpy as np
import os, random, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

import h5py


def custom_loss_wei():
    '''returns mse*log of total haversine distance covered by ground truth 
        trajectory to be used as loss function.
        The weight is used to penalize long trajectories'''
    
    def loss_wei_hav(y_true, y_pred):
                  
        lat_std = 0.01411078948810304
        lat_avg = 41.158590518999034
        lng_std =  0.024607830653497958
        lng_avg = -8.618621150409412
        
        lat1_tr = (y_true[:,:8, 1]*lat_std)+lat_avg
        lat2_tr = (y_true[:,1:9, 1]*lat_std)+lat_avg
        lon1_tr = (y_true[:,:8, 0]*lng_std)+lng_avg
        lon2_tr = (y_true[:,1:9, 0]*lng_std)+lng_avg
        
        lat1_tr = lat1_tr*np.pi / 180
        lat2_tr = lat2_tr*np.pi / 180       
        lon1_tr = lon1_tr*np.pi / 180
        lon2_tr = lon2_tr*np.pi / 180
        
        dlat_tr = lat2_tr - lat1_tr
        dlon_tr = lon2_tr - lon1_tr
        
        REarth = 6371
        
        a = (tf.sin(dlat_tr/2)) **2 + (tf.cos(lat1_tr) * tf.cos(lat2_tr) * (tf.sin(dlon_tr/2)) **2)
        dist = 2*REarth*tf.math.asin(tf.sqrt(a))
        
        weight = K.log(K.sum(dist, axis=-1)+1)
        
        mseloss = tf.keras.losses.MeanSquaredError()
                
        return mseloss(y_true,y_pred, sample_weight=weight)
        
    return loss_wei_hav


def get_dataset(folders):
    '''returns tensors of train& dev set of trajectory data [instances, timesteps, 2] &
    the corresponding tensors with added noise'''
        
    folder = folders['data_folder']
    X_train = pickle.load(open(folder+'train_std_lstm_outlier_detection.p', 'rb')) 
    X_val = pickle.load(open(folder+'dev_std_lstm_outlier_detection.p', 'rb')) 
    
    #load X_train_noise, X_val_noise files for reproduction of results
    X_train_noise = pickle.load(open(folder+'train_noise_std_lstm.p', 'rb')) 
    X_val_noise = pickle.load(open(folder+'dev_noise_std_lstm.p', 'rb')) 
    
    #alternatively  add noise to training, validation, test data and save the files
    '''noise_pct = 0.05
    X_train_noise = X_train + (noise_pct*np.random.normal(loc=0.0, scale=1.0, size=X_train.shape))
    pickle.dump(X_train_noise, open(folder+'train_noise_std_lstm.p', 'wb'))
    X_val_noise = X_val + (noise_pct*np.random.normal(loc=0.0, scale=1.0, size=X_val.shape))
    pickle.dump(X_val_noise, open(folder+'dev_noise_std_lstm.p', 'wb'))
    X_test = pickle.load(open(folder+'test_std_lstm_outlier_detection.p', 'rb'))
    X_test_noise = X_test + (noise_pct*np.random.normal(loc=0.0, scale=1.0, size=testX.shape))
    pickle.dump(X_test_noise, open(folder+'test_noise_std_lstm.p', 'wb'))
    '''
    
    return X_train, X_train_noise, X_val, X_val_noise


def get_dataset_shifted(X):
    '''returns a tensor with 1 row of zeros added to X, to be used as input
    for the decoder'''
    
    zero_pad = np.zeros((1,2))
    X_2 = np.empty([X.shape[0], X.shape[1],X.shape[2]])
    for i in range(len(X)):
        X_2[i] = np.vstack([zero_pad, X[i,:-1,:]])
        
    return X_2


def define_models(n_units, n_input, n_output):
    '''returns train model'''

	# define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True, kernel_regularizer=l2(0.1))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder using encoder_states as initial state
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, kernel_regularizer=l2(0.1))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(n_output))
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model


def train_LSTM_outlier_seq(folders, n_units, wei):
    '''trains & save to h5 file the seq2seq models
    n_units:number of neurons in the LSTM layers'''
        
    X_train, X_train_noise, X_val, X_val_noise = get_dataset(folders)
    X_2_train = get_dataset_shifted(X_train)
    X_2_val = get_dataset_shifted(X_val)

    #n_input,n_output are number of features
    model = define_models(n_units, n_input=2, n_output=2) 
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999)
    
    if wei==True:    
        model.compile(loss=custom_loss_wei(), metrics=['mse'], optimizer = adam)
        filepath = folders['model_folder']+'best_seq.h5' 
                
    else:
        model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = adam)
        filepath = folders['model_folder']+'best_seq_mse.h5'
        
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)
    
    model.fit([X_train_noise,X_2_train], X_train,\
              batch_size=512,epochs=200,shuffle=True,verbose=2,\
              validation_data=([X_val_noise,X_2_val], X_val),\
              callbacks=[early_stopping_monitor,checkpoint])


def predict_sequence(enc, dec, source, n_steps, num_feat):
    '''generates target given source sequence
    enc, dec: the encoder, decoder models
    source: the input sequence [timesteps,2]
    n_steps: num of timesteps, num_feat=2 (lng, lat)'''
    
    source = np.reshape(source,(1,source.shape[0],2))
	# encode the input sequence
    state = enc.predict(source)
	# start of sequence input
    target_seq = np.array([0.0 for _ in range(num_feat)]).reshape(1, 1, num_feat)
	# generate predictions step by step
    output = list()
    for t in range(n_steps):
		# predict next char
        yhat, h, c = dec.predict([target_seq] + state)
		# store prediction
        output.append(yhat[0,0,:])
		# update state
        state = [h, c]
		# update target sequence
        target_seq = yhat
    
    return(np.array(output))  
    

def train_seq2seq(folders, wei, train_model, n_units):
    '''trains seq2seq models, which are saved to h5 files & make predictions
    for our real test set'''
    
    if train_model==True:
        train_LSTM_outlier_seq(n_units, wei)
        
    testX_noise = pickle.load(open(folders['data_folder']+'test_noise_std_lstm.p', 'rb'))
    
    if wei==True:            
        filepath = folders['model_folder']+'best_seq.h5'             
        model = load_model(filepath,custom_objects = {'loss_wei_hav':custom_loss_wei()})
       
    else:        
        filepath = folders['model_folder']+'best_seq_mse.h5'                
        model = load_model(filepath)
    
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(n_units,), name='input_3')
    decoder_state_input_c = Input(shape=(n_units,), name='input_4')
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    #inference model
    preds = []
    for i in range(testX_noise.shape[0]):
        pred_traj_std = predict_sequence(encoder_model, decoder_model, testX_noise[i], \
                         n_steps=testX_noise.shape[1], num_feat=2) #get preds for one traj
        preds.append(pred_traj_std)
    preds = np.array(preds)
    
    if wei==True:
        pickle.dump(preds, open(folders['preds_folder']+'pred_traj_seq.p', 'wb'))
    else:
        pickle.dump(preds, open(folders['preds_folder']+'pred_traj_seq_mse.p', 'wb'))
    
        

    
    



    
    

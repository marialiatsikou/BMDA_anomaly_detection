'''In this file we train LSTM autoencoders (with both weighted & unweighted loss
function) and feedforward autoencoder and save the models to files'''


seed_value= 0
import numpy as np
import tensorflow as tf
import os, random, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from keras.regularizers import l2
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


def get_data(folders):
    '''returns tensors of train & dev set of trajectory data 
    [instances, timesteps, 2] & the corresponding tensors with added noise'''
    
    folder = folders['data_folder']
    
    X_train = pickle.load(open(folder+'train_std_lstm_outlier_detection.p', 'rb')) 
    X_train_noise = pickle.load(open(folder+'train_noise_std_lstm.p', 'rb')) 
    X_val = pickle.load(open(folder+'dev_std_lstm_outlier_detection.p', 'rb'))  
    X_val_noise = pickle.load(open(folder+'dev_noise_std_lstm.p','rb'))  

    return X_train, X_train_noise, X_val, X_val_noise

 
def get_data_ff(folders):
    '''returns arrays of train& dev set of trajectory data 
    [instances, 2*timesteps] & the corresponding arrays with added noise'''
        
    folder = folders['data_folder']
    X_train = pickle.load(open(folder+'train_std_ff_outlier_detection.p', 'rb')) 
    
    X_train_noi_lstm = pickle.load(open(folder+'train_noise_std_lstm.p', 'rb'))
    
    X_train_noi = np.empty((X_train.shape[0], X_train.shape[1]))
    X_train_noi[:,:X_train_noi_lstm.shape[1]] = X_train_noi_lstm[:,:,0]
    X_train_noi[:,X_train_noi_lstm.shape[1]:] = X_train_noi_lstm[:,:,1]
      
    X_val = pickle.load(open(folder+'dev_std_ff_outlier_detection.p', 'rb'))  
    
    X_val_noi_lstm = pickle.load(open(folder+'dev_noise_std_lstm.p', 'rb'))
    
    X_val_noi = np.empty((X_val.shape[0], X_val.shape[1]))
    X_val_noi[:,:X_val_noi_lstm.shape[1]] = X_val_noi_lstm[:,:,0]
    X_val_noi[:,X_val_noi_lstm.shape[1]:] = X_val_noi_lstm[:,:,1]

    return X_train, X_train_noi, X_val, X_val_noi

       
def train_vae(folders, wei):
    '''trains an autoencoder of 1 lstm encoder & 1 lstm decoder'''

    X_train, X_train_noise, X_val, X_val_noise = get_data(folders)
    
    model = Sequential()
    model.add(LSTM(16, input_shape=(X_train.shape[1], X_train.shape[2]),\
                     kernel_regularizer=l2(0.1),return_sequences=False))       
    
    model.add((RepeatVector(X_train.shape[1])))
    
    model.add(LSTM(16, return_sequences=True, kernel_regularizer=l2(0.1)))
      
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999)
    
    if wei==True:    
        model.compile(loss=custom_loss_wei(), metrics=['mse'], optimizer = adam)
        filepath = folders['model_folder']+'best_vae.h5' 
    else:
        model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = adam)
        filepath = folders['model_folder']+'best_vae_mse.h5'
        
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                 save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)
    
    model.fit(X_train_noise, X_train,\
              batch_size=512,epochs=300,shuffle=True,verbose=2,\
              validation_data=(X_val_noise, X_val), 
              callbacks=[early_stopping_monitor,checkpoint])
    
   
def train_ffae(folders):
    '''trains a feedforward autoencoder of 1 encoder & 1 decoder'''

    X_train, X_train_noise, X_val, X_val_noise = get_data_ff(folders)
    
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], kernel_regularizer=l2(0.1),
                    activation='tanh'))       
        
    model.add(Dense(16, kernel_regularizer=l2(0.1), activation='tanh')) 
    model.add(Dense(18)) 
    
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999)
    
   
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = adam)
    filepath = folders['model_folder']+'best_ffae_mse.h5'
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                 save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)
    
    model.fit(X_train_noise, X_train,\
              batch_size=512,epochs=300,shuffle=True,verbose=2,\
              validation_data=(X_val_noise, X_val), 
              callbacks=[early_stopping_monitor,checkpoint])
    
    
def run_lstm_ff(folders):  
    '''train LSTM & feedforward autoencoders'''
    
    weights = [True, False]
    for wei in weights:
        train_vae(folders, wei)
    
    train_ffae(folders)




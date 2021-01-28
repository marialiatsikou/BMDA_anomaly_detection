'''In this file we create train, dev, test set for lstm & feedforward autoencoders
'''

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from geopy import distance


def get_synthetic_experiment_folders():
    '''returns a dict of file folders'''
    
    folders = dict()
    
    folders['data_folder'] = 'data/' #where the data are stored 
    folders['model_folder'] = 'models/' #where the best models are stored
    folders['errors_folder'] = 'errors/' #where the errors are stored
    folders['images_folder'] = 'images/' #where the images stored
    folders['preds_folder'] = 'predictions/' #where the predictions are stored
    
    return folders


def create_id_dict(polyline_df):
    '''returns arrays of longs, lats, ts, ids from the polyline_df
    and a dict 'ranges' trip_id: [start_end] with the indices of start/end for each trip'''
       
    longs =  np.array(polyline_df['lng'].values)
    lats = np.array(polyline_df['lat'].values)
    ts = np.array(polyline_df['ts'].values)
    ids = polyline_df['trip_id'].values    
    
    #first, create a dictionary: {trip_id: [start_idx, end_idx]} (for faster processing)
    ranges = dict() #{trip_id: [start index, end index]
    prev_trip = ids[0]
    start_idx = 0
    for i in range(1, len(ids)):
        this_trip = ids[i]
        if this_trip!=prev_trip:
            ranges[prev_trip] = [start_idx, i]
            prev_trip = ids[i]
            start_idx = i
    ranges[prev_trip] = [start_idx, len(ids)]
    
    return longs, lats, ids, ts, ranges
    
    
def create_traj(polyline_df):
    '''creates lists of trip_ids, start/end ts for each trip, lists of lons, lats, ts
    of all points for each trip. All lists have len= number of trip ids'''
        
    trip_ids, start_ts, all_ts, end_ts, lng_list, lat_list = [],[], [], [], [], []
    longs, lats, ids, ts, ranges = create_id_dict(polyline_df)
    
    for trip in set(ids):
        idx = ranges[trip]
        lo, la, t = longs[idx[0]:idx[1]], lats[idx[0]:idx[1]], ts[idx[0]:idx[1]]
        sorted_stuff = np.array(sorted(zip(t,lo,la))) #sort based on time
        lo, la, t = sorted_stuff[:,1], sorted_stuff[:,2], sorted_stuff[:,0] #unzip
        
        trip_ids.append(trip)
        start_ts.append(t[0]) #start ts of all trips
        end_ts.append(t[-1])
        lng_list.append([k for k in lo])
        lat_list.append([k for k in la])
        all_ts.append([ts for ts in t])
            
    return trip_ids, start_ts, end_ts, lng_list, lat_list, all_ts
            

def find_traj_dist(lng_li, lat_li):
    '''returns dist between first, middle and end point of one trajectory
    lng_li, lat_li are lists of coords for 1 traj'''
    
    middle = int(len(lat_li)/2)
    coord_start = [lat_li[0],lng_li[0]]
    coord_end = [lat_li[-1],lng_li[-1]]
    coord_middle = [lat_li[middle],lng_li[middle]]
    dist = distance.distance(coord_start, coord_middle).m + \
                    distance.distance(coord_middle ,coord_end).m
    return dist
                                   

def remove_small_dist(trip_id, start_t, end_t, lng_li, lat_li, all_t):    
    '''returns 6 arrays of final trip_ids, start_ts, end_ts, lng, lat of
    trajectories & all ts after computing distances between first, middle and 
    end point of each traj & removing those with len<500'''
    
    trip_ids, start_ts, all_ts, end_ts, lng_list, lat_list, all_dist = [],[],[],[],[],[],[]
    dist_li = []
    for i in range(len(lng_li)):
        dist = find_traj_dist(lng_li[i], lat_li[i])
        all_dist.append(dist)
                
        if dist>500:
            trip_ids.append(trip_id[i])
            start_ts.append(start_t[i])
            all_ts.append(all_t[i])
            end_ts.append(end_t[i])
            lng_list.append(lng_li[i])
            lat_list.append(lat_li[i])
            dist_li.append(dist)
            
    return trip_ids, start_ts, end_ts, lng_list, lat_list, all_ts

    
def create_final_traj(trip_ids, start_ts, end_ts, lng_list, lat_list, ts_list, pct):
     '''create 5 arrays of final trip_ids, start_ts, lng, lat, ts of trajectories
     whose length is the pct-th percentile of all traj lengths
     pct: the th percentile defining the length of trajectories'''
    
     #create lists of traj elements excluding some trips based on their duration
     lng, lat, ts, start_time, trip = [],[],[],[],[]
     trip_dur = np.array(end_ts)-np.array(start_ts)
     
     for i in range(len(lng_list)):
         
         if (trip_dur[i]>=180) & (trip_dur[i]<=7200): #reject trips with duration<3' & >120'
             lng.append(lng_list[i])
             lat.append(lat_list[i])
             ts.append(ts_list[i])
             start_time.append(start_ts[i])
             trip.append(trip_ids[i])
     
     '''create new lists for traj with specific length & corresp ids, start, end time
     len_list is list of traject lengths'''
     new_lng, new_lat, new_ts, new_start_ts, new_trip_ids,\
         len_list = [],[],[],[],[],[]
     for i in range(len(lng)):
         len_list.append(len(lng[i]))  #list of trajectories length
         if len(lng[i])!=len(lat[i]):  #assert all lat, lng have same length
             print(i)
             
     #choose thresh s.t. (1-pct)% of trajectories have len>thresh     
     thresh = np.percentile(len_list,pct)  
     dist_list = []
     for i in range(len(lng)):
         #take the traj with len>= thresh,  take the first thresh points of them & exclude dist<200m
         if (len(lng[i])>=thresh): 
             lng_1 = lng[i][0:int(thresh)]
             lat_1 = lat[i][0:int(thresh)]
             dist = find_traj_dist(lng_1, lat_1)
             if dist>200:
                 new_lng.append(lng_1)
                 new_lat.append(lat_1)
                 new_ts.append(ts[i][0:int(thresh)])
                 new_start_ts.append(start_time[i])
                 new_trip_ids.append(trip[i])                 
                 dist_list.append(dist)
                      
     new_lng = np.array(new_lng)
     new_lat = np.array(new_lat)
     new_start_ts = np.array(new_start_ts)
     new_trip_ids = np.array(new_trip_ids)
     new_ts = np.array(new_ts)
     
     return new_trip_ids, new_start_ts, new_ts, new_lng, new_lat
       

def traj_preprocess(polyline_df, pct):
    '''creates 5 arrays of trip_ids, start_ts, lng, lat, ts of trajectories
     whose length is the pct-th percentile of all traj lengths
    '''
    
    #creates lists of trip_ids, start/end ts for each trip, lists of lons, lats
    trip_ids, start_ts, end_ts, lng_list, lat_list, ts_list = create_traj(polyline_df) 
    
    #create list of distances using 3 points of each traj & keep only traj with len>500
    trip_ids, start_ts, end_ts, lng_list, lat_list, ts_list = \
        remove_small_dist(trip_ids, start_ts, end_ts, lng_list, lat_list, ts_list)
    
    #create 6 arrays of final trip_ids, start_ts, end_ts, ts, lng, lat of trajectories
    trip_ids, start_ts, ts, lng, lat = create_final_traj(trip_ids, start_ts,\
                             end_ts, lng_list, lat_list, ts_list, pct)
    
    return trip_ids, start_ts, ts, lng, lat
    
    
def train_dev_test_traj (folder, tr_lng, tr_lat, tr_ts, trip_ids, start_ts):
    '''returns 3 lists of arrays of train, dev, test for lng, lat.
    Split is done so that train, dev, test sets contain data of all months'''
        
    lng_train, lng_dev, lng_test, lat_train, lat_dev, lat_test, time_train,\
        time_dev, time_test,ts_train,ts_dev, ts_test, id_train, id_dev, \
        id_test, ts_month = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    
    #create a list of months'ids with len=num of trips
    for t in start_ts:
        ts_month.append(datetime.fromtimestamp(t).month)         
    ts_month=np.array(ts_month)
    mon_num = [7,8,9,10,11,12,1,2,3,4,5,6]
    for month in mon_num:
        idx = np.where(ts_month==month)[0] #index of the features of this month
        lng_per_mon = tr_lng[np.where(ts_month==month)] #lng of this month        
        lat_per_mon = tr_lat[np.where(ts_month==month)]
        ts_per_mon = tr_ts[np.where(ts_month==month)]
        start_ts_per_mon = start_ts[np.where(ts_month==month)]
        ids_per_month = trip_ids[np.where(ts_month==month)]
        upper_bound = len(idx)
        train_lim, dev_lim = int(0.8*upper_bound), int(0.9*upper_bound)
        lngtrain, lngdev, lngtest = lng_per_mon[0:train_lim], \
                    lng_per_mon[train_lim:dev_lim], lng_per_mon[dev_lim:]
        lattrain, latdev, lattest = lat_per_mon[0:train_lim], \
                    lat_per_mon[train_lim:dev_lim], lat_per_mon[dev_lim:]
        tstrain, tsdev, tstest = ts_per_mon[0:train_lim], ts_per_mon[train_lim:dev_lim],\
                    ts_per_mon[dev_lim:]          
        timetrain, timedev, timetest = start_ts_per_mon[0:train_lim], \
                    start_ts_per_mon[train_lim:dev_lim], start_ts_per_mon[dev_lim:]
        idtrain, iddev, idtest = ids_per_month[0:train_lim], \
                    ids_per_month[train_lim:dev_lim], ids_per_month[dev_lim:]
        for i in range(len(lngtrain)):
            lng_train.append(lngtrain[i])
            lat_train.append(lattrain[i])
            time_train.append(timetrain[i])
            ts_train.append(tstrain[i])
            id_train.append(idtrain[i])
        for i in range(len(lngdev)):
            lng_dev.append(lngdev[i])
            lat_dev.append(latdev[i]) 
            time_dev.append(timedev[i])
            ts_dev.append(tsdev[i])
            id_dev.append(iddev[i])
        for i in range(len(lngtest)):
            lng_test.append(lngtest[i])
            lat_test.append(lattest[i]) 
            time_test.append(timetest[i])
            ts_test.append(tstest[i])
            id_test.append(idtest[i])
        
    train = [np.array(lng_train), np.array(lat_train)]
    dev = [np.array(lng_dev), np.array(lat_dev)]  
    test = [np.array(lng_test),np.array(lat_test), np.array(time_test),\
            np.array(ts_test), np.array(id_test) ]
    
    #save trip start_ts, ts of all points and trip id for test set in pickles
    pickle.dump([np.array(time_test),np.array(ts_test)], \
                 open(folder+'test_timelist_outlier_detection.p', 'wb'))
    pickle.dump(np.array(id_test), open(folder+'test_id_outlier_detection.p', 'wb'))
    
    return train, dev ,test

    
def standarize_feature(folder, x_tr, x_dev, x_test, var):
    '''returns 3 2d arrays (num of instances, timesteps) for training, dev, test
    set of longitudes or latitudes depending on var'''
    
    xtr_mean = np.mean(x_tr)
    xtr_std = np.std(x_tr)
    if var=='lat' :
        pickle.dump([xtr_mean,xtr_std], open(folder+'train_lat_avg_std.p', 'wb'))
    if var=='lng' :
        pickle.dump([xtr_mean,xtr_std], open(folder+'train_lng_avg_std.p', 'wb'))
        
    x_tr_std = np.empty((x_tr.shape[0], x_tr.shape[1]))
    for i in range(x_tr.shape[0]): #for each instance     
        for j in range(x_tr.shape[1]):   #for each timestep         
            x_tr_std[i,j] = (x_tr[i,j]-xtr_mean)/xtr_std
    x_dev_std = np.empty((x_dev.shape[0], x_dev.shape[1]))
    for i in range(x_dev_std.shape[0]):
        for j in range(x_dev_std.shape[1]):            
            x_dev_std[i,j] = (x_dev[i,j]-xtr_mean)/xtr_std
    x_test_std = np.empty((x_test.shape[0], x_test.shape[1]))
    for i in range(x_test_std.shape[0]):
        for j in range(x_test_std.shape[1]):            
            x_test_std[i,j] = (x_test[i,j]-xtr_mean)/xtr_std
    
    return x_tr_std, x_dev_std, x_test_std


def get_coord_lists(folder, polyline_df, pct):
    '''returns 2 lists of 3 2d arrays for train/dev/test set of long, lat
    '''
        
    #create final traj with specific length (pct defines this)
    trip_ids, start_ts, ts, lng, lat = traj_preprocess(polyline_df, pct)  
    
    #get lists [lng, lat] of train/dev/test sets 
    train, dev ,test = train_dev_test_traj(folder, lng, lat, ts, trip_ids, start_ts)  
        
    lng_train, lng_dev, lng_test = train[0], dev[0], test[0]
    lat_train, lat_dev, lat_test = train[1], dev[1], test[1]
    
    standarize = [True, False]
    for std in standarize:
        if std == True:
            #standarize lng, lat arrays s.t. the arrays have zero mean & unit variance
            lng_train_std, lng_dev_std, lng_test_std = standarize_feature(folder, 
                                      lng_train, lng_dev, lng_test, var='lng')
            lat_train_std, lat_dev_std, lat_test_std = standarize_feature(folder,
                                      lat_train, lat_dev,lat_test, var='lat')
            lng_std = [lng_train_std, lng_dev_std, lng_test_std]
            lat_std = [lat_train_std, lat_dev_std, lat_test_std]
        else:
            lng = [lng_train, lng_dev, lng_test]
            lat = [lat_train, lat_dev, lat_test]
    
    return lng_std, lat_std, lng, lat
    

def create_traj_data_lstm(data_list):
    '''concatenates arrays to make a tensor of lng, lat
    data_list: list of arrays of lng, lat
    returns 3d tensor '''
    
    lng, lat = [data_list[i] for i in range(len(data_list))] 
    lng = lng.reshape(lng.shape[0], lng.shape[1], 1)
    lat = lat.reshape(lat.shape[0], lat.shape[1], 1)
    all_x = np.concatenate([lng,lat], axis=2)
    
    return all_x


def create_traj_lstm_dataset(folder, lng, lat, standarize): 
    '''creates train, dev, test set for lstm autoencoders 
    saves to pickles 3 lists [lng, lat] for train/dev/test sets
    '''
         
    lng_train, lng_dev, lng_test = lng[0], lng[1], lng[2]
    lat_train, lat_dev, lat_test = lat[0], lat[1], lat[2]
    x_train = [lng_train, lat_train]
    train_lstm = create_traj_data_lstm(x_train)
    x_dev = [lng_dev, lat_dev]
    dev_lstm = create_traj_data_lstm(x_dev)
    x_test = [lng_test, lat_test]  
    test_lstm = create_traj_data_lstm(x_test)
    
    if standarize == True:
        pickle.dump(train_lstm, open(folder+'train_std_lstm_outlier_detection.p', 'wb'))
        pickle.dump(dev_lstm, open(folder+'dev_std_lstm_outlier_detection.p', 'wb'))
        pickle.dump(test_lstm, open(folder+'test_std_lstm_outlier_detection.p', 'wb'))
    else:
        pickle.dump(test_lstm, open(folder+'test_lstm_outlier_detection.p', 'wb'))
    
    
def create_traj_data_ffnn(data_list):
    '''concatenates arrays to make a tensor of lng, lat
    data_list: list of arrays of lng, lat
    returns 2d array '''
    
    lng, lat = [data_list[i] for i in range(len(data_list))] 
    all_x = np.concatenate([lng,lat], axis=1)
    
    return all_x


def create_traj_ff_dataset(folder, lng, lat): 
    '''creates train, dev, test set for feedforward autoencoder 
    saves to pickles 3 arrays [instances, 2*timesteps] for train/dev/test sets'''
        
    lng_train, lng_dev, lng_test = lng[0], lng[1], lng[2]
    lat_train, lat_dev, lat_test = lat[0], lat[1], lat[2]
    
    x_train = [lng_train, lat_train]
    train_ffnn = create_traj_data_ffnn(x_train)
    x_dev = [lng_dev, lat_dev]
    dev_ffnn = create_traj_data_ffnn(x_dev)
    x_test = [lng_test, lat_test]
    test_ffnn = create_traj_data_ffnn(x_test)

    pickle.dump(train_ffnn, open(folder+'train_std_ff_outlier_detection.p', 'wb'))
    pickle.dump(dev_ffnn, open(folder+'dev_std_ff_outlier_detection.p', 'wb'))
    pickle.dump(test_ffnn, open(folder+'test_std_ff_outlier_detection.p', 'wb'))

        
def create_traj_dataset(folders, pct): 
    '''creates final datasets to be used for autoencoder models (lstm & ff)
    pct: float defining percentile of the trajectory dataset (e.g. if pct=25
    75% of the whole traj dataset is taken into account)
    '''
    
    folder = folders['data_folder']
    polyline_df = pd.read_csv(folder+'polyline_df.csv')
    lng_std, lat_std, lng, lat = get_coord_lists(folder, polyline_df, pct)
    create_traj_lstm_dataset(folder, lng_std, lat_std, standarize=True) 
    create_traj_lstm_dataset(folder, lng, lat, standarize=False) 
    create_traj_ff_dataset(folder, lng, lat)   



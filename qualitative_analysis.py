'''In this file we rank the trajectories in our real test set with Seq model and
both MSE+AVG & WEI_MSE+LOF methods in order to get qualitative insights on 
detected anomalies'''

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from sklearn.neighbors import LocalOutlierFactor


def read_shapefile_great_porto(folders):
    '''returns a shapefile of Porto administrative areas'''
    
    folder = folders['data_folder']
    file = folder+'shapefiles/Portugal_osm.shp'
    return gpd.read_file(file)


def preds_seq2seq(folders, wei):    
    ''' 
    saves array of predicted traj of test set 
    to pickle. 
    wei=True if weighted mse is used as loss function
    Len of array = number of test instances
    '''
            
    if wei==True:
        preds_std = pickle.load(open(folders['preds_folder']+'pred_traj_std_seq.p', 'rb'))
    else:
        preds_std = pickle.load(open(folders['preds_folder']+'pred_traj_std_seq_mse.p', 'rb'))
    
    train_avg_std_lat = pickle.load(open(folders['data_folder']+'train_lat_avg_std.p', 'rb'))
    train_avg_lat, train_stddev_lat = train_avg_std_lat[0], train_avg_std_lat[1]

    train_avg_std_lng = pickle.load(open(folders['data_folder']+'train_lng_avg_std.p', 'rb'))
    train_avg_lng, train_stddev_lng = train_avg_std_lng[0], train_avg_std_lng[1]
    
    preds = np.empty((preds_std.shape[0], preds_std.shape[1],preds_std.shape[2]))
    preds[:,:,0] = (preds_std[:,:,0]*train_stddev_lng)+train_avg_lng  #destandardize preds
    preds[:,:,1] = (preds_std[:,:,1]*train_stddev_lat)+train_avg_lat

    if wei==True:
        pickle.dump(preds, open(folders['preds_folder']+'pred_traj_seq.p', 'wb'))
    else:
        pickle.dump(preds, open(folders['preds_folder']+'pred_traj_seq_mse.p', 'wb'))
    

def errors_per_timestep(folders, wei):
    '''saves to pickles mse of test set in each timestep'''
    
    actual = pickle.load(open(folders['data_folder']+'test_lstm_outlier_detection.p', 'rb')) 
    
    if wei==True:
        preds = pickle.load(open(folders['preds_folder']+'pred_traj_seq.p',  'rb'))
    else:
        preds = pickle.load(open(folders['preds_folder']+'pred_traj_seq_mse.p',  'rb'))
    
    errors = []
    for i in range(preds.shape[0]): 
        err = [] 
        for j in range(preds.shape[1]): 
            a, p = actual[i,j], preds[i,j] 
            e = ((a[0]-p[0])**2)+((a[1]-p[1])**2) 
            err.append(e/2) 
        errors.append(err) 
    
    if wei==True:        
        pickle.dump(np.array(errors), open(folders['errors_folder']+'errors_per_timestep.p', 'wb'))
    else:
        pickle.dump(np.array(errors), open(folders['errors_folder']+'errors_per_timestep_mse.p', 'wb'))
        
        
def find_seq_m_avg_ind(folders):
    '''saves to pickle list of indices of 0,1% highest errors for seq with loss=mse&
    errors ranked by avg values per timestep'''
    
    errors = pickle.load(open(folders['errors_folder']+'errors_per_timestep_mse.p', 'rb'))
    aver = np.mean(errors, axis=1)
    thr = np.percentile(aver, 99.90)
    outlier_id = np.where(aver>thr)[0] # 0.1% have scores>thr & are the most anomalous
    
    outl_ids = []
    for i in range(len(outlier_id)):
        outl_ids.append(outlier_id[i])
    
    pickle.dump(outl_ids, open(folders['errors_folder']+'seq_mse_avg_indices.p', 'wb'))  
    
    
def train_lof_errors(folders, wei):
    '''train Local Outlier Factor on errors of test set & save to pickles 
    the scores
    wei=True if weighted mse is used as loss function while training Seq2Seq'''
       
    errors = pickle.load(open(folders['errors_folder']+'errors_per_timestep.p', 'rb'))
      
    neighb = 100
        
    model = LocalOutlierFactor(novelty=False, n_neighbors=neighb, metric='euclidean')
    preds = model.fit_predict(errors)
    scores = model.negative_outlier_factor_

    pickle.dump(np.array(scores), open(folders['errors_folder'] + 'pred_lof_error_scores_seq.p', 'wb')) 
    

def find_seq_h_lof_ind(folders):
    '''saves to pickle list of indices of 0,1% lowest scores for lstm with
    loss=weight_mse & lof applied on errors'''
    
    lof_scores = pickle.load(open(folders['errors_folder']+
                                  'pred_lof_error_scores_seq.p', 'rb'))
    thr = np.percentile(lof_scores, 0.1) # 0.1% have scores<thr & are the most anomalous
    outlier_id = np.where(lof_scores<thr)[0] 
    
    outl_ids = []
    for i in range(len(outlier_id)):
        outl_ids.append(outlier_id[i])
    
    pickle.dump(outl_ids, open(folders['errors_folder']+'seq_hav_lof_indices.p', 'wb'))  


def create_geodf_act(max_err_id, actual_traj):
    '''returns 2 geodataframes of points & lines of actual traj'''
    
    traj_act = actual_traj[max_err_id]
    traj_act_lng = traj_act[:,0]   
    traj_act_lat = traj_act[:,1] 
    
    coords_act = pd.DataFrame(columns=['lat', 'lng', 'tid','uid'])
    coords_act['lng'] = traj_act_lng
    coords_act['lat'] = traj_act_lat
    coords_act['tid'] = max_err_id
    
    #create df for pred, actual traj (1 row per traj)
    traj_df = pd.DataFrame(columns=['tid'])
    traj_df['tid'] = max_err_id
    
    #create geodataframes for points, traj of actual trajectories
    crs = {'init':'epsg:4326'}    
    point_geo_act = [Point(xy) for xy in zip(coords_act["lng"], coords_act["lat"])]
    geo_points_act = gpd.GeoDataFrame(coords_act, crs=crs, geometry=point_geo_act)
    coord_list_act = []
    for i in range(len(traj_act_lng)):
        coord_list_act.append((traj_act_lng[i],traj_act_lat[i]))
    line_geo_act = [LineString(coord_list_act)]
    geo_traj_act = gpd.GeoDataFrame(traj_df, crs=crs, geometry=line_geo_act)
    return geo_points_act, geo_traj_act

        
def plot_anom_traj(folders, points_act, traj_act, outl_id):
    '''plots actual  trajectory
    points_act, traj_act: geodataframes of points, lines
    outl_id:index of anom traj in test set'''
    
    admin_map = read_shapefile_great_porto(folders) #read shp file of Porto map
    plt.clf()        
    fig, ax = plt.subplots(figsize=(10,10)) 
    admin_map.plot(ax=ax, alpha=0.2, color='grey')  #plot map
    points_act.plot(ax=ax, markersize=10, color="blue", marker="o") #plot points
    traj_act.plot(ax=ax, markersize=10, color="blue", label="true trajectory") 
    #get borders of the map
    xmin, xmax, ymin, ymax = -8.730389, -8.513873, 41.016351, 41.296386  
    ax.grid(False)
    plt.xlim(xmin, xmax)
    plt.xticks([])
    plt.ylim(ymin, ymax)
    plt.yticks([])
    plt.show()
        

def plot_anomalies(folders):    
    '''plots maps of the detected outliers with Seq2Seq either
    with (los==mse, method=avg) or (loss=wei_mse, method=lof)'''
    
    actual_traj = pickle.load(open(folders['data_folder']+'test_lstm_outlier_detection.p', 'rb')) 
    
    seq_m_avg_ind = pickle.load(open(folders['errors_folder']+
                                     'seq_mse_avg_indices.p', 'rb'))
    seq_h_lof_ind = pickle.load(open(folders['errors_folder']+
                                     'seq_hav_lof_indices.p', 'rb'))
    un = list(set(seq_m_avg_ind).union(seq_h_lof_ind))
    inter = list(set(seq_m_avg_ind) & set(seq_h_lof_ind))
    
    #choose only indices that are not common in the 2 lists    
    ind = np.setdiff1d(un,inter) #take only indices tha are in the union but not in intersection
    
    m_avg_ind, h_lof_ind = [], []
    for i in ind:
        if i in seq_m_avg_ind:
            m_avg_ind.append(i)
        if i in seq_h_lof_ind:
            h_lof_ind.append(i)
        
    
    metrics = ['wei_lof', 'mse_avg']
    for m in metrics:
        if m=='wei_lof':
            for i in h_lof_ind:
                points_act, traj_act = create_geodf_act(i, actual_traj)
                plot_anom_traj(folders, points_act, traj_act, i)
        else:
            for i in m_avg_ind:
                points_act, traj_act = create_geodf_act(i, actual_traj)
                plot_anom_traj(folders, points_act, traj_act, i)

                
      
def find_outl_traj_seq(folders):    
    '''find and plot anomalous traj with Seq2Seq model and 2 variants of prediction:
        (loss=mse, method=avg), (loss=wei_mse, method=lof)'''
    
    weights = [True, False]
    for wei in weights:
        preds_seq2seq(folders, wei) #get predicted trajectories
        errors_per_timestep(folders, wei) #get errors per timestep
        
    train_lof_errors(folders, wei=True)
    '''get indices of highest errors for seq2seq with loss=mse and metric=avg & 
    with loss=weighted mse and metric=lof'''
    find_seq_m_avg_ind(folders)
    find_seq_h_lof_ind(folders)
    
    plot_anomalies(folders)
            

def plot_norm_traj(folders):
    '''finds & plots trajectories with the lowest reconstruction errors with the
    method 'wei_mse+lof' '''
    
    lof_scores = pickle.load(open(folders['errors_folder']+'pred_lof_error_scores_seq.p', 'rb'))
    thr = np.percentile(lof_scores, 99.95) # 0.05% have scores>thr & are the most normal
    norm_traj_id = np.where(lof_scores>thr)[0] 
        
    actual_traj = pickle.load(open(folders['data_folder']+
                                   'test_lstm_outlier_detection.p', 'rb')) 
    
    for i in norm_traj_id:
        points_act, traj_act = create_geodf_act(i, actual_traj)
        plot_anom_traj(folders, points_act, traj_act, i) 
    
    

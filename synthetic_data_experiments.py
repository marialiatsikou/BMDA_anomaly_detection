'''In this file we make predictions for all models and evaluate them on 
synthetic datasets'''


import pickle
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras import backend as K 
from keras.models import Model
from keras.layers import Input

import matplotlib.pyplot as plt

from traject_preprocess import get_synthetic_experiment_folders
        

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


def get_synthetic_data(data, synthetic_type, pct_outliers=0.01):
    '''Generates the synthetic dataset. Inputs:
        (a) data to alter
        (b) synthetic type: a string (mixed_full, mixed_half, cycle_half, cycle_back_forth)
        (c) % of the len(data) to alter according to the "synthetic_type" pattern
    Returns:
        (a) the new data, with the first pct % altered according to the synthetic_type
        (b) the binary labels indicating the presence of an anomaly or not
    '''
    num_outliers = int(len(data)*pct_outliers)
    new_data, labels  = [], []
    cnt = 0
    for i in range(len(data)):
        x = data[i]
        if cnt<num_outliers:
            cnt+=1
            
            if synthetic_type=='mixed_full': 
                mixed = [x[0], x[8], x[1], x[7], x[2], x[6], x[3], x[5], x[4]]
            elif synthetic_type=='mixed_half':
                mixed = [x[0], x[1], x[2], x[4], x[3], x[5], x[6], x[7], x[8]]
            elif synthetic_type=='cycle_half':
                mixed = [x[0], x[1], x[2], x[3], x[0], x[1], x[2], x[3], x[4]]
            elif synthetic_type=='cycle_back_forth':
                mixed = [x[3], x[4], x[5], x[4], x[3], x[4], x[5], x[4], x[3]] 
                
            new_data.append(mixed)
            labels.append(1)#"1" for anomaly
        else:
            new_data.append(x)
            labels.append(0) #"0" for inlier
            
    return np.array(new_data), labels


def predict_sequence(enc, dec, source, n_steps, num_feat):
    '''generate target given source sequence
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
        yhat, h, c = dec.predict([target_seq] + state)
        output.append(yhat[0,0,:])
        state = [h, c]
        target_seq = yhat
    
    return(np.array(output))
    
    
def predict_seq2seq(model, testX_std):
    '''Returns predictions for Seq2Seq model'''
    
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_inputs = model.input[1]   # input_2
    decoder_state_input_h = Input(shape=(16,), name='input_3')
    decoder_state_input_c = Input(shape=(16,), name='input_4')
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    #inference model
    preds = []
    for i in range(testX_std.shape[0]):
        pred_traj_std = predict_sequence(encoder_model, decoder_model, testX_std[i], \
                         n_steps=testX_std.shape[1], num_feat=2) #get preds for one traj
        preds.append(pred_traj_std)
        
    return np.array(preds)
    

def make_preds_on_synthetic_data_lof(folders, type_of_synthetic):
    '''This function is for LOF
    Given a "type_of_synthetic" (mixed_full, mixed_half, cycle_half, cycle_back_forth),
    it saves the predictions made by LOF in the "errors_folder"'''
    
    from sklearn.neighbors import LocalOutlierFactor
    
    data_folder = folders['data_folder']
    errors_folder = folders['errors_folder']

    #load the actual data in the test set
    testX = pickle.load(open(data_folder+'test_lstm_outlier_detection.p', 'rb'))#actual, non-normalised, no noise

    #generate synthetic data by altering testX and re-arrange it
    pseudoX, labels = get_synthetic_data(testX, type_of_synthetic)
    pseudoX = np.concatenate((pseudoX[:,:,0],pseudoX[:,:,1]), axis=1)  
    
    #train and make predictions:
    new_model = LocalOutlierFactor(novelty=False, n_neighbors=100, metric='euclidean')
    _ = new_model.fit_predict(pseudoX)
    preds = new_model.negative_outlier_factor_

    #save predictions:
    pickle.dump(preds, open(errors_folder+'lof_'+type_of_synthetic+'.p', 'wb')) 
           
    
def make_preds_on_synthetic_data(folders, modelname, type_of_synthetic):
    '''
    - modelname: vae, vae_mse, seq, seq_mse, ffae_mse
    - type_of_synthetic: mixed_full, mixed_half, cycle_half, cycle_back_forth
    '''
    data_folder = folders['data_folder']
    model_folder = folders['model_folder']
    preds_folder = folders['preds_folder']

    #load the data and generate the synthetic ones:
    testX = pickle.load(open(data_folder+'test_noise_std_lstm.p', 'rb'))#actual, normalised, with noise
    pseudoX, labels = get_synthetic_data(testX, type_of_synthetic)
        
    #for Feed-Forward: rearranging data into (18,1)    
    if modelname=='ffae_mse':
        pseudoX = np.concatenate((pseudoX[:,:,0],pseudoX[:,:,1]), axis=1)      
        
    #load the model
    if modelname in ['seq', 'vae']:
        new_model = load_model(model_folder+'best_'+modelname+'.h5', 
                           custom_objects = {'loss_wei_hav':custom_loss_wei()})
    else:
        new_model = load_model(model_folder+'best_'+modelname+'.h5')

    #make predictions
    if modelname[0:3]=='seq':
        preds = predict_seq2seq(new_model, pseudoX)
    else:
        preds = new_model.predict(pseudoX)    

    #for feedforward: rearranging data again into (9,2)
    if modelname=='ffae_mse':
        thr = int(pseudoX.shape[1]/2)
        testX_3d =  np.empty((testX.shape[0], testX.shape[1],testX.shape[2]))
        testX_3d[:,:,0] = pseudoX[:,:thr]
        testX_3d[:,:,1] = pseudoX[:,thr:]
        
        preds_3d = np.empty((testX.shape[0], testX.shape[1],testX.shape[2]))
        preds_3d[:,:,0] = preds[:,:thr]
        preds_3d[:,:,1] = preds[:,thr:]
        
        pseudoX = testX_3d
        preds = preds_3d
        
    #save predictions:
    pickle.dump(preds, open(preds_folder+modelname+'_'+type_of_synthetic+'.p', 'wb'))
            

def run_avg_lof(folders, modelname, synthetic):
    '''compute reconstruction errors and their average/LOF scores. 
    Store results in "errors" folder'''
    
    from sklearn.neighbors import LocalOutlierFactor
    
    data_folder = folders['data_folder']
    errors_folder = folders['errors_folder']
    predictions_folder = folders['preds_folder']
    
    longs = pickle.load(open(data_folder+'train_lng_avg_std.p', 'rb')) 
    lats = pickle.load(open(data_folder+'train_lat_avg_std.p', 'rb')) 
    

    actuals = pickle.load(open(data_folder+'test_lstm_outlier_detection.p', 'rb')) #actual, non-normalised, non-noise
    actuals, labels = get_synthetic_data(actuals, synthetic) #rearrange them      
    predictions = pickle.load(open(predictions_folder+modelname+'_'+synthetic+'.p', 'rb'))
    
    errors = []
    for i in range(len(actuals)):
        y_true, y_pred = actuals[i], predictions[i]
        err = []
        for t in range(y_true.shape[0]):#for each timestep
            ytrue_lng, ytrue_lat = y_true[t][0], y_true[t][1]
            ypred_lng, ypred_lat = (longs[1]*y_pred[t][0])+longs[0], (lats[1]*y_pred[t][1])+lats[0] #denormalise predictions
            
            e_lng = (ytrue_lng-ypred_lng)**2
            e_lat = (ytrue_lat-ypred_lat)**2
            
            e = np.average([e_lng, e_lat])
            err.append(e)
        errors.append(err)
        
    avg = np.average(errors, axis=1)
    
    #LOF
    model = LocalOutlierFactor(novelty=False, n_neighbors=1000, metric='euclidean')
    model.fit_predict(errors)
    lof = model.negative_outlier_factor_
    
    results = [errors, avg, lof]
    pickle.dump(results, open(errors_folder+synthetic+'_'+modelname+'.p', 'wb'))


def get_evaluation_metric(folders, modelname, synthetic, pct=5, metric='fscore', scr=''): 
    '''for every model & every type of synthetic data this function computes
    precision, recall, fscore for avg & lof method (all models except for LOF, NRR)
    '''
    
    errors_folder = folders['errors_folder']

    scores = pickle.load(open(errors_folder+synthetic+'_'+modelname+'.p', 'rb'))
    labels = [1 for i in range(int(len(scores[0])*0.01))]
    labels.extend([0 for i in range(len(scores[0])-int(len(scores[0])*0.01))])
    
    errors = scores[0]
    anomaly_detector = [scores[1], scores[2]]#avg, lof
    for cnt in range(len(anomaly_detector)): #for every one of avg, lof
        scores = anomaly_detector[cnt]  #all scores for test set
        
        #sort based on scores
        ordered_errors, ordered_labels = zip(*sorted(zip(scores, labels))) 
        ordered_errors, ordered_labels = list(ordered_errors), list(ordered_labels)
        if cnt!=len(anomaly_detector)-1: #for avg, get highest errors up
            ordered_errors.reverse()
            ordered_labels.reverse()
        
        threshold = int(len(errors)*(pct/100))
        tmp = ordered_labels[:threshold] #get the labels up until the top-5% of our test set
        
        #how many anomalies we found in the top-5%
        true_positive = len(np.where(np.array(tmp)==1)[0]) 
        num_actual = int(len(errors)*0.01) #how many anomalies exist overall
        #number of predictions we made until the top-5% of the test set
        num_predictions = len(tmp) 
        
        recall = 100*(true_positive/num_actual)
        precision = 100*true_positive/num_predictions
        fscore = (2*precision*recall)/(precision+recall)
        if metric=='recall':
            fscore = recall
        elif metric=='precision':
            fscore = precision
        scr+='\t'+str(np.round(fscore, 2))
        
    return scr


def evaluate_lof(folders,  synthetic='mixed_full', metric='fscore'):
    '''computes precision, recall, fscore for LOF method applied on
    every type of synthetic data'''
    
    errors_folder = folders['errors_folder']
    
    scores = pickle.load(open(errors_folder+'lof_'+synthetic+'.p', 'rb'))
    labels = [1 for i in range(int(len(scores)*0.01))]
    labels.extend([0 for i in range(len(scores)-int(len(scores)*0.01))])
    labels = np.array(labels)
    
    ordered_errors, ordered_labels = zip(*sorted(zip(scores, labels)))
    ordered_errors, ordered_labels = list(ordered_errors), list(ordered_labels)

    threshold = int(len(labels)*(5/100))
    tmp = ordered_labels[:threshold]
    recall = 100*len(np.where(np.array(tmp)==1)[0])/int(len(labels)*0.01)
    precision = 100*len(np.where(np.array(tmp)==1)[0])/len(tmp)
    fscore = (2*precision*recall)/(precision+recall)
    if metric=='recall':
        fscore = recall
    elif metric=='precision':
        fscore = precision
    print('LOF\t', synthetic, '\t', np.round(fscore,2))
    

def evaluate_random(folders, modelname, synthetic='mixed_full'):
    
    predictions_folder = folders['preds_folder']
    
    scores = pickle.load(open(predictions_folder+modelname+'_'+synthetic+'.p', 'rb'))
    labels = [1 for i in range(int(len(scores)*0.01))]
    labels.extend([0 for i in range(len(scores)-int(len(scores)*0.01))])
    labels = np.array(labels)
    
    fscores, precs, recs = [], [], []
    for i in range(10000):
        scores = np.random.randn(len(labels))
        ordered_errors, ordered_labels = zip(*sorted(zip(scores, labels)))
        ordered_errors, ordered_labels = list(ordered_errors), list(ordered_labels)

        
        threshold = int(len(labels)*(5/100))
        tmp = ordered_labels[:threshold]
        
        recall = 100*len(np.where(np.array(tmp)==1)[0])/int(len(labels)*0.01)
        precision = 100*len(np.where(np.array(tmp)==1)[0])/len(tmp)
        fscore = (2*precision*recall)/(precision+recall)

        precs.append(precision)
        recs.append(recall)
        fscores.append(fscore)
    print(np.average(recs), '\t', np.average(precs), '\t', np.average(fscores))


def get_fscore_over_k(folders, modelname, synthetic, metric='fscore'): #all models except for LOF, NRR

    errors_folder = folders['errors_folder']    

    scores = pickle.load(open(errors_folder+synthetic+'_'+modelname+'.p', 'rb'))
    labels = [1 for i in range(int(len(scores[0])*0.01))]
    labels.extend([0 for i in range(len(scores[0])-int(len(scores[0])*0.01))])
    
    errors = scores[0]
    avg = scores[1]
    lof = scores[2]
    
    anomaly_detector = [avg, lof]
    timeseries = []
    for cnt in range(len(anomaly_detector)):
        fscores = []
        scores = anomaly_detector[cnt]
        
        ordered_errors, ordered_labels = zip(*sorted(zip(scores, labels)))
        ordered_errors, ordered_labels = list(ordered_errors), list(ordered_labels)
        if cnt!=len(anomaly_detector)-1:
            ordered_errors.reverse()
            ordered_labels.reverse()
        
        for pct in range(1,501):
            threshold = int(len(errors)*(pct/1000))
            tmp = ordered_labels[:threshold]
            recall = 100*len(np.where(np.array(tmp)==1)[0])/int(len(errors)*0.01)
            precision = 100*len(np.where(np.array(tmp)==1)[0])/len(tmp)
            fscore = (2*precision*recall)/(precision+recall+0.000000001)

            if metric=='recall':
                fscore = recall
            elif metric=='precision':
                fscore = precision
            fscores.append(fscore)
        timeseries.append(np.array(fscores))
        
    return np.array(timeseries)


def run_all_models(folders):
    '''This function is the main function. Predictions are made for all models
    and then they are evaluated'''
    
    #our synthetic datasets
    syntheses = ['mixed_full', 'mixed_half', 'cycle_half', 'cycle_back_forth'] 
    #our 5 models (excl. LOF)
    models = ['ffae_mse', 'vae_mse', 'seq_mse', 'vae', 'seq'] 
    
    #make predictions
    for synthesis in syntheses:
        #first run LOF on the raw trajectory data
        make_preds_on_synthetic_data_lof(folders,  synthesis) 
        for model in models: 
            #first make the reconstructions and store them in "preds_folder"
            make_preds_on_synthetic_data(folders, model, synthesis) 
            #then use the reconstruction errors to get their average/LOF
            run_avg_lof(folders, model, synthesis) 
            
    #evaluate all models
    eval_metrics = ['precision', 'recall', 'fscore']
    for eval_metric in eval_metrics:
        print('\n\t\t\t\t\t\t', eval_metric)
        for model in models:
            scr = '' #used as a astring for printing out the results
            for synthesis in syntheses:
                scr = get_evaluation_metric(folders, model, synthesis, 5, eval_metric, scr) #"5" for eval@5%
            if model in ['vae', 'seq']:
                model = model + '    '  
            print(model, '\t', scr)
        for synthesis in syntheses:
            evaluate_lof(folders, synthesis, eval_metric)
            #evaluate_random(folders, modelname='seq', synthetic=synthesis)
  

def generate_charts(folders):
    
    img_folder = folders['images_folder']

    cnt = -1
    for modelname in ['vae', 'seq']:
        cnt = -1
        names = ['DSTRT_c', 'DSTRT_p', 'CYCLE_c', 'CYCLE_b', 'SPEED_e', 'SPEED_h']
        for dataset in ['mixed_full', 'mixed_half', 'cycle_half', 'cycle_back_forth']:#, 'traffic_fast', 'traffic_moderate']:
            cnt+=1
            ts_mse = get_fscore_over_k(folders, modelname+'_mse', dataset)
            ts_hav = get_fscore_over_k(modelname, dataset)
            
            x = (1+np.arange(500))/10
            y1, y2, y3, y4 = ts_mse[0], ts_mse[1], ts_hav[0], ts_hav[1]
            
            plt.clf()
            plt.plot(x, y1, label='MSE+AVG', linewidth=2)
            plt.plot(x, y2, label='MSE+LOF', linewidth=2)
            plt.plot(x, y3, label='HVR+AVG', linewidth=2, linestyle='--')
            plt.plot(x, y4, label='HVR+LOF', linewidth=2, linestyle='--')
            if cnt==0:
                plt.ylabel('F1 Measure at k', fontsize=28)
            plt.xlabel('k', fontsize=28)
            if dataset=='cycle_back_forth':
                plt.legend(prop={'size': 20})
            plt.title(names[cnt], fontsize=28)
            plt.grid(alpha=0.2)
            plt.savefig(img_folder+str(dataset)+'_'+str(modelname)+'.png', 
                        dpi=150, bbox_inches='tight')
            

def merge_charts(folders):
    
    from PIL import Image    
    folder = folders['images_folder']
    
    for m in ['vae', 'seq']:
        raw_images = [folder+'mixed_full_'+m+'.png', 
                      folder+'mixed_half_'+m+'.png',
                      folder+'cycle_half_'+m+'.png', 
                      folder+'cycle_back_forth_'+m+'.png']
        images = [Image.open(x) for x in raw_images]
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]
        
        new_im.save(folder+m+'.png')
 
    im1 = Image.open(folder+'vae.png')
    im2 = Image.open(folder+'seq.png')
    dst = Image.new('RGB', (im2.width, im1.height + im2.height-4))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height-4))
    dst.save(folder+'all_charts.png')
    
            
    

        


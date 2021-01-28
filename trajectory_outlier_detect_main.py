'''This file contains the main function'''

from traject_preprocess import get_synthetic_experiment_folders, create_traj_dataset
from seq2seq_model import train_seq2seq
from lstm_ff_models import run_lstm_ff
from synthetic_data_experiments import run_all_models, generate_charts, merge_charts
from qualitative_analysis import find_outl_traj_seq, plot_norm_traj


def outlier_detection_main(train_model):
    
    folders = get_synthetic_experiment_folders() #create a dict of file folders
    create_traj_dataset(folders, pct=25) #create train, dev, test sets
    
    '''train Seq2Seq for weighted & unweighted loss function 
    (if train_model=True)  & save preds for original test set'''
    weights = [True, False] 
    for wei in weights:     
        train_seq2seq(folders, wei, train_model, n_units=16)
    
    if train_model==True:
        run_lstm_ff(folders) #train LSTM & feedforward autoencoders
    
    #make predictions for all models & evaluate them on synthetic data
    run_all_models(folders) 
    generate_charts(folders)
    merge_charts(folders)
    
    '''detect outliers in our real test with Seq model and both MSE+AVG & 
    WEI_MSE+LOF methods'''
    find_outl_traj_seq(folders)    
    plot_norm_traj(folders)


outlier_detection_main(train_model=False)
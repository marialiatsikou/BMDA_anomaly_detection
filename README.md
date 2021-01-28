# BMDA_anomaly_detection

Python code for the paper "A Denoising Hybrid Model for Anomaly Detection in Trajectory Sequences" (BMDA 2021 Workshop). 

Folder contents: 

* data: files neccessary for running the code 
* models: the files of models saved while running the code

First download the file from [here](https://www.dropbox.com/s/jis8cgnb7vglvvc/data.zip?dl=0) and extract its contents in the folder "data" (>500MB). It contains the following:

* shapefiles folder: the files that are needed for  plotting the maps;
* polyline_df.csv: a csv file containing all the valid point coordinates (lng, lat) of all trajectories with the corresponding trip ids and timestamps (each row corresponds to a point);
* XYZ_noise_std_lstm.p: pickles with the noise added in the XYZ (train/dev/test) set.

The file "trajectory_outlier_detect_main.py" contains the main function for running the code. This function has the "train_model" variable, which should be defined by the user. It determines if the models will be trained: If True the models are trained and saved to folder "models" If False the models are loaded from folder "models" in order to make predictions.

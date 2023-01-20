# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:02:13 2022

@author: daizhongpengMNE
"""
import os.path as op
import os
import sys
import numpy as np
from mne.preprocessing import annotate_muscle_zscore
import mne

data_path = r'C:\MSc_Project\Data_Patient\sub006_s1_20220523_b588'
result_path = data_path
file_name = ['rest_eyes_opensss-1.fif','rest_eyes_opensss-2.fif']

path_data = os.path.join(data_path,file_name[0]) 

data1 = mne.io.read_raw_fif(path_data,preload=True)

eog_events = mne.preprocessing.find_eog_events(data1, ch_name='EOG003') 

n_blinks = len(eog_events)
onset = eog_events[:, 0] / data1.info['sfreq'] - 0.25  
duration = np.repeat(0.5, n_blinks)  
description = ['blink'] * n_blinks  
orig_time = data1.info['meas_date']
annotations_blink = mne.Annotations(onset,duration,description,orig_time)

threshold_muscle = 5
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    data1, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])

data1.set_annotations(annotations_blink+annotations_muscle)

annotations_blink.onset

# data1.plot(start=50)
# # Set the channel type as 'eog'
# data1.set_channel_types({'EOG003': 'eog'})
# data1.set_channel_types({'EOG004': 'eog'})

# eog_picks = mne.pick_types(data1.info, meg=False, eog=True)

# scl = dict(eog=500e-6)
# data1.plot(order=eog_picks, scalings=scl, start=50);

data1.plot(start=50);
rawinterp = []
data1 = data1.copy().interpolate_bads(reset_bads=True,origin=(0,0,0))
rawinterp.append(data1)

path_file_results = os.path.join(result_path,'training_rawann-1.fif') 
path_file_results
data1.save(path_file_results,overwrite=True) 

path_data = os.path.join(data_path,file_name[1]) 
data2 = mne.io.read_raw_fif(path_data,preload=True)

# Blinks
eog_events = mne.preprocessing.find_eog_events(data2,ch_name = 'EOG003') 
n_blinks = len(eog_events)
onset = eog_events[:, 0] / data2.info['sfreq'] - 0.25 
duration = np.repeat(0.5, n_blinks)  
description = ['blink'] * n_blinks  
orig_time = data2.info['meas_date']
annotations_blink = mne.Annotations(onset,duration,description,orig_time)

# Muscle
threshold_muscle = 5
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    data2, ch_type="mag", threshold=threshold_muscle, min_length_good=0.25,
    filter_freq=[110, 140])


# Annotate
data2.set_annotations(annotations_blink+annotations_muscle)  

# scl = dict(eog=500e-6)
# data2.plot(order=eog_picks, scalings=scl, start=50);

data2.plot(start=50);
rawinterp = []
data2 = data2.copy().interpolate_bads(reset_bads=True,origin=(0,0,0))
rawinterp.append(data2)

# Save FIF
path_file_results = os.path.join(result_path,'training_rawann-2.fif') 
path_file_results
data2.save(path_file_results,overwrite=True) 







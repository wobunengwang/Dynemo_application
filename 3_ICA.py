# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:04:08 2022

@author: daizhongpengMNE
"""
import os.path as op
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

import mne

data_path = r'C:\MSc_Project\Data_Patient\sub006_s1_20220523_b588'
result_path = data_path

file_name = 'training_raw'

for subfile in range(1, 3):
    path_file = os.path.join(result_path,file_name + 'ann-' + str(subfile) + '.fif') 
    raw = mne.io.read_raw_fif(path_file,allow_maxshield=True,verbose=True,preload=True)
    #raw_resmpl = raw.copy().pick_types(meg=True,eog=True,ecg=True)
    raw_resmpl = raw.copy().pick_types(meg=True)
    raw_resmpl.resample(200)
    raw_resmpl.filter(1, 40)
    if subfile == 1:
        raw_resmpl_all = mne.io.concatenate_raws([raw_resmpl])
    else:
        raw_resmpl.info = raw_resmpl_all.info
        raw_resmpl_all = mne.io.concatenate_raws([raw_resmpl_all, raw_resmpl])
del raw_resmpl

ica = ICA(method='fastica',
    random_state=97,
    n_components=30,
    verbose=True)

ica.fit(raw_resmpl_all,
    verbose=True)

ica.plot_sources(raw_resmpl_all,title='ICA');
ica.plot_components();

# Set the 4 components to exclude
ica.exclude = [4]

# Loop over the subfiles 
for subfile in range(1, 3):
    path_file = os.path.join(result_path,file_name + 'ann-' + str(subfile) + '.fif') 
    #path_file = os.path.join(result_path,file_name[subfile]+'sss.fif') 
    path_outfile = os.path.join(result_path,file_name +'ica-' + str(subfile) + '.fif') 

    raw_ica = mne.io.read_raw_fif(path_file,allow_maxshield=True,verbose=True,preload=True)   
    ica.apply(raw_ica)

    raw_ica.save(path_outfile,overwrite=True) 
    
# chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411']
# chan_idxs = [raw.ch_names.index(ch) for ch in chs]
# raw.plot(order=chan_idxs, duration=5);
# raw_ica.plot(order=chan_idxs, duration=5);
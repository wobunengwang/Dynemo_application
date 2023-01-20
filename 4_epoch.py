# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:40:25 2022

@author: daizhongpengMNE
"""
import os.path as op
import os
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt

datapath =[r'C:\MSc_Project\Data_Control\sub001_20220609_b5e4',
              r'C:\MSc_Project\Data_Control\sub002_20220611_b399',
              r'C:\MSc_Project\Data_Control\sub003_20220611_b47b',
              r'C:\MSc_Project\Data_Control\sub004_20220611_b5aa',
              r'C:\MSc_Project\Data_Control\sub005_20220617_b5e5',
              r'C:\MSc_Project\Data_Control\sub006_20220621_b587',
              r'C:\MSc_Project\Data_Control\sub007_20220622_b460',
              r'C:\MSc_Project\Data_Control\sub008_20220623_b5e0',
              r'C:\MSc_Project\Data_Control\sub009_20220627_b397',
              r'C:\MSc_Project\Data_Control\sub010_20220627_b459',
              r'C:\MSc_Project\Data_Patient\sub001_s1_20210706_b5ec',
              r'C:\MSc_Project\Data_Patient\sub002_s1_20210708_b4b6',
              r'C:\MSc_Project\Data_Patient\sub003_s1_20220311_b392',
              r'C:\MSc_Project\Data_Patient\sub004_s1_20220422_b4bf',
              r'C:\MSc_Project\Data_Patient\sub005_s1_20220513_b483',
              r'C:\MSc_Project\Data_Patient\sub006_s1_20220523_b588']




subjects_to_do = np.arange(0, len(datapath))
for sub in subjects_to_do:
    data_path = datapath[sub]
    result_path = data_path
    
    file_name = 'training_raw'
    raw_list = list()
    events_list = list()
    
    
    for subfile in range(1, 3):
        # Read in the data from the Result path
        path_file = os.path.join(result_path,file_name + 'ica-' + str(subfile) + '.fif') 
        raw = mne.io.read_raw_fif(path_file, allow_maxshield=True,verbose=True,preload=True)
        ## remove power line
        meg_picks = mne.pick_types(raw.info, meg=True)
        freqs = (50,100,150)
        raw = raw.copy().notch_filter(freqs=freqs, picks=meg_picks)
        raw_list.append(raw)
        # events_list.append(events)
        
        
    
    #%% 8.数据分段：创建epochs
    raw = mne.concatenate_raws(raw_list,on_mismatch='ignore')
    epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)
    #epochs可视化
    epochs_c = epochs.copy().pick_types(meg=True,ref_meg=False)
    epochs_c.plot(n_epochs=2,picks='meg')
    
    meg_picks = mne.pick_types(epochs.info, meg=True)
    #保存处理完成的数据
    path_outfile = op.join(data_path, 'rest_epoch.fif') 
    epochs.save(path_outfile,overwrite=True) 
    
    ## plot epochs
    epochs.plot_psd(fmin=2, fmax=200)
    epochs.plot_psd_topomap(ch_type='grad', normalize=True)








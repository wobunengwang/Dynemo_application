# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 21:50:49 2022

@author: daizhongpengMNE
"""

import os.path as op
import os
import sys
import numpy as np

import mne
import matplotlib.pyplot as plt
import scipy

data_path = r'C:\MSc_Project\Data_Control\sub002_20220611_b399'
result_path = data_path
maxfiterpath = r'C:\MSc_Project'
file_name = 'rest_eyes_open'
path_file = os.path.join(data_path,file_name +'_1_raw.fif') 

# crosstalk_file = os.path.join(maxfiterpath,'sub-01_ses-01_meg_sub-01_ses-01_acq-crosstalk_meg.fif')
# cal_file = os.path.join(maxfiterpath,'sub-01_ses-01_meg_sub-01_ses-01_acq-calibration_meg.dat')
crosstalk_file = os.path.join(maxfiterpath,'sub-01_ses-01_meg_sub-01_ses-01_acq-crosstalk_meg.fif')
cal_file = os.path.join(maxfiterpath,'sub-01_ses-01_meg_sub-01_ses-01_acq-calibration_meg.dat')
data1 = mne.io.read_raw_fif(path_file, allow_maxshield=True,preload=True,verbose=True)

data1.info['bads'] = []
data1_check = data1.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
    data1_check, 
    cross_talk=crosstalk_file, 
    calibration=cal_file,
    return_scores=True, 
    verbose=True)

print('noisy =', auto_noisy_chs)
print('flat = ', auto_flat_chs)

data1.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
print('bads =', data1.info['bads'])

data1.fix_mag_coil_types()


data1_sss = mne.preprocessing.maxwell_filter(
    data1,
    cross_talk=crosstalk_file,
    calibration=cal_file,
    verbose=True)

data1.plot_psd(fmax=60, n_fft = 1000);

data1_sss.plot_psd(fmax=60, n_fft = 1000);

path_file_results = os.path.join(result_path,file_name+'sss-1.fif') 
path_file_results
# data1_sss.save(path_file_results,overwrite=True) 

path_file = os.path.join(data_path,file_name+'_2_raw.fif') 
path_file_results = os.path.join(result_path,file_name+'sss-2.fif')     
    
dataTmp = mne.io.read_raw_fif(path_file, allow_maxshield=True,preload=True,verbose=True)
dataTmp.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
dataTmp.fix_mag_coil_types()
dataTmp_sss = mne.preprocessing.maxwell_filter(
   dataTmp,
   cross_talk=crosstalk_file,
   calibration=cal_file,
   verbose=True)
# dataTmp_sss.save(path_file_results,overwrite=True) 
print(path_file_results)

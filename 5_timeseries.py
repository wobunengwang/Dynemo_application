# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:33:41 2022

@author: daizhongpengMNE
"""
import os.path as op
import os
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt

data_path = r'C:\MSc_Project\Data_Control\sub006_20220621_b587'

file_name = 'rest_epoch.fif'

path_file = os.path.join(data_path,file_name) 
epochs = mne.read_epochs(path_file, proj=True, preload=True, verbose=None)

# The get_data() method returns the epoched data as a NumPy array, of shape (n_epochs, n_channels, n_times)
meg_data = epochs.get_data(picks=['mag', 'grad'])


# np.save('C:\mtbi\data.npy',meg_data[:,0:5,:])

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 10:09:11 2023

@author: daizhongpengMNE
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:23:12 2022

@author: daizhongpengMNE
"""

import numpy as np
from os import makedirs
from osl_dynamics.models.dynemo import Config
from osl_dynamics import analysis, data, inference
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
datadir = r'C:\mtbi\osl-dynamics-main\input-6'
inference.tf_ops.gpu_growth()

nsub = 16
# Hyperparameters
config = Config(
    n_modes=3,  
    n_channels=42,
    sequence_length=100, # ?
    inference_n_units=64,
    inference_normalization="layer", # ?
    model_n_units=64, 
    model_normalization="layer", # ?
    learn_alpha_temperature=True,
    initial_alpha_temperature=1.0,
    learn_means=False,
    learn_covariances=True,
    do_kl_annealing=True,
    kl_annealing_curve="tanh",
    kl_annealing_sharpness=10,
    n_kl_annealing_epochs=10,
    batch_size=32,
    learning_rate=0.0025,
    gradient_clip=0.5,
    n_epochs=150,
    multi_gpu=False,
)

# Output id and folders
output_id = r"C:\mtbi\osl-dynamics-main\test_train_dzp"
model_dir = rf"{output_id}\model"
analysis_dir = rf"{output_id}\analysis"
maps_dir = rf"{output_id}\maps"
tmp_dir = rf"{output_id}\tmp"

makedirs(model_dir, exist_ok=True)
makedirs(analysis_dir, exist_ok=True)
makedirs(maps_dir, exist_ok=True)

# ----------------
# Training dataset

# Directory containing source reconstructed data
dataset_dir = datadir

# Load data
print("Reading MEG data")
training_data = data.Data(
    [rf"{dataset_dir}\subj{i}.mat" for i in range(0,nsub)],
    store_dir=tmp_dir,
)

# Plot Psd
data = training_data.subjects
fs = 1000
nper_seg=1024*4
S = np.zeros((int(nper_seg/2+1),config.n_channels,nsub))
for k in range(0,nsub):
    for i in range(0,config.n_channels):
        (f, S[:,i,k])= scipy.signal.welch(data[k][:,i], fs, nperseg = nper_seg)
    plt.figure()
    plt.semilogy(f, S[:,:,k])
    plt.xlim([0, 100])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = (
    "fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz"
)

# nsub = 11
# S = S[:,:,[0,2,5,6,8,9,10,11,12,13,15]]

# Plot theta spatial patterns
power_map = np.average(S[15:29,:,:],axis=0)
power_theta = np.zeros((nsub,config.n_channels,config.n_channels))
for i in range(0,nsub):
    row,col = np.diag_indices_from(power_theta[i,:,:])
    power_theta[i,row,col] = power_map[:,i]
    img = np.average(power_map,axis=1)
img = img-np.mean(img)
img = img/np.max(np.abs(img))
powertheta = np.zeros((config.n_channels,config.n_channels))
row,col = np.diag_indices_from(powertheta)
powertheta[row,col] = img
analysis.power.save(
    # power_map= power_theta,
    power_map= powertheta,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=rf"{maps_dir}/theta_.png",
    subtract_mean=True,
)

# Plot alpha spatial patterns
power_map = np.average(S[33:53,:,:],axis=0)
power_alpha = np.zeros((nsub,config.n_channels,config.n_channels))
for i in range(0,nsub):
    row,col = np.diag_indices_from(power_alpha[i,:,:])
    power_alpha[i,row,col] = power_map[:,i]
img = np.average(power_map,axis=1)
img = img-np.mean(img)
img = img/np.max(np.abs(img))
poweralpha = np.zeros((config.n_channels,config.n_channels))
row,col = np.diag_indices_from(poweralpha)
poweralpha[row,col] = img   
analysis.power.save(
    # power_map= power_alpha,
    power_map= poweralpha,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=rf"{maps_dir}/alpha_.png",
    subtract_mean=True,
)

# Plot beta spatial patterns
power_map = np.average(S[54:122,:,:],axis=0)
power_beta = np.zeros((nsub,config.n_channels,config.n_channels))
for i in range(0,nsub):
    row,col = np.diag_indices_from(power_beta[i,:,:])
    power_beta[i,row,col] = power_map[:,i]
img = np.average(power_map,axis=1)
img = img-np.mean(img)
img = img/np.max(np.abs(img))
powerbeta = np.zeros((config.n_channels,config.n_channels))
row,col = np.diag_indices_from(powerbeta)
powerbeta[row,col] = img      
analysis.power.save(
    # power_map= power_beta,
    power_map = powerbeta,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=rf"{maps_dir}/beta_.png",
    subtract_mean=True,
)

# Plot low gamma spatial patterns
power_map = np.average(S[123:204,:,:],axis=0)
power_lowgamma = np.zeros((nsub,config.n_channels,config.n_channels))
for i in range(0,nsub):
    row,col = np.diag_indices_from(power_lowgamma[i,:,:])
    power_lowgamma[i,row,col] = power_map[:,i]
img = np.average(power_map,axis=1)
img = img-np.mean(img)
img = img/np.max(np.abs(img))
powerlowgamma = np.zeros((config.n_channels,config.n_channels))
row,col = np.diag_indices_from(powerlowgamma)
powerlowgamma[row,col] = img          
analysis.power.save(
    # power_map= power_lowgamma,
    power_map= powerlowgamma,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=rf"{maps_dir}/lowgamma_.png",
    subtract_mean=True,
)

# Plot high gamma spatial patterns
power_map = np.average(S[209:290,:,:],axis=0)
power_highgamma = np.zeros((nsub,config.n_channels,config.n_channels))
for i in range(0,nsub):
    row,col = np.diag_indices_from(power_highgamma[i,:,:])
    power_highgamma[i,row,col] = power_map[:,i]
img = np.average(power_map,axis=1)
img = img-np.mean(img)
img = img/np.max(np.abs(img))
powerhighgamma = np.zeros((config.n_channels,config.n_channels))
row,col = np.diag_indices_from(powerhighgamma)
powerhighgamma[row,col] = img             
analysis.power.save(
    # power_map= power_highgamma,
    power_map= powerhighgamma,
    mask_file=mask_file,
    parcellation_file=parcellation_file,
    filename=rf"{maps_dir}/highgamma_.png",
    subtract_mean=True,
)

########### test region ############
# import scipy
# import matplotlib.pyplot as plt

# fs = 1000
# nper_seg=1024*5
# S = np.zeros((int(nper_seg/2+1),306))
# # meg_data_1 = np.load('C:\mtbi\data.npy')
# meg_data1 = np.reshape(meg_data,(306,55*10000))
# meg_data1 = meg_data1[:,50000:150000]
# for k in range(0,306):
#     (f, S[:,k])= scipy.signal.welch(meg_data1[k,:], fs, nperseg = nper_seg)
# plt.figure()
# plt.semilogy(f, S)
# plt.xlim([0, 100])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()



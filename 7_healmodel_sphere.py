
# #######################################lcmv
import scipy.io
import os
import os.path as op
import numpy as np
import mne
from mne.cov import compute_covariance
from mne.beamformer import make_lcmv, apply_lcmv_cov, make_dics, apply_dics_csd,apply_lcmv_epochs
from mne.time_frequency import csd_morlet
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
from sklearn.preprocessing import normalize
datadir = r'C:\mtbi\osl-dynamics-main\input-6'

resultpath =[r'C:\MSc_Project\Data_Control\sub001_20220609_b5e4',
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

subjects_to_do = np.arange(0,len(resultpath))

for sub in subjects_to_do:

    result_path = resultpath[sub]
    file_name = 'rest_epoch.fif'
    fwd_name = 'training1-volume_fwd.fif'
    path_file  = op.join(result_path,file_name) 
    fwd_fname = op.join(result_path,fwd_name) 
    

    fwd = mne.read_forward_solution(fwd_fname)
    
    epochs = mne.read_epochs(path_file)
    epochs = epochs.pick_types(meg='grad')
    # select the number of epochs [1 epoch = 10 second]
    nepochs = 12
    epochs = epochs[0: nepochs] 
    
    # epochs.resample(200)  
    # epochs = epochs.filter(8, 12).copy()
    # epochs = epochs.pick_types(meg='grad')
    #picks = mne.pick_types(epochs.info, meg='grad')
    
    
    rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative')
    
    # epochs_left  = epochs['left'].filter(50, 70).copy()
    # epochs_right = epochs['right'].filter(50, 70).copy()
    # left_cov = compute_covariance(epochs_left,method='shrunk',rank=rank,n_jobs = 4,verbose=True)
    # right_cov = compute_covariance(epochs_right,method='shrunk',rank=rank,n_jobs = 4,verbose=True)
    # common_cov = left_cov + right_cov 
    
    common_cov = compute_covariance(epochs,method='shrunk',rank=rank,n_jobs = 4,verbose=True)
    
    # common_cov.plot(epochs.info)
    
    filters = make_lcmv(epochs.info, 
                                fwd, 
                                common_cov, 
                                reg=0.05,
                                noise_cov=None, 
                                pick_ori='max-power') 
    
    
    stc = apply_lcmv_epochs(epochs,filters)
    
    
    src_fname = r'C:\mtbi\FLUX-main\dataRaw\MRI\training1-surface_src.fif'
    src_fs = mne.read_source_spaces(src_fname)
    # for subfile in range(0, 10):
    #     src = fwd['src']
    #     stc[subfile].plot(src=src, subject='training1', subjects_dir=r'C:\mtbi\FLUX-main\dataRaw\MRI', mode='stat_map');
   
    src = fwd['src']
    # stc[0].plot(src=src, subject='training1', subjects_dir=r'C:\mtbi\FLUX-main\dataRaw\MRI', mode='stat_map');
    
    # average in the epoch
    mean_stc = np.sum(stc) / len(stc) 
    mean_stc_data = mean_stc.data  
    con_stc_data = stc[0].data
    for nepo in range(1,nepochs):
        con_stc_data = np.concatenate((con_stc_data,stc[nepo].data),axis=1)
    
    # mean_stc_data = stc[0].data
    
    ## extract ROI time series
    template = np.array([[2.0,50,0],[42,34,16],[26,10,56],[18,42,40],[42,50,0],[26,50,24],[58,-22,8],[58,-46,0],[50,10,-24],
    [10,-94,24],[26,-94,8],[50,-70,8],[42,-78,-8],[58,-6,32],[42,-22,56],[10,-30,72],[18,-70,56],[34,-78,40],
    [-54,-46,40],[50,-70,16],[-6,-70,32],[2,-46,24],[2,-54,48],[-22,-62,56],[-38,-78,40],[58,-46,40],
    [-46,-70,16],[-54,-6,32],[-46,-22,56],[-6,-30,72],[-14,-94,24],[-22,-94,8],[-46,-70,8],[-38,-86,0],
    [-62,-22,8],[-62,-46,0],[-46,10,-24],[-46,34,16],[-22,10,56],[-14,42,48],[-38,50,0],[-22,58,16]])
    
    src_coordinates = src[0]['rr']*1000
    src_vertno = src[0]['vertno']
    src_roi_coordinates = src_coordinates[src_vertno,:]
    
    src_fs_coordinates = src_fs[0]['rr']*1000
    src_fs_vertno = src_fs[0]['vertno']
    src_fs_roi_coordinates = src_fs_coordinates[src_fs_vertno,:]
    
    selected_roi_distance = np.zeros((len(template),len(src_roi_coordinates)))
    roi_data = np.zeros((len(template),nepochs*10000))
    for nroi in range(0,len(template)):
        for i in range(0,len(src_roi_coordinates)):
            selected_roi_distance[nroi,i] = abs(np.linalg.norm(template[nroi]-src_roi_coordinates[i]))
        index = np.where(selected_roi_distance[nroi,:]<10)
        roi_data[nroi,:] = np.mean(con_stc_data[index[0],:],0)
  ######################
    X = np.transpose(roi_data)
    X = normalize(X)
    file_path = rf"{datadir}\subj{sub}.mat"
    scipy.io.savemat(file_path, {'X': X})
    

# # stc_rel.plot(src=fwd['src'], hemi = 'both' , views = 'parietal', surface = 'inflated',  subject='training1',
# #                         subjects_dir=r'C:\mtbi\FLUX-main\dataRaw\MRI');

# # stc_RvsL.plot(src=fwd['src'], hemi = 'both' , views = 'parietal', surface = 'inflated',  subject='training1',
# #                         subjects_dir=r'C:\mtbi\FLUX-main\dataRaw\MRI\');
# lims = [0.016, 0.02, 0.036]
# kwargs = dict(src=src, subject='training1', subjects_dir=r'C:\mtbi\FLUX-main\dataRaw\MRI',
#               initial_time=0.087, verbose=True)

# brain = stc.plot_3d(
#     clim=dict(kind='value', lims=lims), hemi='both', size=(600, 600),
#     views=['sagittal'],
#     # Could do this for a 3-panel figure:
#     # view_layout='horizontal', views=['coronal', 'sagittal', 'axial'],
#     brain_kwargs=dict(silhouette=True),
#     **kwargs)










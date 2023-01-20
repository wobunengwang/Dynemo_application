# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:36:52 2023

@author: daizhongpengMNE
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:52:52 2023

@author: daizhongpengMNE
"""
###############################################################################################################
import os.path as op
import mne

data_path = r'C:\MSc_Project\Data_Patient\sub006_s1_20220523_b588'
file_name = 'rest_eyes_open_1_raw.fif'

path_file  = op.join(data_path,file_name)

mri_dir = r'C:\mtbi\FLUX-main\dataRaw\MRI'

subject = 'training1'

Brain = mne.viz.get_brain_class()

brain = Brain(subject, 
                  hemi='lh', 
                  surf='pial.T1',
                  subjects_dir=mri_dir, 
                  size=(800, 600))

brain.add_annotation('aparc.a2009s', borders=False)


conductivity = (0.3,) 
model = mne.make_bem_model(subject,
                           ico=4,
                           conductivity=conductivity,
                           subjects_dir=mri_dir)
bem = mne.make_bem_solution(model)

fname_bem = 'training1-bem-sol.fif' 
mne.write_bem_solution(op.join(mri_dir,fname_bem),
                           bem,
                           overwrite=True)

# mne.viz.plot_bem(subject=subject,
#                 subjects_dir=mri_dir,
#                 brain_surfaces='white',
#                 orientation='coronal');



####################### Coregistration #######################
VALUE = r'C:\Users\daizhongpengMNE\Downloads'
mne.utils.set_config("SUBJECTS_DIR", VALUE, set_env=True)
mne.gui.coregistration()
####################### Coregistration #######################


trans = 'trans.fif'
# data_path = r'C:\mtbi\FLUX-main\dataRaw'
trans_file = op.join(data_path,trans)
print(trans_file)

info = mne.io.read_info(path_file)
# mne.viz.plot_alignment(info, trans_file, subject=subject, dig=True,
#                             meg=['helmet', 'sensors'], subjects_dir=mri_dir,
#                             surfaces='head-dense')



surface = op.join(mri_dir, subject, 'bem', 'inner_skull.surf')
#### select volume or surface
src = mne.setup_volume_source_space(subject, subjects_dir=mri_dir,
                                             surface=surface,
                                             verbose=True)

fname_src = 'training1-surface_src.fif' 
mne.write_source_spaces(op.join(mri_dir,fname_src),
                            src,
                            overwrite=True)




# mne.viz.plot_bem(subject=subject, 
#                      subjects_dir=mri_dir,
#                      brain_surfaces='white', 
#                      src=src, 
#                      orientation='coronal')


fwd = mne.make_forward_solution(path_file, 
                                    trans=trans_file, 
                                    src=src, 
                                    bem=bem,
                                    meg=True, eeg=False, 
                                    mindist=5.,  #TODO: minimum distance of sources from inner skull surface (in mm); can be 2.5
                                    n_jobs = 4, 
                                    verbose=True)


fname_fwd = op.join(mri_dir,
                        subject+'-volume_fwd.fif' )

mne.write_forward_solution(fname_fwd,
                               fwd,
                               overwrite=True)


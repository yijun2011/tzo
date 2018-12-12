#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:43:03 2018

@author: oscar seurat
"""
import numpy as np;
import Loader as ld;
import Combiner as cb;
import matplotlib.pyplot as plt;


# MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
# SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
# MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
# SCAN = 'sub-NC189_ses-20180825_acq-motion_run-02_T1w.nii'

MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/Coregistered/';
SCAN='aligned188ToNoMotionRun01.nii';

mri = ld.mri_scan(MRI_PATH + SCAN);    

x_start = 85;
x_end   = 105;
N = 1000;

# X_OFF = 10;
# Y_OFF = 10;
# W = 100;
# H = 100;

X_OFF = 0;
Y_OFF = 0;
W = 256;
H = 256;

grand_arr  = None;
grand_orig = None;

# WHAT_TO_GENERATE = 'distorted';
# WHAT_TO_GENERATE = 'original';
WHAT_TO_GENERATE = 'orig_corrupted';

for ig in range(x_start, x_end):
    img1 = mri.get_image(ig, axis='x', crop_center=[X_OFF,Y_OFF], crop_wh=[W,H]);
    if (WHAT_TO_GENERATE == 'distorted'):
        imarr = cb.Combiner.loop_example_transf1(img1, [31, 36, 1],
                                                 [2.51, 0],
                                                 [0,0],
                                                 [8.89, 0],
                                                 [0.18, 0.67, np.pi/7],
                                                 num_img = N);
        imarr1 = cb.Combiner.loop_example_transf1(img1, radii_t = [28.25, 31.25, 1], df_range = 5,
                                              phase_p=[2.4, 0],
                                              center=[0,0],
                                              ampl_p = [2, 0.05], amp_range = 4,
                                              sector_t = [0, 3, np.pi/18], sec_range=16,
                                              num_circ_min=5,
                                              num_circ_max=10,
                                              num_img = N);    
        imarr  = np.concatenate((imarr, imarr1), axis=2);
        
        if (ig == x_start):
            grand_arr = imarr;
        else:
            grand_arr = np.concatenate((grand_arr, imarr), axis=2);
        # No distinction in processing between 'original or 'orig_corrupted'   
    elif (WHAT_TO_GENERATE == 'original'): 
        chunk = np.zeros([W, H, 2*N]);
        IMG1  = img1.reshape([W, H, 1]);
        chunk = IMG1 + chunk; # This will copy img 2*N times!
        if (ig == x_start):
            grand_orig = chunk;
        else:
            grand_orig = np.concatenate((grand_orig, chunk), axis=2);
    elif (WHAT_TO_GENERATE == 'orig_corrupted'):
        IMG1  = img1.reshape([W, H, 1]);
        if (ig == x_start):
            grand_orig = IMG1;
        else:
            grand_orig = np.concatenate((grand_orig, IMG1), axis=2);        
    else:
        err_msg = 'generate_distorted_images: unsupported WHAT_TO_GENERATE option given.';
        raise ValueError(err_msg);

save_path = "/Users/yzhao11/Documents/Research/MachineLearning/ArtifactsData/";
if (WHAT_TO_GENERATE == 'distorted'):
    save_file = "DistortedImages_188_X2";
    np.savez_compressed(save_path + save_file + '.npz', grand_arr_x = grand_arr);
elif (WHAT_TO_GENERATE == 'original'):
    save_file = "OriginalImages_188_X2";    
    np.savez_compressed(save_path + save_file + '.npz', grand_orig_x = grand_orig);
elif (WHAT_TO_GENERATE == 'orig_corrupted'):
    save_file = "CorruprtedImages_188_X2";    
    np.savez_compressed(save_path + save_file + '.npz', grand_corr_x = grand_orig);  
else:
    err_msg = 'generate_distorted_images: unsupported WHAT_TO_GENERATE option given.';
    raise ValueError(err_msg);   
    

        
  
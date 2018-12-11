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


MRI_PATH='/scratch/ossowskj/ArtifactsData/';
SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';

mri = ld.mri_scan(MRI_PATH + SCAN);    

x_start = 80;
x_end   = 90;
#N = 4000;
N = 50;

X_OFF = 0;
Y_OFF = 0;
W = 256;
H = 256;

grand_arr  = None;
grand_orig = None;

# WHAT_TO_GENERATE = 'distorted';
WHAT_TO_GENERATE = 'original';

for ig in range(x_start, x_end):
    img1 = mri.get_image(ig, axis='x'); #, crop_center=[X_OFF,Y_OFF], crop_wh=[W,H]);
    print('processing image number ' + str(ig));

    if (WHAT_TO_GENERATE == 'distorted'):
      #  imarr = cb.Combiner.loop_example_transf1(img1, [31, 36, 1], [2.51, 0], [0,0], [8.89, 0], [0.18, 0.67, np.pi/7],
      #                                           num_circ_min = 1,
      #                                           num_circ_max = 10,
      #                                           num_img=N);

        imarr = cb.Combiner.loop_example_transf1(img1, [61, 66, 1.5], [2.51, 0], [0,0], [8.89, 0], [0.18, 0.67, np.pi/7],
                                              f_min=0.06, f_max=0.18, num_img = N);
        if (ig == x_start):
            grand_arr = imarr;
        else:
            grand_arr = np.concatenate((grand_arr, imarr), axis=2);
    else:
        chunk = np.zeros([W, H, N]);
        IMG1  = img1.reshape([W, H, 1]);
        chunk = IMG1 + chunk; # This will copy img N times!
        if (ig == x_start):
            grand_orig = chunk;
        else:
            grand_orig = np.concatenate((grand_orig, chunk), axis=2);        

save_path = "/scratch/ossowskj/ArtifactsData/";
if (WHAT_TO_GENERATE == 'distorted'):
    save_file = "DistortedImages_188_X_Full_Scaled_50";
    np.savez_compressed(save_path + save_file + '.npz', grand_arr_x = grand_arr);
else:
    save_file = "OriginalImages_188_X_Full_Scaled_50";    
    np.savez_compressed(save_path + save_file + '.npz', grand_orig_x = grand_orig);

        
  

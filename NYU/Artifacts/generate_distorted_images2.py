#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:43:03 2018

@author: oscar seurat
"""
import sys
import os
import numpy as np;
import Loader as ld;
import Combiner as cb;
import matplotlib.pyplot as plt;


"""
Input:
    img: 2D numpy array representing the MRI image.
    height: Height to pad image to.
    width: Width to pad image to.
Output:
    Padded image, with each padded pixel equal in value to the top left corner
    pixel.
"""
def pad_image(img,height,width):

    horizontal_space = width - len(img[0]);
    vertical_space = height - len(img);

    color = img[0,0];

    pad_left = int(horizontal_space / 2)
    pad_right = pad_left;
    pad_top = int(vertical_space / 2);
    pad_bottom = pad_top;

    if (horizontal_space % 2 != 0):
        pad_right += 1
    if (vertical_space % 2 != 0):
        pad_bottom += 1

    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode = 'constant', constant_values = (color));


# Patient 188
# MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
# MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/Coregistered/';
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/'
# SCAN='sub-NC189_ses-20180825_acq-motion_run-02_T1w.nii';

os.chdir("../Patient_Scans")
path = os.getcwd()


#MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
# SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
# SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-02_T1w.nii';
#SCAN = 'aligned188ToNoMotionRun01.nii';
# SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';

# Patient 189
# MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
# SCAN = 'sub-NC189_ses-20180825_acq-motion_run-02_T1w.nii';
# SCAN = 'sub-NC189_ses-20180825_acq-motion_run-01_T1w.nii';

# Patient 225
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC225/ses-20180802/anat/';
# SCAN = 'sub-NC225_ses-20180802_acq-motion_run-01_T1w.nii'
# SCAN = 'sub-NC225_ses-20180802_acq-motion_run-02_T1w.nii'

# Patient 227
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC227/ses-20180830/anat/';
# SCAN = 'sub-NC227_ses-20180830_acq-motion_run-01_T1w.nii'
# SCAN = 'sub-NC227_ses-20180830_acq-motion_run-02_T1w.nii'

# Patient 183
# MRI_PATH='/Users/jossowski/Desktop/Fordham/NYU/zhao_dataset_20181011/sub-NC183/ses-20180825/anat/';
# SCAN = 'sub-NC183_ses-20180825_acq-motion_run-01_T1w.nii'
# SCAN = 'sub-NC183_ses-20180825_acq-motion_run-02_T1w.nii'

#path = "/Users/Tim/Desktop/Code/Machine_Learning/TZO/NYU/Patients/"

if __name__ == '__main__':
    args = sys.argv[1:]
    if (len(args) != 2):
        print("Usage: python generate_distorted_images2.py <patient> <orientation(s)>")
        exit(1)
    patient = args[0]
    orientations = args[1].lower()

    if len(orientations) > 3:
        print("Orientation is a string of length at most 3. Ex: \"xyz\"")
        exit(1)


    mri = ld.mri_scan(path + "/" + patient + "/mprage.nii");


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

    for orientation in orientations:
        start = 0;
        end = mri.x_max;
        if (orientation == 'x'):
            end = mri.x_max
        elif (orientation == 'y'):
            end = mri.y_max
        elif (orientation == 'z'):
            end = mri.z_max
        else:
            print("Orientation must be x, y, or z")
            exit(1)

        for ig in range(start, end):
            img1 = pad_image(mri.get_image(ig, axis=orientation, crop_center=[X_OFF,Y_OFF], crop_wh=[W,H]),W,H);
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

                if (ig == start):
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
                if (ig == start):
                    grand_orig = IMG1;
                else:
                    grand_orig = np.concatenate((grand_orig, IMG1), axis=2);
            else:
                err_msg = 'generate_distorted_images: unsupported WHAT_TO_GENERATE option given.';
                raise ValueError(err_msg);

        # save_path = "/Users/yzhao11/Documents/Research/MachineLearning/ArtifactsData/";
        save_path = "Users/Tim/Desktop/Code/Machine_Learning/TZO/NYU/ArtifactsData/";
        if (WHAT_TO_GENERATE == 'distorted'):
            save_file = "DistortedImages_188_X2";
            np.savez_compressed(save_path + save_file + '.npz', grand_arr_x = grand_arr);
        elif (WHAT_TO_GENERATE == 'original'):
            save_file = "OriginalImages_188_X2";
            np.savez_compressed(save_path + save_file + '.npz', grand_orig_x = grand_orig);
        elif (WHAT_TO_GENERATE == 'orig_corrupted'):
            # save_file = "CorruprtedImages_189_X20-130";
            save_file = patient + "_" + orientation.upper();
            np.savez_compressed(path + "/" + patient + "/" + save_file + '.npz', grand_corr_x = grand_orig);
        else:
            err_msg = 'generate_distorted_images: unsupported WHAT_TO_GENERATE option given.';
            raise ValueError(err_msg);

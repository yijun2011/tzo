#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:45:11 2018

@author: oscar seurat
"""
import Loader as ld;
import numpy as np;
import sys;
import matplotlib.pyplot as plt;
import Transformer as tf;
from scipy import stats;

class CompareNii:
    """
    mri0, mri1 are numpy arrays
    """
    def __init__(self, mri0, mri1):
        self.mri0 = mri0;
        self.mri1 = mri1;
        
        self.max_x, self.max_y, self.max_z = self.mri0.img.dataobj.shape;
        if ((self.max_x, self.max_y, self.max_z) != self.mri1.img.dataobj.shape):
            print('CompareNii.CompareNii: the input files do not have the same ' +
                  'sizes (', sys.stderr);
            sys.exit(-1);
            
    def calc_diff(self, img_no, axis, normalize = False):
        img0 = self.mri0.get_image(img_no, axis, rotate=0);
        img1 = self.mri1.get_image(img_no, axis, rotate=0);
        
        if (normalize):
            M0 = np.amax(img0);
            img0 = img0/M0;
            M1 = np.amax(img1);
            img1 = img1/M1;
        
        diff = np.sqrt(np.sum(img0-img1)**2);
        return diff;
    
    def _calc_diff_ft(self, img_no, axis, normalize=False):
        img0 = self.mri0.get_image(img_no, axis, rotate=0);
        img1 = self.mri1.get_image(img_no, axis, rotate=0);
        
        if (normalize):
            M0 = np.amax(img0);
            img0 = img0/M0;
            M1 = np.amax(img1);
            img1 = img1/M1;
        
        T0 = tf.Transformer(img0);
        T1 = tf.Transformer(img1);
        f0 = T0.get_ft_raw();
        f1 = T1.get_ft_raw();
        
        diff = np.sum(np.abs(f0 - f1));
        return diff;
        
    def _calc_diff_ft_series(self, axis, normalize=True, zscore=False,
                            start=0.0, end=1.0):
        size = self.max_x;
        
        if (axis == 'y'):
            size = self.max_y;
        elif (axis == 'z'):
            size = self.max_z;
            
        result = np.zeros(size);
        for s in range(int(start*size), int(end*size)):
                result[s]=self._calc_diff_ft(s, axis, normalize);
                
        if (zscore):
            result = result[~np.isnan(result)];
            result = stats.zscore(result);
            
        return result;       
       
    def calc_diff_series(self, axis, normalize = False, zscore = False,
                         start=0.0, end=1.0, mode = 'Spatial'):
        if (mode == 'Spatial'):
            size = self.max_x;
            
            if (axis == 'y'):
                size = self.max_y;
            elif (axis == 'z'):
                size = self.max_z;
                
            result = np.zeros(size);
            for s in range(int(start*size), int(end*size)):
                    result[s]=self.calc_diff(s, axis);
                    
            if (zscore):
                result = result[~np.isnan(result)];
                result = stats.zscore(result);
            return result;
        
        return self._calc_diff_ft_series(axis, normalize, zscore,
                            start, end);

#####################################################################
#####################################################################

# MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/Coregistered/';
# SCAN0    = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
# SCAN0   = 'sub-NC188_ses-20180825_acq-nomotion_run-02_T1w.nii';
# SCAN1   = 'sub-NC188_ses-20180825_acq-nomotion_run-02_T1w.nii';

#SCAN1    = 'aligned188ToNoMotionRun01.nii'

SCAN0 = 'sub-NC188_ses-20180825_acq-nomotion_run-02_T1w.nii'
SCAN1 = 'NoMotion01aligned188ToNoMotionRun02.nii'

mri0 = ld.mri_scan(MRI_PATH + SCAN0);
mri1 = ld.mri_scan(MRI_PATH + SCAN1);

cN = CompareNii(mri0, mri1);


"""
diff = cN.calc_diff_ft(100, 'z', normalize=True);
print(diff);
"""


# fig = plt.figure(figsize=(10,7));

fig, ax = plt.subplots(1, 3, figsize=(22,13), sharey=True);
zscore = False
start  = 0.1;
end    = 0.9;
mode   = 'Spatial';


axis = 'x'
dS = cN.calc_diff_series(axis=axis, zscore=zscore, start=start, end=end, mode = mode);
ax[0].plot(dS);
ax[0].set_title(mode + ' Differences in ' + axis + ' between' + '\n' + SCAN0 + '\n' + SCAN1);

axis = 'y'
dS = cN.calc_diff_series(axis=axis, zscore=zscore, start=start, end=end, mode = mode);
ax[1].plot(dS);
ax[1].set_title(mode + ' Differences in ' + axis + ' between' + '\n' + SCAN0 + '\n' + SCAN1);

axis = 'z'
dS = cN.calc_diff_series(axis=axis, zscore=zscore, start=start, end=end, mode = mode);
ax[2].plot(dS);
ax[2].set_title(mode + ' Differences in ' + axis + ' between' + '\n' + SCAN0 + '\n' + SCAN1);

plt.show();





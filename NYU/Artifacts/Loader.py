#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function
import nibabel as nib;

import matplotlib.pyplot as plt;
import numpy as np;
import sys;

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

class mri_scan:
    def __init__(self, path):
        self.W_MAX = 10000;
        self.H_MAX = 10000;
        self.path = path;
        self.img  = nib.load(path);
        self.x_max, self.y_max, self.z_max = self.img.shape;
        
    def dim(self):
        return [ self.z_max, self.y_max, self.x_max];
    
    @staticmethod
    def _rotate_90_ccw(img):
        [n, m] = img.shape;
        out_img = np.zeros((m, n));
        for x in range(m):
            for y in range(n):
                out_img[x, y]=img[y,-x];
        return out_img;
    
    @staticmethod
    def _rotate_180(img):
        [n, m] = img.shape;
        out_img = np.zeros((m, n));
        for x in range(m):
            for y in range(n):
                out_img[x, y]=img[-y,-x];
        return out_img;
    
    def _validate_slice(self, slice_no, axis='z'):
        if (axis != 'x' and axis != 'y' and axis != 'z' ):
            print('mri_scan._validate_slice: incorrect axis given.',
                  file=sys.stderr);
            sys.exit(-1);
            
        if (axis == 'x' and (slice_no < 0 or slice_no > self.x_max )):
            print('mri_scan._validate_slice: incorrect slice number given. x_max is ',
                  + self.x_max, sys.stderr);
            sys.exit(-1);
            
        if (axis == 'y' and (slice_no < 0 or slice_no > self.y_max )):
            print('mri_scan._validate_slice: incorrect slice number given. y_max is ',
                  + self.y_max, sys.stderr);
            sys.exit(-1);
            
        if (axis == 'z' and (slice_no < 0 or slice_no > self.z_max )):
            print('mri_scan.validate_slice: incorrect slice number given. z_max is ',
                  + self.z_max, sys.stderr);
            sys.exit(-1);
            
    """ In addition to retrieving the image, the below function also crops it
        according to the "later" arguments; the default is no cropping """
     
    def get_image(self, slice_no, axis='z', rotate = 1,
                  crop_center = (0,0), 
                  crop_wh = (100000, 100000)):
        self._validate_slice(slice_no, axis);

        if (axis == 'x'):
            img = self.img.dataobj[slice_no,:,:];
        elif (axis == 'y'):
            img = self.img.dataobj[:,slice_no,:];
        else:
            img = self.img.dataobj[:,:,slice_no];
        
        if (rotate == 1):
            img = self._rotate_90_ccw(img);
            
        w, h = img.shape;
        w = np.amin([crop_wh[0], w]);
        h = np.amin([crop_wh[1], h]);
        
        x = crop_center[0];
        y = crop_center[1];
        
        img = img[x:x+w, y:y+h];
        
        return img;
        
    def show_slice(self, slice_no, axis='z', rotate = 1):
        img = self.get_image(slice_no, axis, rotate);
        fig = plt.figure(figsize=(10, 10));   
        plt.imshow(img);
        plt.show();

if 0:
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-motion_run-01_T1w.nii';
    
    mri = mri_scan(MRI_PATH + SCAN);
    print(mri.dim());
    mri.show_slice(100, 'y', rotate=1);

    
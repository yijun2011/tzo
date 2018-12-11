#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:02:24 2018

@author: oscar seurat
"""

import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib.widgets import TextBox;
import sys;

class verifier:
    def __init__(self, original, distorted):
        if (original.shape != distorted.shape):
            print('verifier.verifier: the original and distorted image arrays ' + 
                  'must have the same dimensions.', sys.stderr);
            sys.exit(-1);
        
        self.orig_arr = original;
        self.dist_arr = distorted;
        self.total_images = original.shape[2];
        
        # Let's set up image display gear
        self.fig, self.ax = plt.subplots(1, 2, figsize = (10, 7));
        plt.subplots_adjust(bottom=0.3);
        
        self.current_slice = 1000;
        
        self.ax[0].imshow(self.orig_arr[:,:, self.current_slice]);
        self.ax[1].imshow(self.dist_arr[:,:, self.current_slice]);
        
        img_no_pos    = plt.axes([0.48, 0.1, 0.06, 0.1]); # x, y, w, h
        self.img_tbox = TextBox(img_no_pos, 'Image No.: ', initial=str(self.current_slice));
    
    
    def _update(self, val):
        self.current_slice = int(val);
        self.ax[0].imshow(self.orig_arr[:,:, self.current_slice]);
        self.ax[1].imshow(self.dist_arr[:,:, self.current_slice]);
        self.fig.canvas.draw_idle();
        
    def initialize(self):
        self.img_tbox.on_submit(self._update);
        
###################################################################
###################################################################
        
MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
        
orig_file = MRI_PATH + 'OriginalImages_188_X.npz';
mngl_file = MRI_PATH + 'DistortedImages_188_X.npz';

# orig_file = MRI_PATH + 'orig_test.npz';
# mngl_file = MRI_PATH + 'mngl_test.npz';

ORIG = np.load(orig_file);
orig_arr = ORIG['grand_orig_x'];

MNGL = np.load(mngl_file);
mngl_arr = MNGL['grand_arr_x'];

VRF = verifier(orig_arr, mngl_arr);
VRF.initialize();
plt.show();


        
        
        
        
        


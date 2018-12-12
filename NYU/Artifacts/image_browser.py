#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:02:52 2018

@author: oscar seurat
"""

import matplotlib.pyplot as plt;
from keras.models import Model;
from keras.models import load_model

from matplotlib.widgets import TextBox;
import numpy as np;
import Transformer as tr;
import cloudpickle as pickle;
import sys;

"""
This class is meant to take in a set of images stored as a numpy array (usually
that array is gotten from a .npz file

x_test    the numpy array to display in matplotlib
"""
class image_browser:
    def __init__(self, x_test):

        MX = np.amax(np.abs(x_test));
        self.x_test = x_test/MX;
        
        self.idxB = 0;
        self.idxE = x_test.shape[0];
        
        self.current_slice = 0;
            
        p = self.current_slice;
        
        # Let's set up image display gear
        self.fig, self.ax  = plt.subplots(1, 1, figsize = (10, 8));
        self.fig.tight_layout();
        plt.subplots_adjust(bottom=0.20, top=0.9);
        
        self.ax.imshow(self.x_test[self.current_slice, :, :, 0]);
        self.ax.set_title('Image No: ' + str(self.current_slice));
       
        
        
        img_no_pos = plt.axes([0.48, 0.08, 0.04, 0.04]); # x, y, w, h
        label      = 'Image No.:   \nOut of ' + str(self.idxE) + '  ';
        self.img_tbox = TextBox(img_no_pos, label, initial=str(self.current_slice));
       
    def _update(self, val):
        p = int(val);
        self.current_slice = p;
        
        image_x = self.x_test[self.current_slice,:,:, 0];
        self.ax.imshow(image_x);
        
    def initialize(self):
        self.img_tbox.on_submit(self._update);
        
###################################################################
###################################################################

if 1:
    FILE_DIR = '/Users/yzhao11/Documents/Research/MachineLearning/ArtifactsData/';
    FILE_NAME    ='OriginalImages_188_X_Full_Scaled_50.npz';
    NORMALIZED    = False;

    x_file = FILE_DIR + FILE_NAME;
    
    MNGL = np.load(x_file);
    x_test = MNGL['grand_orig_x'];
    
    x_test= np.transpose(x_test, [2,0,1]);
    
    a, b, c = x_test.shape;
    x_test  = np.reshape( x_test, [a, b, c, 1]);
    
    iB = image_browser(x_test);                
    iB.initialize();
    plt.show();
    
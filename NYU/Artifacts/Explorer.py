#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:33:00 2018

@author: oscar seurat
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import Loader as ld;

class explorer:
    
    def __init__(self, scan):
        self.scan = scan;
        self.fig, self.ax = plt.subplots(figsize=(12,12));
        plt.subplots_adjust(left=0.10, bottom=0.27);
        
        x_max = scan.x_max; y_max = scan.y_max; z_max = scan.z_max;
        x_init = x_max/2; y_init=y_max/2; z_init=z_max/2;
        
        self.image = scan.get_image(z_init);
        self.ax.imshow(self.image);
        self.ax.set_title('x = '+', '
                          'y = '+', '
                          'z = '+str(z_init));
                          
        axcolor = 'lightgoldenrodyellow'
        x_slider_pos = plt.axes([0.18, 0.07, 0.65, 0.03], facecolor=axcolor);
        y_slider_pos = plt.axes([0.18, 0.12, 0.65, 0.03], facecolor=axcolor);        
        z_slider_pos = plt.axes([0.18, 0.17, 0.65, 0.03], facecolor=axcolor);        
         
        self.x_slider = Slider(x_slider_pos, 'X', 0, x_max, valinit=x_init);
        self.y_slider = Slider(y_slider_pos, 'Y', 0, y_max, valinit=y_init);       
        self.z_slider = Slider(z_slider_pos, 'Z', 0, z_max, valinit=z_init); 

    def _update_x(self, val):
        if (val == 0):
            return;
        x = val;
        self.y_slider.set_val(0);
        self.z_slider.set_val(0);
        self.image = self.scan.get_image(int(x), 'x');
        self.ax.imshow(self.image);
        self.ax.set_title('x = '+str(int(x)) + ', '
                          'y = '+', '
                          'z = ');
        self.fig.canvas.draw_idle();
        
    def _update_y(self, val):
        if (val == 0):
            return;
        y = val;
        self.x_slider.set_val(0);
        self.z_slider.set_val(0);
        self.image = self.scan.get_image(int(y), 'y');
        self.ax.imshow(self.image);
        self.ax.set_title('x = ' + ', '
                          'y = '+str(int(y))+', '
                          'z = ');
        self.fig.canvas.draw_idle();      
                         
    def _update_z(self, val):
        if (val == 0):
            return;
        z = val;
        self.x_slider.set_val(0);
        self.y_slider.set_val(0);
        self.image = self.scan.get_image(int(z), 'z');
        self.ax.imshow(self.image);
        self.ax.set_title('x = ' + ', '
                          'y = '+', '
                          'z = '+str(int(z)));       
        self.fig.canvas.draw_idle();

    def initialize(self):
        self.x_slider.on_changed(self._update_x);                      
        self.y_slider.on_changed(self._update_y);
        self.z_slider.on_changed(self._update_z);
    
if 1:
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/Coregistered/';
    # SCAN = 'aligned188ToNoMotionRun01.nii'
    
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
    
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
    # SCAN = 'sub-NC189_ses-20180825_acq-motion_run-01_T1w.nii'
    
        
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    # SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii'
    
         
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC183/ses-20180825/anat/';
    # SCAN = 'sub-NC183_ses-20180825_acq-nomotion_run-02_T1w.nii'   

    mri = ld.mri_scan(MRI_PATH + SCAN);
    exp = explorer(mri);
    exp.initialize();
    plt.show();

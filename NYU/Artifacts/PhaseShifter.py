#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:50:36 2018

@author: oscar seurat
"""

import Loader as ld;
import Transformer;
import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib.widgets import Slider, Button, RadioButtons;

class phase_shifter:
    def __init__(self, image):
        # image is simply an np.array
        self.image = image;
        self.trm   = Transformer.Transformer(self.image);
        
        self.size_x, self.size_y = image.shape;
        self.safe_val = np.amin([self.size_x, self.size_y])/2;
        self.max_val  = np.amax([self.size_x, self.size_y])/2;
        
        self.inner_r     = self.safe_val/4;
        self.outer_r     = self.safe_val/3;
        self.phase_shift = np.pi/3;
        self.center_x    = 0;
        self.center_y    = 0;
        self.ampl        = 1;
        self.sector_a1   = 0;
        self.sector_a2   = 2*np.pi;
        
        self.trm_img, self.flt_spec = self.trm.example_transf1(self.inner_r,
                                                               self.outer_r,
                                                               self.phase_shift,
                                                               [self.center_x,
                                                               self.center_y],
                                                               self.ampl,
                                                               [self.sector_a1,
                                                               self.sector_a2]);       
#        
#        # The image-transforming function
#        self.t_func = lambda ft: Transformer.Transformer.ring_const_phase_shift(ft,
#                                                                    self.inner_r,
#                                                                    self.outer_r,
#                                                                    self.phase_shift,
#                                                                    [self.center_x,
#                                                                     self.center_y],
#                                                                    self.ampl,
#                                                                    [self.sector_a1,
#                                                                     self.sector_a2]);
#        # self.sp_func is a spatial transform -- in this case we want to have
#        # the histogram of the original image                                                                       
#        self.sp_func = lambda trm: Transformer.Transformer.hist_match(trm, self.image);
#        
#        # self.flt_spec is the Fouerier spectrum of the  of the applied Fourier
#        # filter
#        self.flt_spec = self.trm.get_filter_spectrum(self.t_func);
#        
#        # We apply the Fourier filter first thand the spatial transform to 
#        # get the original histogram
#        trm_tmp_img = self.trm.apply_ft_filter(self.t_func);
#        self.trm_img = Transformer.Transformer.apply_sp_filter(trm_tmp_img, self.sp_func);                                
        
        self.fig, self.ax = plt.subplots(1, 3, figsize=(18,13));
        plt.subplots_adjust(left=0.10, bottom=0.36);

        self.ax[0].imshow(self.image);
        self.ax[1].imshow(self.flt_spec);
        self.ax[2].imshow(self.trm_img);

        axcolor = 'lightgoldenrodyellow'
        inner_r_pos  = plt.axes([0.18, 0.27, 0.65, 0.03], facecolor=axcolor);
        outer_r_pos  = plt.axes([0.18, 0.22, 0.65, 0.03], facecolor=axcolor);
        
        ph_shft_pos  = plt.axes([0.18, 0.17, 0.28,  0.03], facecolor=axcolor);
        ampl_pos     = plt.axes([0.55, 0.17, 0.28,  0.03], facecolor=axcolor);
        
        center_x_pos  = plt.axes([0.18, 0.12, 0.28, 0.03], facecolor=axcolor);
        center_y_pos  = plt.axes([0.55, 0.12, 0.28, 0.03], facecolor=axcolor); 
        
        sector_a1_pos  = plt.axes([0.18, 0.07, 0.28, 0.03], facecolor=axcolor);
        sector_a2_pos  = plt.axes([0.55, 0.07, 0.28, 0.03], facecolor=axcolor);

        self.inner_r_slider  = Slider(inner_r_pos, 'Inner R',  0, self.safe_val, valinit=self.inner_r);
        self.outer_r_slider  = Slider(outer_r_pos, 'Outer R',  0, self.safe_val,  valinit=self.outer_r);       
        self.ph_shift_slider = Slider(ph_shft_pos, 'P. Shift', 0, 2*np.pi,       valinit=0);
        self.ampl_slider     = Slider(ampl_pos,    'Amplif.',  0, 16,            valinit=1);
        self.center_x_slider = Slider(center_x_pos,'Center X', 0, self.safe_val, valinit=0);
        self.center_y_slider = Slider(center_y_pos,'Center Y', 0, self.safe_val, valinit=0);
        self.sector_a1_sider = Slider(sector_a1_pos,'Sec. A1', 0, 2*np.pi,       valinit=0);
        self.sector_a2_sider = Slider(sector_a2_pos,'Sec. A2', 0, 2*np.pi,       valinit=2*np.pi); 
        
    
    def _update_common(self):
#        self.t_func = lambda ft: Transformer.Transformer.ring_const_phase_shift(ft,
#                                                    self.inner_r,
#                                                    self.outer_r,
#                                                    self.phase_shift,
#                                                    [self.center_x,
#                                                     self.center_y],
#                                                    self.ampl,
#                                                    [self.sector_a1,
#                                                     self.sector_a2]);
#        self.flt_spec = self.trm.get_filter_spectrum(self.t_func);
#   
#        # Apply the Fourier transform filter and then match the 
#        # histogram of the result to the original image
#        trm_tmp_img = self.trm.apply_ft_filter(self.t_func); 
#        self.trm_img = Transformer.Transformer.apply_sp_filter(trm_tmp_img,
#                                                               self.sp_func);

        self.trm_img, self.flt_spec = self.trm.example_transf1(self.inner_r,
                                                               self.outer_r,
                                                               self.phase_shift,
                                                               [self.center_x,
                                                               self.center_y],
                                                               self.ampl,
                                                               [self.sector_a1,
                                                               self.sector_a2]);        
        self.ax[0].imshow(self.image);
        self.ax[1].imshow(self.flt_spec);
        self.ax[2].imshow(self.trm_img);
        self.fig.canvas.draw_idle();
        
    def _update_inner_r(self, val):
        self.inner_r = val;
        self._update_common();
        
    def _update_outer_r(self, val):
        self.outer_r = val;
        self._update_common(); 
        
    def _update_phase_shift(self, val):
        self.phase_shift = val;
        self._update_common();  
        
    def _update_center_x(self, val):
        self.center_x = val;
        self._update_common();   
        
    def _update_center_y(self, val):
        self.center_y = val;
        self._update_common(); 
        
    def _update_ampl(self, val):
        self.ampl = val;
        self._update_common(); 
        
    def _update_sector_a1(self, val):
        self.sector_a1 = val;
        self._update_common();   
        
    def _update_sector_a2(self, val):
        self.sector_a2 = val;
        self._update_common();     
        
    def initialize(self):
        self.inner_r_slider.on_changed(self._update_inner_r);
        self.outer_r_slider.on_changed(self._update_outer_r);
        self.ph_shift_slider.on_changed(self._update_phase_shift);
        self.ampl_slider.on_changed(self._update_ampl);
        self.center_x_slider.on_changed(self._update_center_x);
        self.center_y_slider.on_changed(self._update_center_y);
        self.sector_a1_sider.on_changed(self._update_sector_a1);
        self.sector_a2_sider.on_changed(self._update_sector_a2);        
      
###################################################################
###################################################################
        
MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';

mri = ld.mri_scan(MRI_PATH + SCAN);       
img = mri.get_image(100);

psh = phase_shifter(img);
psh.initialize();
plt.show();

   
        
        
        
        
        
        
        
        
        
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:13:10 2018

@author: oscar seurat
"""

from scipy import signal;
from scipy import misc;
import matplotlib.pyplot as plt;
import numpy as np;
import sys;
from random import randint;
from skimage.util import view_as_blocks;
import matplotlib.pyplot as plt;
import Loader as ld;
import Transformer as td;
from matplotlib.widgets import Slider;

import Convolver as cv;

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
img = mri.get_image(107, 'x');

cV = cv.convolver(img, 16); # kernel size
#convolutions = cV._generate_random_convolutions(5, 16);
convolutions = cV._generate_random_lconv(6, 9);

cV.display(convolutions, 0, 9, show_kernel=True); # start index, number of images


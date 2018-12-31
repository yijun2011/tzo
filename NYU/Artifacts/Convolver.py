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

"""
The idea for this class is to apply lots of convolutions to an image and see
what happens.
"""


class convolver:
    TUNING_LIMIT = 6;
    def __init__(self, image, size):
        if (size < 2):
            print ('convolver.convolver: the size of the kernel should be >= 2.',
                   sys.stderr);
            sys.exit(-1);
            
        self.img_     = image;
        self.size_    = size;
    
    """
    This function generates random convolutions and outputs them in a 3d array
    """
        
    def _generate_random_convolutions(self, non_zeros, num_conv):
        non_zeros = int(non_zeros);
        if (non_zeros <= 0):
            print('convolver.convolver: the number of non zero elements must be > 0.',
                  sys.stderr);
            return [];
        # Create an array of num_conv images
        conv_arr = np.zeros([num_conv, self.size_, self.size_])
        for s in range(num_conv):
            # Generate coordinates of num_conv 1's in the
            # kernel
            for u in range(min(non_zeros, self.size_**2)):
                x = randint(0, self.size_ - 1);
                y = randint(0, self.size_ - 1);
                conv_arr[s, x, y] = 1;
            # Divide by acual sum of non_zeros
            SS = np.sum(conv_arr[s,:,:]);
            print ('Dividing by '+ str(SS))
            conv_arr[s, :, :] = conv_arr[s,:,:] * (1.0/float(SS));
        
        return conv_arr;
    
    """
    This function generates perturbed linear convolutions (by Gaussian noise)
    """
    
    def _generate_random_lconv(self, non_zeros, num_conv, max_a = 3 , sigma = 0.5): 
        conv_arr = np.zeros([num_conv, self.size_, self.size_])
        for s in range(num_conv):
            # Choose the direction
            a = max_a*np.random.uniform(-1, 1);
            for u in range(min(non_zeros, self.size_**2)):
                epsilon = np.random.normal(0, sigma)
                x = randint(0, self.size_ - 1);
                y = int(a*(x - 0.5*self.size_) + 0.5*self.size_ + epsilon);
                while (y >= self.size_):
                    x = int((x-0.5*self.size_)/2 + 0.5*self.size_);
                    y = int(a*(x - 0.5*self.size_) + 0.5*self.size_);
            
                conv_arr[s, x, y] = 1;
            # Divide by acual sum of non_zeros
            SS = np.sum(conv_arr[s,:,:]);
            print ('Dividing by '+ str(SS))
            conv_arr[s, :, :] = conv_arr[s,:,:] * (1.0/float(SS));
        
        return conv_arr;       
        
    """
    Display convolutions in a grid
    """
    def display(self, conv_arr, start_idx, num, show_kernel = False):
        
        total_num_img, x, y = conv_arr.shape;
        if (start_idx + num > total_num_img):
            print('convolver.display: the staring image and the number of images reach past the end of the array.',
                  sys.stderr);
            return;
            
        if (not num in [1, 4, 9, 16, 25]):
            print('convolver.display: the number of images to display must be one of the numbers 1, 3, 9, 16, 25',
                  sys.stderr);
            return;
    
        D = int(np.sqrt(num));
        Q = 0;
        
        if (show_kernel):
            Q = 1; 
        
        fig, axarr = plt.subplots(D, (Q+1)*D, figsize=(12, 12));
        for q in range(D):
            for r in range(D):
                conv = conv_arr[q*D + r, :, :];
                print('Index: (' + str(q) + ', ' + str(r) + '). Convolving with the following kernel');
                print (repr(conv));
                
                img_conv = signal.convolve2d(self.img_, conv, boundary = 'symm', mode='same');
                if (show_kernel):
                    axarr[q, Q*(r+1) + r - 1 ].imshow(conv);
                    
                axarr[q, Q*(r+1) + r].imshow(img_conv);
        plt.tight_layout();
        plt.show();
        
        
    """
    This funcion  applies a single kernel to the internal image and then displays
    the kernel, the original image, the convolved image, and the the 
    log spectrum of the convolved image.
    """
    def display_single(self, kernel):
         img_conv = signal.convolve2d(self.img_, kernel, boundary = 'symm', mode='same'); 
         tF = td.Transformer(self.img_);
         img_frr  = tF.get_ft_spectrum();
         fig, axarr = plt.subplots(1, 4, figsize = (15, 5));
         axarr[0].imshow(kernel);
         axarr[1].imshow(img_conv);
         axarr[2].imshow(img_frr);
         axarr[3].imshow(img);
         plt.tight_layout();
         plt.show();
         
    """
    tuner taks the given kernel with at most 5 non-zero elements an allows to
    interactively change the values of those entries (usually the initial non-zeros
    are equal to 1). The so re-weighted matrix is then convolved with the internal
    image (the image which was given to the constructor).
    """
    def tuner(self, kernel):
         m, n = kernel.shape;
         
         if (m != self.size_ or n != self.size_):
             print('convolver.tuner: the kernel must be of size (' + str(self.size_) + ', ' +  str(self.size_) + ').',
                   sys.stderr);
             return;
             
         F = (kernel != 0);
         chk = np.vectorize(lambda x: int(x));
         non_zero = np.sum(chk(F));
         if (non_zero > convolver.TUNING_LIMIT):
             print('convolver.tuner: only kernels with at most ' + str(convolver.TUNING_LIMIT) + ' are allowed.',
                   sys.stderr);
             return;
             
         self.my_kernel = kernel;  
         # Find positions of the non zeros
         self.non_zeros = [];
         for xx in np.arange(self.size_):
             for yy in np.arange(self.size_):
                 if (self.my_kernel[xx,yy] != 0):
                     self.non_zeros.append(np.array([xx,yy]));

         self.img_conv = signal.convolve2d(self.img_, kernel, boundary = 'symm', mode='same'); 
         tF = td.Transformer(self.img_);
         self.img_frr  = tF.get_ft_spectrum();
         
         self.fig, self.ax = plt.subplots(1, 4, figsize=(23, 9));
         self.fig.subplots_adjust(bottom=0.35, left=0.02, top=0.97, right=0.98)
         
         self.ax[0].imshow(self.my_kernel, aspect='equal');
         self.ax[1].imshow(self.img_conv, aspect='equal');
         self.ax[2].imshow(self.img_frr, aspect='equal');
         self.ax[3].imshow(self.img_, aspect='equal');
         # plt.tight_layout();
         
         self.sliders    = [];
         for ss in np.arange(non_zero):
             bbox = plt.axes( [0.05, 0.03 + ss*0.05, 0.15, 0.03]); # x, y, width, height
             # Each slider is responsible for a non-zero entry in the kernel
             S    = Slider(bbox, 'Entry No. ' + str(ss), 0, 1, valinit = 1);
             S.on_changed(lambda val, ss = ss: self._update_common(ss, val));
             self.sliders.append(S);

    def _update_common(self, slider_no, val):
        # Remember the slider's value is already set; you need only to update
        # the kernel matrix and the display
        print ('The slider no is: ' + str(slider_no));
        [u, v] = self.non_zeros[slider_no];
        self.my_kernel[u, v] = val;
        
        self.img_conv = signal.convolve2d(self.img_, self.my_kernel, boundary = 'symm', mode='same'); 
        tF = td.Transformer(self.img_);
        self.img_frr  = tF.get_ft_spectrum();
         
        self.ax[0].imshow(self.my_kernel);
        self.ax[1].imshow(self.img_conv);
        self.ax[2].imshow(self.img_frr);
        self.ax[3].imshow(self.img_);
        print(repr(self.my_kernel))
        self.fig.canvas.draw_idle();        
         
##############################################################################
##############################################################################

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
#
cV = convolver(img, 16); # kernel size
#convolutions = cV._generate_random_convolutions(5, 16);
#convolutions = cV._generate_random_lconv(6, 9);
###
#cV.display(convolutions, 0, 9, show_kernel=True); # start index, number of images


# GOOD Matrx
#CC = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,  0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.11111111, 0.        , 0.11111111, 0.        ,   0.11111111, 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.11111111, 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.11111111, 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.11111111],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.11111111],
#       [0.        , 0.        , 0.        , 0.        , 0.11111111,   0.        , 0.        , 0.11111111, 0.        ]]);
    
#CC = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,  0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],  
#       [0.        , 0.        , 0.1111111         , 0., 0.1111111 ,   0.        , 0.11111111       , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.1111111 , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.1111111 , 0.        ,   0.        , 0.        , 0.        , 0.        ],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.11111111],
#       [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.11111111],
#       [0.        , 0.        , 0.        , 0.        , 0.11111111,   0.        , 0.        , 0.11111111, 0.        ] ]);
#       

CC = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.      ],
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.1111111 ,        0.        ],  
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0         ],
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.       ],
               [0.        , 0.        , 0.        , 0.        , 0.111111  ,   0.        , 0.        , 0.        , 0.       ],
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.         , 0.        ],
               [0.        , 0.111111  , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.        , 0.        ,   0.        , 0.        , 0.        , 0.]]);
    
    
CC2 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.33333333, 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.33333333, 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.33333333],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ],
   [0.        , 0.        , 0.        , 0.        , 0.        ,
    0.        , 0.        , 0.        , 0.        ]]);
    
CC4 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.33333333,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.33333333, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.33333333,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ]])
    
    
CC5 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.33333333],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.33333333, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.33333333, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ]])
    
CC6 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.33333333, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.33333333, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.33333333,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ]])
    
    
CC7 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.33333333,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.33333333, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.33333333, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ]])
    
CC8 = np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0.2, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ]]);
    
CC9 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.422222  , 0.    , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.33333333, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.44444   , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]]);
    
CC13 = np.array([[0.        , 0.        , 0.        , 0.33333333, 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.33333333, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.33333333, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        ]]);
    
DD1 =  np.array([[0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0.2, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]]);
    
DD2 =  np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    
DD3 = np.array([[0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    

DD4 = np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    
EE1 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.16666667, 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.16666667,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.16666667,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.16666667, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.16666667, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.16666667, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ]]);
    
CC15 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.16666667, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.16666667, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.16666667, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.16666667,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.16666667,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.16666667, 0.        , 0.        ,
        0.        ]]);

    
CC16 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.16666667],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.16666667, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.16666667, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.16666667, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.16666667, 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.16666667, 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ]])
    
CC17 = np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0.2, 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. , 0. ]])
   
    
DD5 = np.flip(DD4, axis=1);
# print(repr(CC10));
    
#cV.display_single(CC13);
    

#    
cV.tuner(CC17);
plt.show();


#conv=np.array([[0, 0, 0, 0.5, 0.5, 0, 0],
#               [0, 0, 4, 0, 0, 8, 0],
#               [0, 0.5, 0, 0, 0, 0, 0.70],
#               [0.5, 0.1, 0, 0, 0, 0, 0.30],
#               [1, 0, 0, 0, 0, 0, 0.15],
#               [0, 0, 0, 0, 0, 0.5, 0],
#               [0, 0, 0, 0, 1, 0, 0]]);
 #=============================================================================
#conv=np.array([[0, 0, 0, 1, 1, 0, 0],
#               [0, 0, 1, 0, 0, 1, 0],
#               [0, 1, 0, 0, 0, 0, 1],
#               [0, 1, 0, 0, 0, 0, 1],
#               [1, 0, 0, 0, 0, 0, 1],
#               [0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 1, 0, 0]])*(1/12.0);


#conv=np.array([[4, 0,    0,   0,    4],
#               [0, 2,    0,   2,    0],
#               [0, 0,    1,   0,    0],
#               [0, 2,    0,   2,    0],
#               [4, 0,    0,   0,    4]]);
#    
#conv = conv *(1/np.sum(conv));
# =============================================================================

#conv=np.array([[0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 1, 1, 0, 0],
#               [0, 1, 0, 0, 0, 1, 0],
#               [0, 1, 0, 0, 0, 0, 1],
#               [1, 0, 0, 1, 0, 0, 1],
#               [0, 0, 1, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1, 0]]);


#conv=np.array([[0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 1, 0, 0, 0],
#               [0, 0, 1, 0, 0, 1, 0, 0],
#               [0, 1, 0, 0, 0, 0, 1, 0],
#               [1, 0, 0, 0, 0, 0, 1, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 1, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0]]) * (1/14.0);


""" 
################### OLD CODE ##########################
img_conv = signal.convolve2d(img, conv, boundary = 'symm', mode='same');


fig = plt.figure(figsize=(10,10));
plt.imshow(img_conv)
plt.show();

#######################################################
"""
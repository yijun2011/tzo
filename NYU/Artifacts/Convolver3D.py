#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:24:24 2018

@author: oscar seurat
"""

import Loader as ld;
from scipy import ndimage as ndi;
import numpy as np;
from mpl_toolkits.mplot3d import Axes3D;
import matplotlib.pyplot as plt;

class convolver3d(ld.mri_scan):
    def __init__(self, path):
        super().__init__(path);
        
    """
    We want to be able to generate a convolver3d from a given 
    mri_scan object 
    """
    @classmethod
    def from_mri_scan(cls, mri_scan):
        new_conv3d = cls(path = None);
        new_conv3d.img = mri_scan.img;
        new_conv3d.x_max = mri_scan.x_max;
        new_conv3d.y_max = mri_scan.y_max;
        new_conv3d.z_max = mri_scan.z_max;
        return new_conv3d;
           
    """
    This method convolves the given kernel with underlying 3D image and 
    returns a brand new independent convolver with the convolved cube as
    its underlying image.
    """
    def convolve3d(self, kernel):
        new_scan = ld.mri_scan.copy_scan(self, copy_img = False);
        conv_out = np.zeros([self.x_max,self.y_max, self.z_max]);
        ndi.convolve(self.img.dataobj, kernel, output = conv_out, mode='constant', cval=0.0);
        new_scan.img.dataobj=conv_out;
        new_convolver = convolver3d.from_mri_scan(new_scan);
        return new_convolver;
    
    """
    Generate a 3D convolution kernel:
    size      --> dimension of the cube
    func      --> a T/F function of the pixel (x,y,z) which decides whether a pixel
                  gets turned on
    non_zeros --> how many non-zero pixels should we have
    var       --> a magnitude of a random deviation from the pixel given by func
    prob      --> probability of the deviation occuring
    """
    
    @staticmethod
    def generate_f_kernel(size, func, num_non_zeros, var = 1, prob = 0.25, seeded = True):
        
        cube = np.zeros([size, size, size]);
        non_zeros = [];
        if (seeded):
            np.random.seed(1000);
        
        # Kernels will be generally small, so we'll
        # apply the brute-force approach
        for x in np.arange(size):
            for y in np.arange(size):
                for z in np.arange(size):
                    if (func(x,y,z) == True):
                        non_zeros.append([x,y,z]);
                        
        L = len(non_zeros);
        if (L == 0):
            msg = 'convolver3d.generate_f_kernel: coordinates of no points satisfy the required condition.';
            raise ValueError(msg);
            
        if (var >= 1):
            to_perturb = np.random.random_integers(0, high=L-1,
                                                   size=int(prob * L));
            for ss in np.arange(len(to_perturb)):
                coord = np.random.randint(0,2);
                pt_num = to_perturb[ss];
                point = non_zeros[pt_num];
                actual_var = np.random.randint(0,max(var,1));
                if (point[coord] < size - actual_var):
                    point[coord] = point[coord]+ actual_var;
                    
                non_zeros[pt_num] = point;
            
        # Now we have to pick only num_non_zeros aout of the non_zeros
        to_choose = np.random.random_integers(0, high = L-1, size=num_non_zeros);
        final_choice = [ non_zeros[jj] for jj in to_choose ];
    
        for item in final_choice:
            cube[item[0],item[1], item[2]]= 1;
            
        return cube;
    
    

    def show_kernel_convolved(self, kernel, slice_no, axis, rotate):
        x, y, z = kernel.nonzero();
        fig = plt.figure(figsize = (20, 10));

        ax = fig.add_subplot(1, 2, 1, projection='3d', aspect=1.0);
        ax.scatter(x, y, z);
        ax.set_title('Kernel')

        img = self.get_image(slice_no, axis, rotate);        
        ax = fig.add_subplot(1, 2, 2);
        ax.imshow(img);
        ax.set_title('Convolved')
        plt.tight_layout();
        plt.show();
        
        
    def show_kernel_orig_convolved(self, orig, kernel, slice_no, axis, rotate):
        x, y, z = kernel.nonzero();
        fig = plt.figure(figsize = (22, 8))
        img = orig.get_image(slice_no, axis, rotate);           
        ax = fig.add_subplot(1, 3, 1);
        ax.imshow(img);
        
        ax.set_title('Original')        
        
        ax = fig.add_subplot(1, 3, 2, projection='3d', aspect=1.0);
        ax.scatter(x, y, z);
        ax.set_title('Kernel')

        img = self.get_image(slice_no, axis, rotate);        
        ax = fig.add_subplot(1, 3, 3);
        ax.imshow(img);
        ax.set_title('Convolved')
        plt.tight_layout();
        plt.show();
               
        

    
##########################################################
##########################################################
     
if 1:
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';   

    cN = convolver3d(MRI_PATH + SCAN);
    
    
#    kernel = np.zeros([16, 16, 16]);
#    kernel[2, 2, 15] = 1;
#    kernel[7, 7, 14] = 1;
#    kernel[9, 10, 9] = 1;
#    kernel[14, 14, 2] = 1;
    
    
    func = lambda x, y, z: (x - 16)**2 + (y-16)**2 + (z-16)**2 >= 58 and \
    (x-16)**2 + (y-16)**2 + (z-16)**2 <= 60  and x >= 16 and y >= 16 and z >= 16;
   

   # func = lambda x, y, z: z == 2*x +1.5*y + 8 and np.remainder(x, 2) == 0 and np.remainder(y, 2) == 0 and np.remainder(z, 2)==0;
    kernel = convolver3d.generate_f_kernel(32, func, 16, var = 0, seeded=False);
    convolved = cN.convolve3d(kernel);
  #  convolved.show_slice(89, 'x', rotate=1);
    convolved.show_kernel_orig_convolved(cN, kernel, 89, 'x', rotate=1);
    

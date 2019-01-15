#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:09:06 2018

@author: oscar seurat
"""
import numpy as np;
import sys;
from numpy import random;
import math;
import Loader as ld;
import matplotlib.pyplot as plt;
import Transformer;
import Convolver;
import scipy.signal;

class Combiner:
    def __init__(self, img1, img2):
        # Images are assumed to be numpy arrays
        m1, n1 = img1.shape;
        m2, n2 = img2.shape;
        
        if (m1 != m2 or n1 != n2):
            print('Combiner.combiner: the images to combine should have the ' + 
                  'same sizes.', sys.stderr);
            sys.exit(-1);
            
        self.img1 = img1;
        self.img2 = img2;
        self.inner_rect = 0.80; """Currently (11/10/2018), the combiner picks
         a similar rectangle (remember similar triangles?) centrally (or
         symmertically) located in the image and select from it randomly
         centers of circular masks """
        
    """
    Apply mask: 1 chooses pixels from self.img2, 0 from self.img1
    """
    def _simple_merge(self, mask):
        
        m, n = mask.shape;
        p, q = self.img1.shape;
        
        if (m != p or n != q):
            print('Combiner._simple_merge: the combining mask is not of the ' + 
                  'same size as the images to be combined.', sys.stderr);
        
        c_img = self.img1*(1-mask) + mask * self.img2;
        return c_img;
    
    """
    This simply returns mask whose all non-zero values are located
    in the circle (center, radius)
    """
    def _circular_mask(self, center, radius):
        m, n = self.img1.shape;
        mask = np.zeros([m, n]);
        
        idx = [(x, y) for x in range(m) for y in range(n)
                if (math.sqrt((x-center[0])**2+(y-center[1])**2) <= radius)] 
        # Note: slow
        for pp in range(len(idx)):
            mask[idx[pp]] = 1;
        
        return mask;
    
    """
    Given an array of masks find their union
    The array of mask is intended to be short
    (<= 10 elems or so). In 
    Masks are assumed to be stacked along dimension 0.
    The mask_array is a cube (not necessarily even sided)
    """
    @staticmethod
    def _mask_union(mask_array):
        shape = mask_array.shape;
        if (len(shape)<2 or len(shape) > 3):
            print('Combiner._mask_union: the mask array must contain at least ' + 
                   'one mask(layer) and be at most 3 dimensional.', sys.stderr);
            sys.exit(-1);
             
        if (len(shape)==2):
            return mask_array;
             
        ret_mask = np.amax(mask_array, axis=2);
        return ret_mask;
         
         
    def random_circle_mask_array(self, k, radius_min, radius_max):
        m, n = self.img1.shape;
        if (radius_min <= 0 or radius_max <= 0 or
            radius_min > radius_max or
            radius_max > np.amin([m,n])/2):
            print('Combiner.random_circle_mask: rradii of circular masks are ' + 
                  'incorrect.', sys.stderr);
            sys.exit(-1);
        
        mask_array = np.zeros([m, n, k]);
            
        L = int(self.inner_rect*m);
        D = (m-L)/2;
        
        LL = int(self.inner_rect*m);
        DD = (n-LL)/2;
        for s in range(k):
            center_x = random.randint(D, m-D);
            center_y = random.randint(DD, n-DD);
            radius   = random.randint(radius_min, radius_max);
            mask = self._circular_mask([center_x, center_y], radius);
            mask_array[:,:, s] = mask;
            
        return mask_array;
    
    """
    This function is a bespoke one ("one off" to a degree). It takes the Transformer's method
    example_transf1 and applies it with a range of parameters producing
    an np.array of images. The arguments are based on those of Transormer.example_transf1,
    except that those with _t extension are pairs/triple in which the second element/third
    is a delta by which to increment/decrement the first element. The augmentation
    is repeated n times.
    **** DISTORTION PARAMETERS ****
    df_range is the distortion frequency range (an int) -- how much in maximum will the original
             circle be shifted randomly to generate varying frequencies of interference
             (note: the delta of the df_range shifts is given in the second element of the
             radii-t argument; 1 (pixel0 is a good value of delta)
             
    ph_range is the phase distortion range - to generate different values of phase shifts
             in order to vary that parameter value over the many generated distortions;
             again the delta is given in the second element of the pahse_p parameter
             
    amp_range is the amplitude of the distortion; we want to vary that parameter
             across all the distortion we make (again, the delta of the distorion is given
             as the second element of the argument)
             
    sec_range is the shift in the angles of the sector of directions for distortions
             that will be generated; the extent of the sector and its delta is 
             given in the sector_t argument; then that sector is shifted (randomly)
             by the sec_range parameter.
             
    ******* MASK PARAMETERS *******
    f_min is the fraction of min(img.shape) that the smaller radii of (important!)
          distortion circles will have (there will be anumber of such circles)
          
    f_max is the fraction of max(img.shape) that the larger of the radii of 
          distorion circles will have
          
    These lines of code explain the action of f_min, f_max (R1, R2 are the radii)
    R1 = int(f_min*np.amin([r,c]));
    R2 = int(f_max*np.amin([r,c]));
          
    num_circ_min is the minimal number of distortion circles that will be generated
    
    num_circ_max is the maximal number of circles that will be generated
    *******************************
    
    aux_img -- an image I2 which meant to 1) be distorted by transf specified by the
             previous parameters, and 2) overlayed over the original image in the 
             spots specified by the previous params. Normally I2 will be the original
             image itself (in other words we're simply distorting the original image).
             The need for this parameter arose out of the need to combine different types
             of distortions and I2 is really meant to be the original image distorted by
             a different independent transformation.
    """
    @staticmethod
    def loop_example_transf1(img, radii_t, phase_p, center, ampl_p, sector_t, num_img=4,
                             df_range = 10, ph_range = 3, amp_range = 3, sec_range = 10,
                             f_min=0.1, f_max=0.3, num_circ_min=1, num_circ_max=4,
                             aux_img = []):
        if (len(radii_t) != 3):
            print('Combiner.loop_exampe_transf1: the radii argument must be a tripple.',
                  sys.stderr);
            # sys.exit(-1);
            raise ValueError;
            
        if (len(phase_p) != 2):
            print('Combiner.loop_exampe_transf1: the phase_p argument must be a pair.',
                  sys.stderr);
            # sys.exit(-1);
            raise ValueError;

        if (len(ampl_p) != 2):
            print('Combiner.loop_exampe_transf1: the ampl_p argument must be a pair.',
                  sys.stderr);
            # sys.exit(-1); 
            raise ValueError;             

        if (len(sector_t) != 3):
            print('Combiner.loop_exampe_transf1: the sector_t argument must be a triple.',
                  sys.stderr);
            # sys.exit(-1);
            raise ValueError;
        
        rad1  = radii_t[0];  rad2 = radii_t[1];   dlt_r  = radii_t[2];
        phase = phase_p[0];                       dlt_ph = phase_p[1];
        ampl  = ampl_p[0];                        dlt_a  = ampl_p[1];
        alph1 = sector_t[0]; alph2 = sector_t[1]; dlt_s  = sector_t[2];
        
        r, c = img.shape;
        aout = np.zeros([r,c,num_img]);
        
        
        T = Transformer.Transformer(img);
        if (len(aux_img) != 0):
            T = Transformer.Transformer(aux_img);
           
        img_cntr = 0;
        for s in range(num_img):
            rr = random.randint(-df_range, df_range);
            pp = random.randint(0,ph_range);
            aa = random.randint(0,amp_range);
            AA = random.randint(0,sec_range);
            
            try:
                # This call generates distortion across the whole image
                img2, ignore = T.example_transf1(rad1 + rr*dlt_r, rad2 + rr*dlt_r,
                                                 phase + pp*dlt_ph, [0,0],
                                                 ampl  + aa*dlt_a,
                                                 [alph1 + AA*dlt_s, alph2 + AA*dlt_s],
                                                 fix_hist = True);
            except:
                continue;
            
            # The below  few calls select a part of the image (a mask) to apply the 
            # distortion to
            cB   = Combiner(img, img2);
            nn   = random.randint(num_circ_min, num_circ_max);
            # Image size is (r, c). What should the range of mask circle radii
            # be? Guess: between f_min=105, f_max=30%. Note that there will be many
            # such circles.
            R1 = int(f_min*np.amin([r,c]));
            R2 = int(f_max*np.amin([r,c]));
            mA   = cB.random_circle_mask_array(nn, R1, R2);
            mU   = Combiner._mask_union(mA);
            img3 = cB._simple_merge(mU);
            img4 = Transformer.Transformer.hist_match(img3, img);
            if (np.std(img4)!= 0):
                img_cntr += 1;
                aout[:,:, s] = img4;
        
        aout = np.delete(aout, range(img_cntr, num_img), 2);
        return aout;
          
#######################################################################
#######################################################################
        
if __name__ == "__main__": 
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
    # SCAN = 'sub-NC189_ses-20180825_acq-motion_run-02_T1w.nii'
    
    mri = ld.mri_scan(MRI_PATH + SCAN);    
   # img1 = mri.get_image(88, axis='x', crop_center=[10,10], crop_wh=[100,100]);
    img1 = mri.get_image(100, axis='x');
    
   # fig = plt.figure(figsize= (7,7));
   # plt.imshow(img1);
   # plt.show();
    
    """
    T = Transformer.Transformer(img1);
    # img2, ignore = T.example_transf1(60,65, np.pi, [0,0], 16, [1.5*np.pi, 1.82*np.pi],
    #                                 fix_hist = True);
                                     
    img2, ignore = T.example_transf1(55,60, 2.51, [0,0], 8.89, [5.18, 5.67],
                                     fix_hist = True);
    
    cB   = Combiner(img1, img2);
    mA   = cB.random_circle_mask_array(2, 10, 40);
    mU   = Combiner._mask_union(mA);
    img3 = cB._simple_merge(mU);
    img4 = Transformer.Transformer.hist_match(img3, img1);
    
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 10));
    ax[0].imshow(img1);
    ax[1].imshow(img2)
    ax[2].imshow(img4);
    plt.show();
    """   
 
    # imarr = Combiner.loop_example_transf1(img1, [60, 65, 1], [2.51, 0], [0,0], [8.89, 0], [0.18, 0.67, np.pi/7], 9);
    
    # U should 2, 3, or 4
    U =2;
    # imarr = Combiner.loop_example_transf1(img1, radii_t = [52.25, 55.25, 0.1], df_range = 5, phase_p=[2.4, 0], center=[0,0],
    #                                      ampl_p = [0.1, 0.00], amp_range = 4, sector_t = [0, 3, np.pi/18], sec_range=16,
    #                                      num_circ_min = 12, num_circ_max = 15, n = U**2);
    
    # These are good settings for 100 x 100 images
    # imarr = Combiner.loop_example_transf1(img1, [31, 36, 1], [2.51, 0], [0,0], [8.89, 0], [0.18, 0.67, np.pi/7], 4);
    
    

    kernel = np.zeros((16,16))
    kernel[10,3] = 0.2
    kernel[12,12] = 0.2
    kernel[5,5] = 0.2
    kernel[2,2] = 0.2
    kernel[3,10] = 0.2
    
    # The below code uses Timek's masking mechannism; I actually do not need it
    """
    aux_img= Convolver.generate_blended_images_constant_kernel(img1, 1, kernel);
    aux_img = aux_img[:,:, 0];
    """
    aux_img = scipy.signal.convolve2d(img1, kernel, boundary = 'symm', mode='same');
    imarr = Combiner.loop_example_transf1(img1, [61, 66, 1.5], [2.51, 0], [0,0], [8.89, 0], [0.18, 0.67, np.pi/7],
                                          f_min=0.06, f_max=0.18, num_img = U**2, aux_img = aux_img);
    
    # save_path = "/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/";
    # save_file = "1000imagesZ";
    # np.savez_compressed(save_path + save_file, imarr=imarr);
   

    fig, ax = plt.subplots(U,U, figsize=(14,16));
    for q in range(U):
        for r in range(U):
            ax[q,r].imshow(imarr[:,:, U*q + r], interpolation='nearest', aspect='equal');        
    plt.show();
        
        
    """
    Configuration 1:
    
    Radii [r1, r2] = [35.40 , 38.40] (spacing 3) vary from r1 = 25 --> r2 = 45.50
    Phase = 2.19 <-- do not vary
    Ampl (!) = 0.27 --> vary from 0.1 --> 0.4
    center = [0,0]
    A1, A2 = [1.59, 4.59] (spacing 3) vary from 0 --> .27

    """    
        
        

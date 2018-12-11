#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np;
import Loader as ld;
import matplotlib.pyplot as plt;
import sys;
import math;
import cmath;
from skimage import exposure, img_as_float;

class Transformer:
    def __init__(self, image):
        self.image = image;
    
    @staticmethod
    def log_transform(image, multiplier = 1, base = 2.71828182845904):
        new_image = multiplier * np.log(1+image) / np.log(base);
        return new_image;
    
    
    """
    Given angles a1 < a2 (and starting from the x-axis in the ccw direction as
    is the convention) the below function decides whether a point (x,y) is within
    the cone spanned by (a1, a2)
    """
    @staticmethod
    def is_in_sector(center, angle_pair, point_2D, symmetric = 1):
        ret_val = False;
        a1 = angle_pair[0];
        a2 = angle_pair[1];
        
        if (a1 > a2):
            # print('Transformer.is_in_sector: the first angle of the sector ' + 
            #      'must smaller than the second.', sys.stderr);
            # sys.exit(-1);
            msg_str = 'Transformer.is_in_sector: the first angle of the sector must smaller than the second.';
            raise ValueError(msg_str);
        
        x = point_2D[0] - center[0];
        y = point_2D[1] - center[1];
        
        z = complex(x, y);
        alpha = cmath.phase(z);
        
        if (alpha < 0):
            alpha = 2*np.pi + alpha;
            
        if (symmetric == 0 or a2 >= a1 + np.pi):
            if (alpha >= a1 and alpha <= a2):
                ret_val = True;
        else:
            s = (alpha + np.pi) % (2*np.pi);
            if ((alpha >= a1 and alpha <= a2) or 
                (s >= a1 and s <= a2)):
                ret_val = True;   
            
        return ret_val;
        
    """
    This function is an auxiliary function which multiplies
    every pixel by (-1)^(x+y). This has the effect that FourierTransform(0,0)
    moves to the center.
    """
    @staticmethod
    def _shift_to_center(image):
         m, n = image.shape;
         u = [(-1)**x for x in range(m)];
         u = np.asarray(u).reshape(m,1);
         v = [(-1)**y for y in range(n)];
         v = np.asarray(v).reshape(1,n);
         w = u*v;
         shifted = w*image;
         return shifted;
     
    def get_ft_raw(self):
        shifted = self._shift_to_center(self.image);
        FT = np.fft.fft2(shifted);
        return FT;
    
    def get_ft_spectrum(self):
        return Transformer.log_transform(abs(self.get_ft_raw()));
    
    """
    apply_ft_filter applies the given agrument function to the 
    Fourier transform of the stored image. The function takes in
    an FFT output of self.image and produces a transformed version
    of it. The return value is the transformed image.
    """
    def apply_ft_filter(self, function):
        FT  = self.get_ft_raw();
        FTT = function(FT);
        img = np.fft.ifft2(FTT).real;
        # Important: undo the shift of F(0,0) to the center (by applying it again)
        IMG = self._shift_to_center(img);
        return IMG;
    
    
    @staticmethod
    def apply_sp_filter(img, function):
        img = function(img);
        return(img);
        
        
    @staticmethod
    def power_stretch(img, gamma):
        imf = img_as_float(img);
        N = np.amin(imf);
        shift = 0;
        if (N < 0):
            shift = abs(N)
            imf += shift;
            
        M = np.amax(imf);
        imf_ = imf/M;
        img_out = M*imf_**gamma + shift ;
        return img_out;
    
    """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image
    
        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
    """
    @staticmethod
    def hist_match(source, template): 
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
    
        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, indices, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
    
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
    
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[indices].reshape(oldshape)
    
    """
    Simple validator
    """
    @staticmethod
    def _validate_radius(ft, radius):
        m, n = ft.shape;
        r = np.amin((m, n));
        
        if (radius > r/2):
#            print('Transformer.high_pass_circular: to large radius given: ' +
#                  'maximum allowable radius value is ' + str(r/2) + '.', sys.stderr);
#            sys.exit(-1);   
            msg_str = 'Transformer.high_pass_circular: to large radius given: maximum allowable radius value is ' + str(r/2) + '.'
            raise ValueError(msg_str);
    """
    Some simple filters.
    high_pas_01 looks at the centered Fourier transform, carves out in it a circle
    with a given center and radius, and then sets all the values in that circle
    to 0.
    """
    @staticmethod
    def high_pass_01(ft, radius, center = [0, 0], sector = [0, 2*np.pi]):
        Transformer._validate_radius(ft, radius);

        m, n = ft.shape;
        idx = [ (x, y) for x in range(m) for y in range(n)
                if (  (math.sqrt((x-(m/2.0 + center[0]))**2+(y-(n/2.0 + center[1]))**2) <= radius) and
                    (Transformer.is_in_sector([m/2, n/2], sector, [x, y]) == True))];
                  
        I = np.ones((m,n));
        # Note: slow
        for pp in range(len(idx)):
            I[idx[pp]] = 0;
        
        return I * ft;
    
    """
    The low_pas_01 looks at the centered Fourier transform, carves out in it a circle
    with a given center and radius as above, but then sets all the values outside of
    that circle to 0.
    """
    @staticmethod
    def low_pass_01(ft, radius, center = [0, 0], sector = [0, 2*np.pi]):
        Transformer._validate_radius(ft, radius);
        
        m, n = ft.shape;
        idx = [ (x, y) for x in range(m) for y in range(n)
                if ((math.sqrt((x-(m/2.0 + center[0]))**2+(y-(n/2.0 + center[1]))**2) <= radius) and
                    (Transformer.is_in_sector([m/2, n/2], sector, [x, y]) == True))];
                  
        I = np.zeros((m,n));
        for pp in range(len(idx)):
            I[idx[pp]] = 1;
        
        return I * ft;  
    
    """
    This function "carves out" a ring in the centered Fourier transform and 
    zeroes everything outside of it. This action is obtained by a suitable
    composition of high_pass_01 and low_pass_01 functions.
    """
    @staticmethod
    def ring_pass_01(ft, radius_min, radius_max, center = [0, 0], sector = [0, 2*np.pi]):
        if (radius_min > radius_max):
#            print('Transformer.high_pass_01_ring: min radius is greatr than max radius',
#                  sys.stderr);
#            sys.exit(-1);
            msg_str = 'Transformer.high_pass_01_ring: min radius is greatr than max radius.';
            raise ValueError(msg_str);
        
        ft_max = Transformer.low_pass_01(ft, radius_max, center, sector);
        ft_min = Transformer.high_pass_01(ft_max, radius_min, center, sector);
        return ft_min;

    """
    This functtion takes the centered Fourier transform, "carves out' in it a circle
    of the given radius and center, and changes phase of every value located
    in that circle. In addition, the amplification parameter scales the changed
    values by the given parameter.
    """
    @staticmethod
    def low_pass_const_phase_shift(ft, radius, angle, center = [0, 0], amplification = 1,
                                   sector = [0, 2*np.pi]):
        Transformer._validate_radius(ft, radius);
        
        m, n = ft.shape;
        idx = [ (x, y) for x in range(m) for y in range(n) 
                if ((math.sqrt((x-((m/2.0) + center[0]))**2+(y-((n/2.0) + center[1]))**2) <= radius) and
                    (Transformer.is_in_sector([m/2, n/2], sector, [x, y]) == True))];
        
        pshift = cmath.rect(1, angle);
        I = np.ones((m,n), dtype=np.complex_);
        for pp in range(len(idx)):
            I[idx[pp]] = pshift * amplification;
        
        return I * ft;
    
    """
    This functtion takes the centered Fourier transform, "carves out' in it a circle
    of the given radius and center, and changes phase of every value located
    **outside** of that circle. In addition, the amplification parameter scales
    the changed values by the given parameter.
    """
    @staticmethod
    def high_pass_const_phase_shift(ft, radius, angle, center = [0, 0], amplification = 1,
                                    sector = [0, 2*np.pi]):
        Transformer._validate_radius(ft, radius);
        
        m, n = ft.shape;
        idx = [ (x, y) for x in range(m) for y in range(n) 
                if ((math.sqrt((x-((m/2.0) + center[0]))**2+(y-((n/2.0) + center[1]))**2) > radius) and
                    (Transformer.is_in_sector([m/2, n/2], sector, [x, y]) == True))];
        
        pshift = cmath.rect(1, angle);
        I = np.ones((m,n), dtype=np.complex_);
        for pp in range(len(idx)):
            I[idx[pp]] = pshift * amplification;
        
        return I * ft;
    
    """
    This function "carves out" a ring with the given radii and center and changes in it the 
    phase of the Fourier transform. It also magnifies the modulus of the values
    in the ring.
    """
    @staticmethod
    def ring_const_phase_shift(ft, radius_min, radius_max, angle, center = [0,0], amplification = 1.0,
                               sector = [0, 2*np.pi]):
        if (radius_min > radius_max):
#            print('Transformer.high_pass_01_ring: min radius is greatr than max radius.',
#                  sys.stderr);
#            sys.exit(-1);
            msg_str = 'Transformer.high_pass_01_ring: min radius is greatr than max radius.';
            raise ValueError(msg_str);
            
        ft_max = Transformer.low_pass_const_phase_shift(ft, radius_max, angle,
                                                        center, amplification, sector);
        ft_out = Transformer.low_pass_const_phase_shift(ft_max, radius_min, -angle,
                                                        center, 1.0/amplification, sector);
        return ft_out;
        
    """
    This is a general function which applies a filter to the centered Fourier
    transform and produces log transformed spectrum of the result (for testing
    purposes).
    """
    def get_filter_spectrum(self, filter):
        ft = self.get_ft_raw();
        filtered = filter(ft);
        spectrum = Transformer.log_transform(abs(filtered));
        return spectrum;
    
    """
    This is a particular function which implements one kind of FT based
    transformation (distortion actually). The reason for it is to generate
    distorted images automatically
    """
    def example_transf1(self, inner_r, outer_r, ph_shift, center, ampl, sector,
                        fix_hist = True):
        # Set the function which distorts Fourier Transform of the image
        t_func = lambda x_ft: Transformer.ring_const_phase_shift(x_ft,
                                                                 inner_r,
                                                                 outer_r,
                                                                 ph_shift,
                                                                 center,
                                                                 ampl,
                                                                 sector); 
                                                                           
        # Get the function which transform the image spatially after it has 
        # been altered by t_func
        sp_func = lambda x_img: Transformer.hist_match(x_img, self.image);
        
        # Distort the Fourier transform of the image and get back the so
        # distoted image
        trf_tmp_img = self.apply_ft_filter(t_func);
        
        # Just in case it was needed, get the spectrum of the FT
        flt_spec = self.get_filter_spectrum(t_func);
        
        # Fix the histogram to the original image
        if (fix_hist == True):
            trm_img = Transformer.apply_sp_filter(trf_tmp_img, sp_func); 
        else:
            trm_img = trf_tmp_img;
            
        return [trm_img, flt_spec];
        
##############################################################
##############################################################
    
if 0:
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
    # SCAN = 'sub-NC189_ses-20180825_acq-motion_run-02_T1w.nii'
    
    mri = ld.mri_scan(MRI_PATH + SCAN);    
    img = mri.get_image(100);
    
    T = Transformer(img);
    
    #tfunc = lambda ft: Transformer.low_pass_const_phase_shift(ft, 60, np.pi/2.5,[0, 0]);
    #tfunc = lambda ft: Transformer.ring_const_phase_shift(ft, 60,70,np.pi/1.45, [0, 0], 6);
    
    tfunc = lambda ft: Transformer.ring_const_phase_shift(ft, 60,65, np.pi, [0,0], 16, [1.5*np.pi, 1.82*np.pi]);
    lp_spec = T.get_filter_spectrum(tfunc);
    
    
    #function = lambda img: Transformer.high_pass_const_phase_shift(img, 65, np.pi/1.45, [0, 0], 8);
    
    #function = lambda img: Transformer.ring_pass_01(img, 35, 88, [0, 0]);
    
    # function = lambda img: Transformer.ring_const_phase_shift(img, 60,65, np.pi/1.75, [0, 0], 8);
    
    # sp_func = lambda img: Transformer.power_stretch(img, 1.0);
    
    sp_func = lambda x: Transformer.hist_match(x, img);
    t_img = T.apply_ft_filter(tfunc);
    s_img = Transformer.apply_sp_filter(t_img, sp_func);
    
    fit, ax = plt.subplots(1, 3, figsize=(10, 7));
    ax[0].imshow(img);
    ax[1].imshow(lp_spec)
    ax[2].imshow(s_img);
    plt.show();



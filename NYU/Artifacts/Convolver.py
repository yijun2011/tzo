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
from scipy.linalg import circulant
from mpl_toolkits.mplot3d import Axes3D;



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
    Generate a 2D convolution kernel:
    size      --> dimension of the cube
    func      --> a T/F function of the pixel (x,y,z) which decides whether a pixel
                  gets turned on
    non_zeros --> how many non-zero pixels should we have
    var       --> a magnitude of a random deviation from the pixel given by func
    prob      --> probability of the deviation occuring
    """

    @staticmethod
    def generate_f_kernel(size, func, num_non_zeros, var = 1, prob = 0.25, seeded = True):

        window = np.zeros([size, size]);
        non_zeros = [];
        if (seeded):
            np.random.seed(1000);

        # Kernels will be generally small, so we'll
        # apply the brute-force approach
        for x in np.arange(size):
            for y in np.arange(size):
                if (func(x,y) == True):
                    non_zeros.append([x,y]);

        L = len(non_zeros);
        if (L == 0):
            msg = 'convolver3d.generate_f_kernel: coordinates of no points satisfy the required condition.';
            raise ValueError(msg);

        if (var >= 1):
            to_perturb = np.random.random_integers(0, high=L-1,
                                                   size=int(prob * L));
            for ss in np.arange(len(to_perturb)):
                coord = np.random.randint(0,1);
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
            window[item[0],item[1]]= 1;

        return window/len(final_choice);


    """
    This function generates random convolutions and outputs them in a 3d array
    """

    def _generate_random_convolutions(self, non_zeros, num_conv, seeded = True):
        if seeded:
            np.random.seed(1000);

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
    This function generates perturbed linear convolutions (perturbed by Gaussian noise),
    that is ones whose non_zero entries in the kernel form a line
    """


    def _generate_random_lconv(self, non_zeros, num_conv, max_a = 3 , sigma = 0.5, seeded = True):
        if seeded:
            np.random.seed(1000)
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
    Auxiliary function to extract a submatrix S from a given image. The retrieved
    submatrix is intended to be later convolved with the given kernel.
    S's size will be the same as that of K and S's entries will
    depend on the pixel around which S is centered. If S "sticks" out of the image, 0's will
    be used as image's pixel values.
    
    kernel  -- the kernel which (later) will be convolved with the image
    pixel_x -- row of the point in the image over which kernel's center is superimposed
    pixel_y -- column of the point in the image over which kernel's center is superimposed
    
    """
    def _get_phantom_submatrix(self, kernel, pixel_x, pixel_y):
        K, ignore = kernel.shape;
        phantom = np.zeros([K, K]);
        
        i  = pixel_x; j = pixel_y;
        C  = int(np.ceil((K/2) - 1));
        C_ = int(np.floor(K/2));
        
        N, M = self.img_.shape;
        
        # Let's fill the matrix 
        for s in np.arange(-C, C_+1):
            if (i + s < 0 or i + s > N - 1 ):
                continue;
            for r in np.arange(-C, C+1):
                if (j + r < 0 or j + r > M - 1):
                    continue;
                phantom[C+s, C+r] = self.img_[i+s, j+r];
                
        return phantom;
    
    # Flip a matrix about diagonal
    @staticmethod
    def _flip_diag(matrix, diag):
        N, M = matrix.shape;
        new_matrix = np.zeros([N, M]);
        [idx_x, idx_y] = np.nonzero(matrix);
        K = len(idx_x);
        
        if (diag == 'NE'): # Top points to NE
            for k in np.arange(K):
                new_matrix[N - idx_x[k]-1, M - idx_y[k] - 1] = \
                matrix[idx_x[k], idx_y[k]];
        else:
             new_matrix = np.transpose(matrix);
                
        return new_matrix;
                
        
    def _convolve_pixel(self, kernel, pixel_x, pixel_y, diag_flip = False):
        if (diag_flip):
            kernel = convolver._flip_diag(kernel, 'NE');
        phantom = self._get_phantom_submatrix(kernel, pixel_x, pixel_y);
        res = np.multiply(kernel, phantom);
#        print('res is:')
#        print(res);
#        print('\n');
#        print(np.sum(res));
#        print('\n');
        res = np.sum(np.sum(res));
        return(res);
        
        
        
    @staticmethod
    def _rotate_ccw(kernel, angle_rad):
        K, ignore = kernel.shape;
        new_kernel = np.zeros([K, K]);
        NN = np.count_nonzero(kernel);
          
        C  = int(np.ceil((K/2) - 1));
        C_ = int(np.floor(K/2));
          
        for s in np.arange(-C, C_+1):
            for r in np.arange(-C, C_+1):
                  # Need to use ceil/floor and eliminate value that stick out
                  # of the kernel
                s_ = np.rint(s*np.cos(angle_rad) + r*np.sin(angle_rad));
                r_ = np.rint(-s*np.sin(angle_rad) + r*np.cos(angle_rad));
                # I am taking advange of the fact that entries of the
                # kernel are >= 0
                if (C + s_ >= 0 and C + s_ < K and
                    C + r_ >= 0 and C + r_ < K and
                    np.count_nonzero(new_kernel) <= NN):
                    new_kernel[C+s, C+r] = kernel[int(C + s_), int(C + r_)];
                else:
                    new_kernel[C+s, C+r] = 0;

        # Make sure that the entries of a kernel add up to 1
        ss = np.sum(new_kernel);
        new_kernel /= ss;
        return new_kernel;
    
    """
    We're given an N x M image (matrix) I and a point (i, j) in it. We define
    "center" MID of I as  MID = (R, C) where C = ceil( (M/2) - 1  ) (Note that R is
    really intended to be >= N )
    
    We want to know what is the angle < pi formed by 1) the vertical line passing 
    through MID and 2) the line passing through MID and the point (p, q). Directions
    to the left from the line 1) are positive, while those to the right negative.
    """
                
    def _get_angle(self, pixel_x, pixel_y, R): # R should be N
        i = pixel_x;
        j = pixel_y;
        
        N, M  = self.img_.shape;
        C     =   int(np.ceil((M/2) - 1));
        sign  = np.sign(C - j);
        hpnse = np.sqrt( (C-j)**2 + (R-i)**2);
        adjcn = R - i;
        
        angle = 0;
        if (hpnse != 0):
            angle = np.arccos(adjcn/hpnse);
        return(sign * angle);

    """
    Let's assume that the image width is M. As in the _get_angle function, we define
    the special center as a point MID = (radius, M/2 ). 
    This function shrinks the kernel in way that is proportional to the distance
    between the kernel center (p, q) and the MID. That is at the kernel gets
    smaller and smaller as we approach MID at which point its support becomes
    a single point (the kernel's center).
    """
    def _shrink_kernel(self, kernel, pixel_x, pixel_y, R):
        p = pixel_x;
        q = pixel_y;
        
        N, M  = self.img_.shape;
        C     =   int(np.ceil((M/2) - 1)); 
        
        # The distance of the tl vertex to the MID point
        MX = np.sqrt(C**2 + N**2);
        DX = np.sqrt((C - q)**2 + (R - p)**2);
        ratio = DX/MX;
        
        # Now apply the "ratio" to the (square) kernel
        K, ignore = kernel.shape;
        Ck  = int(np.ceil((K/2) - 1));
        Ck_ = int(np.floor((K/2)));
        nnz_arr = np.nonzero(kernel);
        nnz_x = nnz_arr[0];
        nnz_y = nnz_arr[1];
        new_kernel = np.zeros([K, K]);
        for i in np.arange(len(nnz_x)):
            new_u = Ck  + int(np.rint(ratio*(nnz_x[i] - Ck)));
            new_v = Ck_ + int(np.rint(ratio*(nnz_y[i] - Ck_)));
            new_kernel[new_u, new_v] = kernel[nnz_x[i], nnz_y[i]];
        
        
        # Make sure that the entries of a kernel add up to 1
        ss = np.sum(new_kernel);
        new_kernel /= ss;
        return new_kernel;
    
    """
    Rotation and shrinking is applied to the given kernel. The parameters 
    of rotation depend on the pixel in the image we are over, and the the 
    position of the MID (the center referred to in the above functions)
    """
    def _symmetric_precession(self, kernel, pixel_x, pixel_y, R):
         angle   = self._get_angle(pixel_x, pixel_y, R);
         kernel1 = self._rotate_ccw(kernel, angle);
         kernel2 = self._shrink_kernel(kernel1, pixel_x, pixel_y, R);
         return kernel2;
        
    """
    Given a kernel apply it to the underlying image in a convolution like manner,
    but the kernel is morphed as we slide it over the matrix. Let's say that the
    size of the image is N.
    
    1) The kernel is not modifed at pixel (0, N/2) and for all rows i > radius
       (radius is a peramater and generally is thought to be > N)
                
    2) As we deviate from the line j = N/2 the kernel's support is rotated by the
       angle proportional to the angle between the following two lines:
           a) theline formed by the convolved pixel and point with coordinagtes
              (radius, N/2)  (the value of radius is intended to be > N)
           b) the vertical line j = N/2

    3) Good to have: to account for the limited angular rotation of a human head, we cap
       the angle of rotation at the value that comes from the deviation angle of
       arctan[ (N/2 / radius) ] (angle given by the tl corner of the image)
       
    4) As the convolved pixel moves from row i = 0 to row i = radius, the support of
       the kerner shrinks "perspectively" along the line connecting the convolved
       pixel and the point (radius, N/2);
    """
    def var_convolve(self, kernel, precession_func, user_data = None):
        N, M = self.img_.shape;
        new_image = np.zeros([N, M]);
        for p in np.arange(N):
            for q in np.arange(N):
                kern_local = precession_func(kernel, p, q, user_data);
                val = self._convolve_pixel(kern_local, p, q);
                new_image[p, q] = val;
                
        return new_image;
        
    """
    Display convolutions with kernels givven conv_arr in a grid form.
    That is both the kernel and its effect on the underlying image will be shown
    by matplotlib in a grid form.

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
    the kernel, the original image, the convolved image, and the
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

"""
Given an array, reflects the value of each entry over the horizontal line
cutting the array in half.
Input:
    arr: A numpy 2d array
Return:
    A different numpy 2d array
"""
def flip_vertical(arr):
    kernel = np.copy(arr)
    height = len(kernel)
    for i in range(int(height/2)):
        for j in range(len(kernel[0])):
            kernel[i][j], kernel[height-1-i][j] = kernel[height-1-i][j], kernel[i][j]
    return kernel

"""
Given an array, reflects the value of each entry over the vertical line
cutting the array in half.
Input:
   arr: A numpy 2d array
Return:
    A different numpy 2d array
"""
def flip_horizontal(arr):
    kernel = np.copy(arr)
    width = len(kernel[0])
    for j in range(int(width/2)):
        for i in range(len(kernel)):
            kernel[i][width-1-j], kernel[i][j] = kernel[i][j], kernel[i][width-1-j]
    return kernel

"""
Given an array, reflect each point through the origin (center element)
Input:
    arr: A numpy 2d array
Return:
    A different numpy 2d array
"""
def flip_through_origin(arr):
    return flip_vertical(flip_horizontal(arr))


"""
Generate random kernels according to parametric equations
Input:
    None
Return:
    A list of numpy 2d arrays
"""
def generate_random_kernels():
    num_curves = 9
    #interval = [0, np.random.randint(2, 6)]
    interval = [0, np.random.uniform(1, 3.5)]
    coeff1 = np.random.randint(1,15)
    coeff2 = np.random.randint(1,15)
    radius1 = np.random.randint(6,8)
    radius2 = np.random.randint(6,8)
    #radius = 5
    points_per_kernel = 5

    # print("size: 16")
    # print("num_curves: " + str(num_curves))
    # print("points_per_kernel: " + str(points_per_kernel))
    # print("radius1: " + str(radius1))
    # print("radius2: " + str(radius2))
    # print("Coefficients: " + str(coeff1) + ", " + str(coeff2))
    # print("Interval: " + str(interval))
    return generate_kernels_parametric(16, num_curves, points_per_kernel, radius1, radius2, [coeff1, coeff2], interval)

 
"""
Generates a list of 2D kernels by taking time slices of a parametric equation of the form
x = c + a * sin(b * pi * t)
y = d + e * cos(f * pi * t)
Input:
    size: The width and height of the kernel
    num_curves: The number of kernels to generate
    points_per_kernel: The number of activation points in each kernel
    radius1: The maximum radius of the curve along the x direction (a in the equation)
    radius2: The maximum radius of the curve along the y direction (e in the equation)
    coeff: A list of length 2 corresponding to coefficients b,f respectively
    interval: A list of length 2 denoting the minimum time value, and the maximum time
    value in that order
"""
def generate_kernels_parametric(size, num_curves, points_per_kernel, radius1, radius2, coeff = [6,5], interval = [0, 2 * np.pi]):
    kernels = []
    kernel = np.zeros((size,size))
    starts = np.linspace(interval[0], interval[1], num_curves + 1)
    for k in range(len(starts)-1):
        arr = np.copy(kernel)
        t = np.linspace(starts[k],starts[k+1], points_per_kernel)
        y = radius2 * np.cos(coeff[1] * np.pi * t) + size/2
        x = radius1 * np.sin(coeff[0] * np.pi * t) + size/2
        for i in range(len(t)):
            arr[min(int(x[i]),15),min(int(y[i]),15)] = 1
        kernels.append(arr/points_per_kernel)
    return np.array(kernels)


"""
Generates N distorted images based on a single clear image by convolving it 
with random 2D kernels
Input: A numpy 2D array: img
       Number of distorted images to generate: N
Output: An image array storing the distorted images. Shape is (img.x,img.y,N)
"""
def generate_distorted_images(img, N):
    x,y = img.shape;
    imarr = np.zeros((x,y,N));

    kernels = set();
    while len(kernels) < N:
        if (len(kernels) % 500 == 0):
            print("Convolved " + str(len(kernels)) + " images");
        random_kernels = generate_random_kernels();
        for kernel in random_kernels:
            if (len(kernels) < N):
                tmp = len(kernels);
                kernels.add(random_kernels.tostring());
                if len(kernels) != tmp:
                    image = signal.convolve2d(img, kernel, boundary = 'symm', mode='same');
                    imarr[:,:,tmp] = td.Transformer.hist_match(image,img);

    return imarr;


"""
A class representing a window mask.

Constructor:
    m: positive integer height of the mask
    n: positive integer width of the mask
    offset: location of the maximum value in the mask relative to
    the center
    exponent: Exponent in the super gaussian equation
    kappa: constant kappa in the super gaussian equation
"""
class Mask:
    def __init__(self, m, n, offset, exponent, kappa):
        self.height = m;
        self.width = n;
        self.offset = offset;
        self.exponent = exponent;
        self.kappa = kappa;
        self.mask = self.generate_blend_mask(m,n,offset,exponent,kappa);


    """
    Generates a window according to the Super-Gaussian formula:
    If the maximum value occurs at the point (x,y):

    w(u,v) = e ^ (([-((r_uv) ** n)] / k)
    r_uv = sqrt(r_u ** 2 + r_v ** 2)
    r_u = u/x - 1
    r_v = v/y - 1

    Input:
        m: integer height of the mask
        n: integer width of the mask
        offset: list of integers offset of mask in x and y direction
        exponent: The exponent (n) in the formula
        kappa: The constant (k) in the formula
    Output:
        2D numpy array representing the mask
    """
    @staticmethod
    def generate_blend_mask(m,n,offset,exponent,kappa):
        mask = np.zeros((m,n));
        for i in range(m):
            for j in range(n):
                r_u = i/(m/2) - 1;
                r_v = j/(n/2) - 1;
                r_uv = np.sqrt(r_u**2 + r_v**2);
                new_x = i - offset[0];
                new_y = j - offset[1];
                if (-1 < new_x < m and -1 < new_y < n):
                    mask[new_x,new_y] = np.exp(-(r_uv**exponent)/kappa);
        return mask

    """
    Applies a mask to both an original and convolved image. NOTE: this modifies the
    convolved image in place.

    Input:
        orig: 2D real valued numpy array representing the original image
        convolved: 2D real valued numpy array representing the convolved image
        mask: A 2D real valued numpy array representing the mask
    Output:
        The result of applying the mask to the distorted and original image
    """
    def apply_mask(self,orig,convolved):
        for i in range(len(convolved)):
            for j in range(len(convolved[0])):
                convolved[i][j] = self.mask[i][j] * convolved[i][j] + (1 - self.mask[i][j]) * orig[i][j];
        return convolved;

    """
    Plots the mask associated with this class
    """
    def plot_mask(self):
        self.fig = plt.figure();
        self.ax = self.fig.add_subplot(111,projection='3d');


        axcolor = 'lightgoldenrodyellow'
        kappa_slider_pos = plt.axes([0.18, 0.07, 0.65, 0.03], facecolor=axcolor);
        kappa_slider = Slider(kappa_slider_pos, 'Kappa', 0, 1, valinit=0.5);
        exponent_slider_pos = plt.axes([0.18, 0.12, 0.65, 0.03], facecolor=axcolor);
        exponent_slider = Slider(exponent_slider_pos, 'Exponent', 1, 10, valinit=3);

        y = np.arange(len(self.mask))
        x = np.arange(len(self.mask[0]))
        self.X,self.Y = np.meshgrid(x,y);
        self.surf = self.ax.plot_surface(self.X,self.Y,self.mask);

        kappa_slider.on_changed(self.update_kappa);
        exponent_slider.on_changed(self.update_exponent);

        plt.show();

    def update_kappa(self,val):
        self.kappa = val;
        self.mask = self.generate_blend_mask(self.height, self.width, self.offset, self.exponent, val);
        y = np.arange(len(self.mask))
        x = np.arange(len(self.mask[0]))
        self.X,self.Y = np.meshgrid(x,y);
        self.surf.remove();
        self.surf = self.ax.plot_surface(self.X,self.Y,self.mask, color = 'BLUE');
        self.fig.canvas.draw();
    
    def update_exponent(self,val):
        self.exponent = val;
        self.mask = self.generate_blend_mask(self.height, self.width, self.offset, val, self.kappa);
        y = np.arange(len(self.mask))
        x = np.arange(len(self.mask[0]))
        self.X,self.Y = np.meshgrid(x,y);
        self.surf.remove();
        self.surf = self.ax.plot_surface(self.X,self.Y,self.mask, color = 'BLUE');
        self.fig.canvas.draw();


    """
    Shows the mask in a 3D plot.

    Input:
        A 2D numpy array representing the mask
    Output:
        None

    """
    def show_mask(self):
        y = np.arange(len(self.mask))
        x = np.arange(len(self.mask[0]))
        fig = plt.figure();
        ax = fig.add_subplot(111,projection='3d');
        axcolor = 'lightgoldenrodyellow'
        kappa_slider_pos = plt.axes([0.18, 0.07, 0.65, 0.03], facecolor=axcolor);
        x_slider = Slider(kappa_slider_pos, 'X', 0, 1, valinit=0.5);
        X,Y = np.meshgrid(x,y);
        ax.plot_surface(X,Y,self.mask);
        plt.show();


"""
Generate N random distortionts of an original image based on blending with
a windowed mask.

Input: A numpy 2D array: img
       Number of distorted images to generate: N
Output: An image array storing the distorted images. Shape is (img.x,img.y,N)
"""
def generate_blended_distorted_images(img, N):
    x,y = img.shape;
    imarr = np.zeros((x,y,N));
    
    kernels = set();
    while len(kernels) < N:
        if (len(kernels) % 500 == 0):
            print("Convolved " + str(len(kernels)) + " images");
        random_kernels = generate_random_kernels();
        for kernel in random_kernels:
            if (len(kernels) < N):
                tmp = len(kernels);
                kernels.add(random_kernels.tostring());
                if len(kernels) != tmp:
                    x1 = np.random.randint(-int(x/4), int(x/4))
                    x2 = np.random.randint(-int(y/4), int(y/4))
                    mask = Mask(x,y, [x1,x2],np.random.uniform(2, 7),0.03);
                    image = signal.convolve2d(img, kernel, boundary = 'symm', mode='same');
                    image = mask.apply_mask(img, image);
                    imarr[:,:,tmp] = td.Transformer.hist_match(image,img);

    return imarr;

##############################################################################
##############################################################################

# MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/Coregistered/';
# SCAN = 'aligned188ToNoMotionRun01.nii'

if __name__ == '__main__':
    np.set_printoptions(linewidth = 160);
    MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii';
    
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC189/ses-20180825/anat/';
    # SCAN = 'sub-NC189_ses-20180825_acq-motion_run-01_T1w.nii'
    
        
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
    # SCAN = 'sub-NC188_ses-20180825_acq-nomotion_run-01_T1w.nii'
    
         
    # MRI_PATH = '/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC183/ses-20180825/anat/';
    # SCAN = 'sub-NC183_ses-20180825_acq-nomotion_run-02_T1w.nii'  
    rotation_test = False;
    if (rotation_test):
        cc9 = np.array([[0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.33333333, 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]]);
       
        II = np.identity(11);
        
        U = convolver._rotate_ccw(II, np.pi/2);
        print('\n')
        print(II);
        print('\n\n');
        print(U);
        sys.exit(0);
        
        
    phantom_test = False;
    if (phantom_test == True):
        T = circulant(np.arange(11));
        tC = convolver(T, 5);
        kernel = np.identity(5);
        kernel = convolver._rotate_ccw(kernel, np.pi/2);
        pixel_x = 5;
        pixel_y = 5;
        P = tC._get_phantom_submatrix(kernel, pixel_x, pixel_y);
        print('\n');
        print(kernel);
        print('\n');
        print(tC.img_);
        print('\n');
        print(P);
        print('\n');
        cx = tC._convolve_pixel(kernel, pixel_x, pixel_y);
        print(cx);
        angl = tC._get_angle(10, 10, 10);
        print('The angle is ' + str(angl))
        sys.exit(0);
        
    shrinkage_test = False;
    if (shrinkage_test):
        T = circulant(np.arange(21));
        ignore = 5;
        tC = convolver(T, ignore);
        kernel = np.identity(11);
        kernel = convolver._rotate_ccw(kernel, np.pi/2);
        print('\n');
        print(kernel); print('\n');
        print(tC.img_); print('\n');
        new_kern = tC._shrink_kernel(kernel, 15, 8, 21);
        print(new_kern);
        sys.exit(0);
        
        
    flip_test = False;
    if (flip_test):
        M = np.array([[1,1,1],
             [1,1,0],
             [1,0,0]]);
             
        MM = convolver._flip_diag(M, 'NW');
        print('\n');
        print(M);
        print('\n')
        print(MM);
        sys.exit(0);
        
    
    mri = ld.mri_scan(MRI_PATH + SCAN);
    img = mri.get_image(107, 'x'); #crop_center = (0,0), crop_wh = (200, 200));

    cV = convolver(img, 16); # kernel size

    # Example of using display:
    #convolutions = cV._generate_random_convolutions(5, 16);
    #convolutions = cV._generate_random_lconv(6, 9);
    ###
    #cV.display(convolutions, 0, 9, show_kernel=True); # start index, number of images

    # Example of using display_single
    # DD5 = np.flip(DD4, axis=1);
    # print(repr(CC10));   
    #cV.display_single(CC13);
    
    CC17 = np.array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])
    
    test_var_conv = True;
    if (test_var_conv):
       # II = np.zeros([9,9]);
       # II[4,2] = 0.3333;  II[4,5] = 0.3333;  II[4,6] = 0.3333; 
       # print(II);
        NN, MM = cV.img_.shape;
        new_image = cV.var_convolve(CC17, cV._symmetric_precession, user_data = int(NN));
        fig, ax = plt.subplots(1, 2, figsize = (15, 8));
        ax[0].imshow(cV.img_);
        ax[1].imshow(new_image);
        plt.show();
        sys.exit(0);
        

    # CC17 = convolver._rotate_ccw(CC17, np.pi/3);
    cV.tuner(CC17);
    plt.show();
    
    print("hello");
    #a = np.random.randint(1,10) / 10;
    #m = Mask(30,30,[0,0],5, 0.03);
    #show_mask(generate_blend_mask(30,30,[15,15],3, 0.025));
    # mri = ld.mri_scan(MRI_PATH + SCAN);
    # img = mri.get_image(100, 'x');
    # imarr = generate_blended_distorted_images(img,10);

    # img = np.repeat(img[:, :, np.newaxis], 10, axis=2)

    # VRF = ipc.verifier(img, imarr);
    # VRF.initialize();
    # plt.show();




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



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:58:27 2018

@author: oscar seurat
"""

import matplotlib.pyplot as plt;
n = 4

images= [40, 46, 42, 48 ];

# images=[50, 51, 52]

fig, axarr = plt.subplots(2,n, figsize=(20,13));
for i in range(n):
    # display original
    axarr[0,i].imshow(mngl_arr_test[images[i],:,:,0]);
    axarr[1,i].imshow(restored_img[images[i],:,:,0]);

plt.show()
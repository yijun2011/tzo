#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:08:36 2018

@author: oscar seurat
"""

import matplotlib.pyplot as plt;
n = 4

fig, axarr = plt.subplots(2,n, figsize=(20,13));
for i in range(n):
    # display original
    axarr[0,i].imshow(mngl_arr_test[i+180,:,:,0]);
    axarr[1,i].imshow(restored_img[i+180,:,:,0]);

plt.show()


"""
n = 4
#plt.figure(figsize=(20, 13));
fig, axarr = plt.subplots(2,n, figsize=(20,13));
for i in range(n):
    # display original
    axarr[0,i].imshow(sample_train[i+44,:,:,0]);
    axarr[1,i].imshow(restored_train[i+44,:,:,0]);

plt.show()
"""
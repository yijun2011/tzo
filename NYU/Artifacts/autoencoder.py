#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:55:43 2018

@author: oscar seurat
"""

from keras.layers import Input, Conv2D, Dense;
from keras.models import Model, Sequential;
from keras import losses;
from keras import optimizers;
from keras.callbacks import ModelCheckpoint;
from keras.callbacks import EarlyStopping;
from keras.callbacks import ReduceLROnPlateau

import numpy as np;

import matplotlib.pyplot as plt;

import training_logger as tlg;

# Let's load data first
MRI_PATH='/Users/yzhao11/Documents/Research/MachineLearning/MRI/zhao_dataset_20181011/sub-NC188/ses-20180825/anat/';
        
# orig_file = MRI_PATH + 'OriginalImages_188_X.npz';
# mngl_file = MRI_PATH + 'DistortedImages_188_X.npz';

# orig_file = MRI_PATH + 'orig_test.npz';
# mngl_file = MRI_PATH + 'mngl_test.npz';

ORIG = np.load(orig_file);
orig_arr = ORIG['grand_orig_x'];

MNGL = np.load(mngl_file);
mngl_arr = MNGL['grand_arr_x'];

orig_arr = np.transpose(orig_arr, [2,0,1]);
mngl_arr = np.transpose(mngl_arr, [2,0,1]);

orig_arr = orig_arr[15000:50000,:,:];
mngl_arr = mngl_arr[15000:50000,:,:];

MX = np.amax(np.abs(orig_arr));
NX = np.amax(np.abs(mngl_arr));

orig_arr /= MX;
mngl_arr /= NX;

NN = orig_arr.shape[0];
NNx = int(0.85*NN);

"""
np.random.seed(7);
want    = np.random.randint(0, NN, NNx);
dntwant = np.ones(NN, dtype=bool);
dntwant[want]=False;


orig_arr_train = orig_arr[want   , :, :];
orig_arr_test  = orig_arr[dntwant, :,:];

"""


orig_arr_train = orig_arr[0:NNx, :, :];
orig_arr_test  = orig_arr[NNx: , :, :];  

a, b, c = orig_arr_train.shape;
orig_arr_train = np.reshape( orig_arr_train, [a, b, c, 1]);

a, b, c = orig_arr_test.shape;
orig_arr_test = np.reshape( orig_arr_test, [a, b, c, 1]);
    
"""
mngl_arr_train = mngl_arr[want   , :, :];
mngl_arr_test  = mngl_arr[dntwant, :, :]; 
"""
mngl_arr_train = mngl_arr[0:NNx, :, :];
mngl_arr_test  = mngl_arr[NNx: , :, :];  

a, b, c = mngl_arr_train.shape;
mngl_arr_train = np.reshape( mngl_arr_train, [a, b, c, 1]);

a, b, c = mngl_arr_test.shape;
mngl_arr_test = np.reshape(mngl_arr_test, [a, b, c, 1]);                

input_shape = (100, 100, 1);
filter0 = 2;
filter1 = 4;
filter2 = 8;
filter3 = 16;
filter4 = 1;
filter5 = 1;

th_layer0 = 4;
th_layer1 = 16;
th_layer2 = 64;
th_layer3 = 256;
th_layer4 = 256;

model = Sequential();

"""
model.add(Conv2D(th_layer0, kernel_size=(filter0,filter0), strides=(1,1), activation='relu',
                 padding='same', input_shape=input_shape));   
"""

model.add(Conv2D(th_layer1, kernel_size=(filter1,filter1), strides=(1,1), activation='relu',
                 padding='same', input_shape=input_shape));
  

model.add(Conv2D(th_layer2, kernel_size=(filter2, filter2), strides=(1,1), activation='relu',
                 padding='same'));
          
"""
model.add(Conv2D(th_layer3, kernel_size=(filter3, filter3), strides=(1,1), activation='relu',
                 padding='same'));
                 
model.add(Conv2D(th_layer4, kernel_size=(filter4, filter4), strides=(1,1), activation='relu',
                 padding='same'));
"""

model.add(Conv2D(1, kernel_size=(filter5, filter5), strides=(1,1), activation='relu',
                 padding='same'));
                 

#model.compile(loss=losses.mean_squared_error, optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)) ;

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.adam());

logger = tlg.training_logger();
cPoint = ModelCheckpoint(filepath='./Archive/M4_8_1_32b.hdf5', verbose=1, save_best_only=True)
eStop  = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=8, verbose=1,  restore_best_weights=True);

rPlat  = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=0.00001)

model.fit(mngl_arr_train, orig_arr_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(mngl_arr_test, orig_arr_test),
                callbacks=[logger, cPoint, eStop, rPlat]);
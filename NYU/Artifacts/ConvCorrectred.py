#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:44:47 2018

@author: oscar seurat
"""

import TrainingLogger as tlg;
from keras.callbacks import ModelCheckpoint;
from keras.callbacks import EarlyStopping;
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense;
from keras.models import Model, Sequential;
from keras import losses;
from keras import optimizers;

import ModelManager as MM;

DATA_DIR = '/Users/yzhao11/Documents/Research/MachineLearning/ArtifactsData/';
X_FILE   = DATA_DIR + 'mngl_test.npz';
Y_FILE   = DATA_DIR + 'orig_test.npz';

mm = MM.model_manager('ConvCorrectred', X_FILE, Y_FILE);
mm.set_logger();
mm.set_checkpoint();
mm.set_early_stopper();
mm.set_lr_reducer();

model = Sequential();
# Convolution layers
filter0 = 4; th_layer0 = 16;
filter1 = 9; th_layer1 = 81;
filter2 = 1; th_layer2 = 64;
filter3 = 5; th_layer3 = 25;
filter4 = 1; th_layer4 = 1;



model.add(Conv2D(th_layer0, kernel_size=(filter0,filter0), strides=(1,1), activation='relu',
                 padding='same', input_shape=mm.input_shape));   


model.add(Conv2D(th_layer1, kernel_size=(filter1,filter1), strides=(1,1), activation='relu',
                 padding='same'));
                          
                          
model.add(Conv2D(th_layer2, kernel_size=(filter2,filter2), strides=(1,1), activation='relu',
                 padding='same'));                         
                          
                           
model.add(Conv2DTranspose(th_layer3, kernel_size=(filter3,filter3), strides=(1,1), activation='relu',
                 padding='same'));                         
                          
                          
model.add(Conv2DTranspose(th_layer4, kernel_size=(filter4,filter4), strides=(1,1), activation='relu',
                 padding='same'));                           

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.adam());

mm.set_model(model);
mm.model.summary();

mm.run(30, 256);


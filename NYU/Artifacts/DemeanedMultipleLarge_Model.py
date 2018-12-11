#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:44:47 2018

@author: oscar seurat
"""

from keras.layers import Conv2D, Conv2DTranspose;
from keras.layers.normalization import BatchNormalization;
from keras.models import Sequential;
from keras import losses;
from keras import optimizers;

import ModelManager as MM;

DATA_DIR = '/home/ossowskj/NYU/ArtifactsData/';
X_FILE   = DATA_DIR + 'DistortedImages_188_X_Full_Scaled.npz';
Y_FILE   = DATA_DIR + 'OriginalImages_188_X_Full_Scaled.npz';

mm = MM.model_manager('DemeanedMultipleLarge_Model', X_FILE, Y_FILE, normalized=False);
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
model.add(BatchNormalization());


model.add(Conv2D(th_layer1, kernel_size=(filter1,filter1), strides=(1,1), activation='relu',
                 padding='same'));
model.add(BatchNormalization());                      
                          
model.add(Conv2D(th_layer2, kernel_size=(filter2,filter2), strides=(1,1), activation='relu',
                 padding='same'));                         
model.add(BatchNormalization());

                           
model.add(Conv2DTranspose(th_layer3, kernel_size=(filter3,filter3), strides=(1,1), activation='relu',
                 padding='same')); 
model.add(BatchNormalization());
                          
                          
model.add(Conv2DTranspose(th_layer4, kernel_size=(filter4,filter4), strides=(1,1), activation='relu',
                 padding='same'));
model.add(BatchNormalization());

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.adam());

mm.set_model(model);
mm.model.summary();

mm.run(20,8);


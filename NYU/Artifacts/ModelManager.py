#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 22:11:15 2018

@author: oscar seurat
"""

import os;
import sys;
import time;
import numpy as np;
import TrainingLogger as tlg;
from keras.callbacks import ModelCheckpoint;
from keras.callbacks import EarlyStopping;
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense;
from keras.models import Model, Sequential;
from keras import losses;
from keras import optimizers;

from pathlib import Path;
import cloudpickle as pickle;
import copy;

class model_manager:
    # MODEL_DIR='/Users/yzhao11/Documents/Research/MachineLearning/Models/';
    # MODEL_DIR='/u/erdos/yzhao11/NYU/Models/';
    MODEL_DIR= os.path.expanduser('~')+'/NYU/Models/';
    def __init__(self, name, x_file, y_file,
                 model_dir = MODEL_DIR,
                 train_frac = 0.7,
                 valid_frac = 0.15,
                 test_frac  = 0.15,
                 normalized  = False):
        
        x_path = Path(x_file);
        y_path = Path(y_file);
        if (not x_path.exists()): 
            print('model_manager.model_manager: the x input file ' + x_file +
                  ' does not exist.', sys.stderr);
            sys.exit(-1);
            
        if (not y_path.exists()):  
            print('model_manager.model_manager: the y input file ' + y_file +
                  ' does not exist.', sys.stderr);
            sys.exit(-1);           
        
        self.name = name;
        
        tstr = time.strftime("%Y%m%d_%H%M%S");
        dirpath = model_dir + name + '_' + tstr;
        try:  
            os.mkdir(dirpath)
        except OSError:
            print("model_manager.model_manager: failed to create directory " + 
                   dirpath + ".", sys.stderr); 
            sys.exit(-1);
        else:  
            print("model_manager.model_manager: created directory " + 
                   dirpath + ".", sys.stdout);
                  
        self.dirpath = dirpath + '/';
        
        try:
            symlink = x_file;
            dest    = self.dirpath + name + '_X';
            os.symlink(symlink, dest);
        except OSError:
            print("model_manager.model_manager: failed to create the symlink " + 
                   dest + " to " + symlink + ".", sys.stderr); 
            sys.exit(-1); 
        else:
            print("model_manager.model_manager: created the symlink " + 
                   symlink + ".", sys.stdout); 
                      
        try:
            symlink = y_file;
            dest = self.dirpath + name + '_Y';
            os.symlink(symlink, dest);
        except OSError:
            print("model_manager.model_manager: failed to create the symlink " + 
                   dest + " to " + symlink + ".", sys.stderr); 
            sys.exit(-1); 
        else:
            print("model_manager.model_manager: created the symlink " + 
                  dest + " to " + symlink + ".", sys.stdout);   
                  
        self.x_file = x_file;
        self.y_file = y_file;
        
        # Input training array
        ORIG = np.load(self.x_file);
        orig_arr = ORIG['grand_arr_x'];
        
        # Output training array
        DEST = np.load(self.y_file);
        dest_arr = DEST['grand_orig_x'];
        
        if (orig_arr.shape != dest_arr.shape):
            print('model_manager.model_manager: x input array and y input array ' + 
                  ' have different dimensions.', sys.stderr);
            sys.exit(-1);
        
        orig_arr = np.transpose(orig_arr, [2,0,1]);
        dest_arr = np.transpose(dest_arr, [2,0,1]);
        
        MX = np.amax(np.abs(orig_arr));
        NX = np.amax(np.abs(dest_arr));
        
        orig_arr /= MX;
        dest_arr /= NX;

        a, b, c = orig_arr.shape;
        self.mean_x = (-1)*np.ones([b,c]);

        a_, b_, c_ = dest_arr.shape;        
        self.mean_y = (-1)*np.ones([b_, c_]);
        
        NN = orig_arr.shape[0];
        NNa = int(train_frac * NN);
        NNb = int((train_frac + valid_frac) * NN)
        
        self.normalized = False;
        if (normalized == True):
            self.normalized = True;
            self.mean_x = np.mean(orig_arr[0:NNa,:,:], axis = 0);
            orig_arr -= self.mean_x;
            self.mean_y = np.mean(dest_arr[0:NNa,:,:], axis = 0);
            dest_arr -= self.mean_y;
            self.save_means();
 
        orig_arr = np.reshape(orig_arr, [a, b, c, 1]);
        dest_arr = np.reshape(dest_arr, [a_, b_, c_, 1]);
        
        # Process the input array 
        orig_arr_train = orig_arr[0:NNa,    :, :, :];
        orig_arr_valid = orig_arr[NNa:NNb , :, :, :];

        # Process the mangled array
        dest_arr_train = dest_arr[0:NNa,   :, :];
        dest_arr_valid  = dest_arr[NNa:NNb, :, :];
                          
        self.x_train = orig_arr_train;
        self.y_train = dest_arr_train;
        self.x_valid = orig_arr_valid;
        self.y_valid = dest_arr_valid;
        
        self.input_shape = (b, c, 1);
        
        self.logger        = None;
        self.checkpoint    = None;
        self.early_stopper = None;
        self.lr_reducer    = None;
        self.model         = None;
        
    def set_logger(self):
        self.logger = tlg.training_logger();
        
    def set_checkpoint(self):
        self.checkpoint = ModelCheckpoint(filepath=self.dirpath + self.name + '.hdf5',
                                          verbose=1,
                                          save_best_only=True);
    
    def set_early_stopper(self):
        self.early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.000003,
                                           patience=12, verbose=1,
                                           restore_best_weights=True);
                                          
    def set_lr_reducer(self):
        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.63, patience=5, verbose=1, min_lr=0.000001);
        
    def set_model(self, model):
        self.model = model;
        
    def run(self, epochs = 10, batch_size=256):
        if (self.model == None):
            print('model_manager.run: model is not set up. Cannot run.', sys.stderr);
            sys.exit(-1);
            
        callbacks = [self.logger, self.checkpoint, self.early_stopper, self.lr_reducer];
        callbacks = [x for x in callbacks if x is not None];
        
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_data=(self.x_valid, self.y_valid),
                       callbacks = callbacks);

    def save_means(self):
        file_name = self.dirpath + self.name + '_data_means.p';
        f = open(file_name, 'wb');
        pickle.dump([self.mean_x, self.mean_y], f);
        f.close();
        
    def save_yourself(self, no_data = True):
        file_name = self.dirpath + self.name + '_Model_Manager.p';
        f = open(file_name, 'wb');
        
        if (no_data == True):
            twin = copy.deepcopy(self);
            twin.x_train = [];
            twin.y_train = [];
            twin.x_valid = [];
            twin.y_valid = [];
        else:
            twin = self;  # shallow copy
        
        pickle.dump(twin, f);
        f.close();        
        
#############################################################################
#############################################################################
if 0:
    #DATA_DIR = '/home/ossowskj/NYU/ArtifactsData/';
    DATA_DIR = os.path.expanduser('~') + '/NYU/ArtifactsData/';
    X_FILE   = DATA_DIR + 'mngl_test.npz';
    Y_FILE   = DATA_DIR + 'orig_test.npz';
    
    mm_name = 'ToyModel';
    mm = model_manager(mm_name, X_FILE, Y_FILE, normalized=True);
    mm.set_logger();
    mm.set_checkpoint();
    mm.set_early_stopper();
    mm.set_lr_reducer();
    
    model = Sequential();
    filter0 = 8; th_layer0 = 32;
    filter1 = 8; th_layer1 = 1;
    
    model.add(Conv2D(th_layer0, kernel_size=(filter0,filter0), strides=(1,1), activation='relu',
                     padding='same', input_shape=mm.input_shape));   
    
    
    model.add(Conv2DTranspose(th_layer1, kernel_size=(filter1,filter1), strides=(1,1), activation='relu',
                     padding='same'));
    
    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.adam());
    
    mm.set_model(model);
    mm.run(10, 256);
    

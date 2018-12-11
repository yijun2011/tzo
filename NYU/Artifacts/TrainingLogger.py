#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:59:25 2018

@author: oscar seurat
"""

import keras;
import sys;

class training_logger(keras.callbacks.Callback):
    def __init__(self):
        self.batch_no = 0;
        self.losses = [];
        pass;
        
    def on_batch_begin(self, epoch, logs={}):
        self.batch_no += 1;
        print('\n\n * training_logger: starting batch number ' +
              str(self.batch_no) + '\n');
        
    def on_batch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'));
        print('\n\n * trainig_logger: the loss at the end of batch ' +
              str(self.batch_no) + ' is ' + str(self.losses[-1]) + '\n');
              
        
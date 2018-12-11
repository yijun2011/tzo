#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:02:52 2018

@author: oscar seurat
"""

import matplotlib.pyplot as plt;
from keras.models import Model;
from keras.models import load_model

from matplotlib.widgets import TextBox;
import numpy as np;
import Transformer as tr;
import cloudpickle as pickle;
import sys;

class res_browser:
    def __init__(self, model, x_test, y_test, normalized = False,
                 mean_img_x = [], mean_img_y = []):
        
        if (x_test.shape != y_test.shape):
            err_msg = 'ResBrowser.ResBrowser: the x, y data have incorrect dimensions.';
            raise ValueError(err_msg);
        
        MX = np.amax(np.abs(x_test));
        MY = np.amax(np.abs(y_test));
        
        self.x_test = x_test/MX;
        self.y_test = y_test/MY; 
        
        self.idxB = 0;
        self.idxE = x_test.shape[0];
        
        self.current_slice = 0;
            
        self.model  = model;
        if (not isinstance(model, Model)):
            err_msg = 'ResBrowser.ResBrowser: the model passed in must be a Keras Model.';
            raise ValueError(err_msg);
            
        if (normalized == True and (len(mean_img_x)==0 or len(mean_img_y)==0)):
            err_msg1 = 'ResBrowser.ResBrowser: when working with normalized ';
            err_msg2 = 'data, the x and y means must be given.';
            raise ValueError(err_msg1 + err_msg2);
            
        self.normalized = normalized;
        self.mean_img_x = mean_img_x;
        self.mean_img_y = mean_img_y;
            
        p = self.current_slice;
        if (normalized == True):
            self.y_model = self.predict_from_demeaned(self.x_test,
                                                     mean_img_x,
                                                     mean_img_y,
                                                     p);
        else:
            self.y_model = self.model.predict(self.x_test[p:(p+1),:,:,:]);
        
        # Let's set up image display gear
        self.fig, self.ax = plt.subplots(1, 3, figsize = (18, 8));
        self.fig.tight_layout();
        plt.subplots_adjust(bottom=0.10);
        
        self.ax[0].imshow(self.x_test[self.current_slice, :, :, 0]);
        self.ax[0].set_title('Input Image');
        self.ax[1].imshow(self.y_model[0,:,:,0]);
        self.ax[1].set_title('Model Output');
        self.ax[2].imshow(self.y_test[self.current_slice, :, :, 0]);
        self.ax[2].set_title('Expected Imge');
        
        
        img_no_pos = plt.axes([0.48, 0.08, 0.04, 0.04]); # x, y, w, h
        label      = 'Image No.:   \nOut of ' + str(self.idxE) + '  ';
        self.img_tbox = TextBox(img_no_pos, label, initial=str(self.current_slice));
       
    def _update(self, val):
        p = int(val);
        self.current_slice = p;
        
        if (self.normalized == True):
            self.y_model = self.predict_from_demeaned(self.x_test,
                                                     self.mean_img_x,
                                                     self.mean_img_y,
                                                     p);
        else:
            self.y_model = self.model.predict(self.x_test[p:(p+1),:,:,:]);
        
        image_x = self.x_test[self.current_slice,:,:, 0];
        self.ax[0].imshow(image_x);
        
        model_img_raw = self.y_model[0,:,:,0];
        # model_img     = tr.Transformer.hist_match(model_img_raw, image_x);
        self.ax[1].imshow(model_img_raw);
        
        self.ax[2].imshow(self.y_test[self.current_slice,:,:,0]);       
        self.fig.canvas.draw_idle();
        
    def initialize(self):
        self.img_tbox.on_submit(self._update);
        
    def predict_from_demeaned(self,
                              test_array_x,
                              mean_image_x,
                              mean_image_y,
                              slice_no):
        # keras' predict needs a 4 dimensional array 
        
        mu_image_x = mean_image_x.copy();
        shape      = mu_image_x.shape;
        new_shape  = np.concatenate( ( (1,), shape, (1,) ) );
        mu_image_x = mu_image_x.reshape(new_shape);
        
        mu_image_y = mean_image_y.copy();
        mu_image_y = mu_image_y.reshape(new_shape);

        image = test_array_x[slice_no, :, :, 0];
        image = image.reshape(new_shape);
        
        diff      = image - mu_image_x;
        diff_pred = self.model.predict(diff[0:1, :, :, :]);
        pred      = diff_pred[0,:,:, 0] + mu_image_y[0,:,:,0];
        pred      = pred.reshape(new_shape);
        return pred;
        
        
###################################################################
###################################################################

if 1:
    # MODEL_ARCHIVE = '/Users/yzhao11/Documents/Research/MachineLearning/Models/ToyModel_20181125_121929/';
    MODEL_ARCHIVE = '/Users/yzhao11/Documents/Research/MachineLearning/Models/DemeanedMultipleLarge_Model_20181130_001513/';
    MODEL_NAME    ='DemeanedMultipleLarge_Model';
    NORMALIZED    = True;
    
    MODEL_FILE   = MODEL_NAME + '.hdf5';
    
    loaded_model = load_model(MODEL_ARCHIVE + MODEL_FILE);
    loaded_model.summary();
    
    MRI_PATH=MODEL_ARCHIVE;
    y_file = MRI_PATH + MODEL_NAME + '_Y';
    x_file = MRI_PATH + MODEL_NAME + '_X';
    
    ORIG = np.load(y_file);
    y_test = ORIG['grand_orig_x'];
    
    MNGL = np.load(x_file);
    x_test = MNGL['grand_arr_x'];
    
    x_test= np.transpose(x_test, [2,0,1]);
    y_test = np.transpose(y_test, [2,0,1]);
    
    a, b, c = x_test.shape;
    x_test  = np.reshape( x_test, [a, b, c, 1]);
    
    a, b, c = y_test.shape;
    y_test = np.reshape( y_test, [a, b, c, 1]);
    

    # Get the mean image files
    mean_img_x = [];
    mean_img_y = [];
    
    if (NORMALIZED == True):
        MEANS_FILE = MODEL_NAME + '_data_means.p';
        MEANS_PATH = MODEL_ARCHIVE + MEANS_FILE;
        try:
            ff = open(MEANS_PATH, 'rb');
        except:
            print('ResBrowswer: failed to open the file ' + MEANS_PATH + '.' +
                  ' Attempting to run without the image means.',
                  sys.stderr);
            NORMALIZED = False;
        else:
            mean_img_x, mean_img_y = pickle.load(ff);
            ff.close();
    
    rB = res_browser(loaded_model, x_test, y_test,
                     normalized = NORMALIZED,
                     mean_img_x=mean_img_x,
                     mean_img_y=mean_img_y);
    rB.initialize();
    plt.show();
    
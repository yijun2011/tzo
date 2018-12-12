#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed. Dec 05, 2018

@author: oscar seurat
"""
import sys;
import numpy as np;

FILE_NAME = sys.argv[1];
START     = sys.argv[2];
END       = sys.argv[3];

print('extract_part: getting the file ' + FILE_NAME + '. Extracting slices ' + 
      START + ' through ' + END + ' (python 0-based notation).', sys.stdout);

START = int(START);
END   = int(END);

if (START >= END):
    print('extract_part: the starting index must be < than the ending one.',
          sys.stderr);
    sys.exit(-1);

x_file = FILE_NAME;

MNGL = np.load(x_file);
x_test = MNGL['grand_arr_x'];

[a, b, c] = x_test.shape;
x_test = x_test[:, :, START:END];

NEW_FILE_NAME = FILE_NAME + '_' + str(START)+ '_' + str(END) + '.npz';
np.savez_compressed(NEW_FILE_NAME, grand_arr_x = x_test);
        
 
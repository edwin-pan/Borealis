# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:00:26 2020

@author: Edwin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import string

def labelmaker(fname):
    label = ''
    for char in fname:
        if string.ascii_lowercase.count(char) or string.ascii_uppercase.count(char):
            label+=char
        else:
            break
    return label


if __name__ == '__main__':
    print('----- Unit Test for rdc2_to_png -----')
    
    # Grab current repository directory
    repo_dir = os.path.abspath(os.getcwd())[:-8]
    
    # Grab png & processed directory
    data_processed_dir = os.path.join(repo_dir, 'data\\processed')
    data_png_dir = os.path.join(repo_dir, 'data\\png')
    
    # Create directories if they don't exist
    if not os.path.exists(data_png_dir):
        os.makedirs(data_png_dir)
    
    # Grab all files
    filenames_all = next(os.walk(data_processed_dir))[2] # Grab all scan directories
    filenames = []
    classes = []
    classes_dir = []
    data = []
    for idx, fname in enumerate(filenames_all):
        if fname.endswith('.npy'):
            filenames.append(fname)
            
            # Load these data files
            local_data = np.load(os.path.join(data_processed_dir, fname))
            local_data = np.abs(local_data)
            
            # Get rid of Zero Doppler
            local_data[:,local_data.shape[1]//2] = np.min(local_data)
            
            # Normalize
            print(fname + ": " + str(np.max(local_data)))
            local_data = local_data / np.max(local_data)
            local_data *= 255
            local_data = np.clip(local_data, 0, 255)

            data.append(local_data)
            
            # Check label
            label = labelmaker(fname)
            class_dir = os.path.join(data_png_dir, label)
            if label not in classes:
                classes.append(label)
                
                # Create directory
                classes_dir.append(class_dir)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                
            
            # Save to corresponding directory
            plt.imsave(os.path.join(class_dir,fname[:-4]+'.png'), np.abs(data[-1]))
            

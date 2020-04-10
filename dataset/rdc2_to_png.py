# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:00:26 2020

@author: Edwin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import string
import cv2
import mmwave.dsp as dsp

plt.close('all')

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
    threshold = 180
    
    for idx, fname in enumerate(filenames_all):
        if fname.endswith('.npy'):
            filenames.append(fname)
            
            # Load these data files
            local_data = np.load(os.path.join(data_processed_dir, fname))
            local_data = np.abs(local_data)
            
            # Get rid of Zero Doppler
            local_data[:,local_data.shape[1]//2] = np.min(local_data)
            
            # Perform CFAR
            fft2d_sum = local_data.astype(np.int64)
            thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                      axis=0,
                                                                      arr=fft2d_sum.T,
                                                                      l_bound=1.5,
                                                                      guard_len=32,
                                                                      noise_len=128)
        
            thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum,
                                                                  l_bound=2.5,
                                                                  guard_len=16,
                                                                  noise_len=64)
        
            thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
            det_doppler_mask = (local_data > thresholdDoppler)
            det_range_mask = (local_data > thresholdRange)
        
            # Get indices of detected peaks
            full_mask = (det_doppler_mask & det_range_mask)
            det_peaks_indices = np.argwhere(full_mask == True)
        
            # peakVals and SNR calculation
            peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
            snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
                    
            # Apply mask
            fft2d_sum[~full_mask] = 0
            
            local_data = fft2d_sum
            
            # Normalize
            local_data = local_data / np.max(local_data)
            local_data *= 255
            local_data = np.clip(local_data, 0, 255)
            local_data = local_data.astype(np.uint8)
            
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            local_data = clahe.apply(local_data)

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
            print("Saving", fname)
            plt.imsave(os.path.join(class_dir,fname[:-4]+'.png'), np.abs(data[-1]))


# =============================================================================
# BELOW THIS LINE IS DEBUG CODE, COMMENTED OUT. NOT FOR GENERAL USE
# =============================================================================
    # print("debug")

    # local_data = np.load('C:\\Users\\Edwin\\Documents\\UIUC_Spring_2020\\ECE-497-Thesis\\borealis\\data\\processed\\driving0_1.npy')
    # local_data = np.abs(local_data)
    # local_data[:,local_data.shape[1]//2] = np.min(local_data)
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(local_data)
    # plt.title("pre CFAR")
    
    
    # fft2d_sum = local_data.astype(np.int64)
    # thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
    #                                                           axis=0,
    #                                                           arr=fft2d_sum.T,
    #                                                           l_bound=1.5,
    #                                                           guard_len=32,
    #                                                           noise_len=128)

    # thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
    #                                                       axis=0,
    #                                                       arr=fft2d_sum,
    #                                                       l_bound=2.5,
    #                                                       guard_len=16,
    #                                                       noise_len=64)

    # thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
    # det_doppler_mask = (local_data > thresholdDoppler)
    # det_range_mask = (local_data > thresholdRange)

    # # Get indices of detected peaks
    # full_mask = (det_doppler_mask & det_range_mask)
    # det_peaks_indices = np.argwhere(full_mask == True)

    # # peakVals and SNR calculation
    # peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    # snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
            
    # # Apply mask
    # fft2d_sum[~full_mask] = 0
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(fft2d_sum)
    # plt.title("post CFAR")

    # check = np.where(full_mask == False)[0]
    # print(len(check))
    
    # # Normalize
    # fft2d_sum = fft2d_sum / np.max(fft2d_sum)
    # fft2d_sum *= 255
    # fft2d_sum = np.clip(fft2d_sum, 0, 255)
    # fft2d_sum = fft2d_sum.astype(np.uint8)
    
    # # Contrast Limited Adaptive Histogram Equalization
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # fft2d_sum = clahe.apply(fft2d_sum)
    
    # plt.figure(figsize=(15,15))
    # plt.imshow(fft2d_sum)
    # plt.title("post Histogram Equalization")
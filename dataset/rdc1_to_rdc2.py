# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 05:16:08 2020

@author: Edwin
"""

import numpy as np
import scipy.signal
import scipy.io
import matplotlib.pyplot as plt
import os
import json 


def rdc1_to_rdc2(fpath, visualize_flag=False):
    """ Takes RDC1 data from the Uhnder Judo module and converts it to RDC2
    
    Args:
        fpath (string): Directions to the desired *_rdc1.bin scan file (ex: scan_000000_rdc1.bin)
        visualize_flag (bool, default=False): Flag if user wants to see the resulting plot
        
    Returns:
        rdc2 (np.array): A 2D numpy array of shape (num_range_bins, num_doppler_bins) 
    
    """
    
    parts = os.path.split(fpath)
    ppath = os.path.join(parts[0], parts[1][:-8]+'info.json')
    epath = os.path.join(parts[0], parts[1][:-4]+'exp.bin')
    rpath = os.path.join(parts[0], parts[1][:-8]+'range_bins.bin')
    
    print(ppath)
    # Load scan parameters
    with open(ppath) as json_file:
      params = json.load(json_file)
      
    range_resolution = params['range_bin_width'] # meters
    doppler_resolution = params['doppler_bin_width'] # meters/sec
    
    # Check that data is correct size
    data_size = os.path.getsize(fpath)
    assert data_size == 2*(2*params['total_vrx']*params['num_range_bins']*params['num_pulses']), "Input data needs to be the correct shape! (data_size=" + str(data_size) + " , correct=" + str(2*(2*params['total_vrx']*params['num_range_bins']*params['num_pulses']))
    
    data  = np.fromfile(fpath, np.int16) # data -> checkpoint1
    
    # Convert binary to complex data
    time_rdc = (data[0::2]+1j*data[1::2]).reshape((params['total_vrx'], params['num_range_bins'], params['num_pulses']), order='F').transpose(1,0,2) # -- checkpointN_0
    
    # BEGIN calcCorrExpFromMx
    RangeBins, NVrx, N = time_rdc.shape
    rangeExponentsMxAll = np.zeros((RangeBins,10))
    
    expData = np.fromfile(epath, np.ubyte)
    
    expData = np.unpackbits(expData, bitorder='little')
    
    for i in range(RangeBins):
        rangeExponentsMxAll[i, 0] = np.packbits(expData[128*i:(128*i)+4], bitorder='little')
        for j in range(8):
            fillval = np.packbits(expData[128*i+4+13*j:(128*i)+4+13*(j+1)], bitorder='little')
            rangeExponentsMxAll[i, j+1] = fillval[0]+256*fillval[1]
        rangeExponentsMxAll[i, 9] = np.packbits(expData[128*i+108:128*(i+1)], bitorder='little')[0]
    
    # rangeExponentsMxAll -- checkpointN_1
    baseRangeExponent = rangeExponentsMxAll[:,0]
    rangeExponentsMx = rangeExponentsMxAll[:,1:9]
    rangeExpReserved = rangeExponentsMxAll[:,9]
        
    rdc1expMx = np.zeros((RangeBins,N))
    for i in range(RangeBins):
        currBase = np.single(baseRangeExponent[i])
        currRangeExponents = np.ones((1,N))*currBase;
        validRexpCol = np.sum(rangeExponentsMx[i,:] > 0) # zeros are unused exponents
        
        if validRexpCol > 0:
            for itidx in range(validRexpCol):
                currRangeExponents[0, int(rangeExponentsMx[i,itidx]):] += 1
    
        expo = np.power(2, currRangeExponents)
        rdc1expMx[i,:] = expo
        expo = np.int32(np.tile(expo, [NVrx, 1]))
        
        rI = np.int32(np.real(time_rdc[i, :, :]))
        rQ = np.int32(np.imag(time_rdc[i, :, :]))
    
    
        if NVrx == 1  or N==1:
            rI = np.transpose(rI)
            rQ = np.transpose(rQ)
            
        time_rdc[i, :, :] = rI*expo + 1j*rQ*expo
    
    # END calcCorrExpFromMx
        
    # Valid Range Bin Masking
    rawRangeBins = np.fromfile(rpath, np.int16)
    rawRangeBins = rawRangeBins[rawRangeBins>=0]
    rangeBins = np.sort(rawRangeBins)
    rangeGateOrdering = np.argsort(rawRangeBins)
    rangeGateMid = rangeBins*params['range_bin_width']
    
    validrbinmask = rangeGateMid >= 0
    
    time_rdc = time_rdc[rangeGateOrdering[validrbinmask],:,:]*np.power(2,params['rdc1_software_exponent'])
    
    
    # Begin processing (RDC1->RDC2)
    window = scipy.signal.windows.nuttall(params['num_pulses'])
    rdc2 = np.fft.fft(time_rdc, axis=-1)
    rdc2 = np.fft.fftshift(rdc2, axes=-1)
    rdc2 *= window
    rdc2 = np.sum(rdc2, axis=1)
    
    if visualize_flag:
        plt.imshow(np.abs(rdc2), origin='lower',extent=(-rdc2.shape[1]*doppler_resolution/2,rdc2.shape[1]*doppler_resolution/2,0,range_resolution*rdc2.shape[0]))
    
    return rdc2


if __name__ == '__main__':
    print('----- Unit Test for rdc1_to_rdc2 -----')
    filepath = 'C:\\Users\\Edwin\\Documents\\PreSense\\uDoppler-Classification\\data\\raw\\Judo\\20200331\\192.168.95.1_20200331105730_3.7meters'    
    
    savepath = 'C:\\Users\\Edwin\\Documents\\PreSense\\uDoppler-Classification\\data\\processed'
    
    files_per_dir = 4
    scanned_person = 'YLC'
    
    for i in range(files_per_dir):
        # Process the data
        data = rdc1_to_rdc2(os.path.join(filepath, 'scan_00000' + str(i) + '_rdc1.bin'), visualize_flag=True)

        # Save the data
        np.save(os.path.join(savepath, scanned_person+str(i)), data)
    
    
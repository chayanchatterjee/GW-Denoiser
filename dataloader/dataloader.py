# -*- coding: utf-8 -*-
"""Data Loader"""

from SampleFileTools1 import SampleFile
import pandas as pd
import h5py
import sys

class DataLoader:
    """Data Loader class"""
    
    def __init__(self, det, data):
        
        self.det = det
        self.data = data
       
    def load_data(data_config):
        """Loads dataset from path"""
        
        # Check training or testing data
        if(self.data == 'train'):
            df = h5py.File('../BBH_sample_files/default_snr.hdf', 'r')
        elif(self.data == 'test'):
            df = h5py.File('../BBH_sample_files/default_snr-20_test.hdf', 'r')
        
        # Obtain data for a given detector
        if(self.det == 'Hanford'):
            strain = df['injection_samples']['h1_strain']
            signal = df['injection_samples']['h1_signal']
            
        elif(self.det == 'Livingston'):
            strain = df['injection_samples']['l1_strain']
            signal = df['injection_samples']['l1_signal']
            
        elif(self.det == 'Virgo'):
            strain = df['injection_samples']['v1_strain']
            signal = df['injection_samples']['v1_signal']
            
        else:
            sys.exit('Detector not available. Quitting.')
        
        df.close()
        
        return strain, signal
    
    
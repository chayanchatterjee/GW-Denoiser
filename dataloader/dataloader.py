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
       
    def load_data(self, data_config):
        """Loads dataset from path"""
        
        # Check training or testing data
        if(self.data == 'train'):
            df = h5py.File(data_config.path_train, 'r')
        elif(self.data == 'test'):
            df = h5py.File(data_config.path_test, 'r')
        
        # Obtain data for a given detector
        if(self.det == 'Hanford'):
            strain = df['injection_samples']['h1_strain'][0:10]
            signal = df['injection_parameters']['h1_signal'][0:10]
            
        elif(self.det == 'Livingston'):
            strain = df['injection_samples']['l1_strain'][0:10]
            signal = df['injection_parameters']['l1_signal'][0:10]
            
        elif(self.det == 'Virgo'):
            strain = df['injection_samples']['v1_strain'][0:10]
            signal = df['injection_parameters']['v1_signal'][0:10]
            
        else:
            sys.exit('Detector not available. Quitting.')
        
        df.close()
        
        return strain, signal
    
    
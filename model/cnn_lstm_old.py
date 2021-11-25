# -*- coding: utf-8 -*-
"""CNN-LSTM model"""

''' 
 * Copyright (C) 2021 Chayan Chatterjee <chayan.chatterjee@research.uwa.edu.au>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
'''

# standard library

# internal
from __future__ import print_function
from .base_model import BaseModel
from SampleFileTools1 import SampleFile
#from dataloader.dataloader import DataLoader

# external
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns

#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from numpy import array


import tensorflow as tf
#tf.enable_eager_execution()

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
           devices=devices_names[:n_gpus])


import numpy as np
import pandas as pd

from scipy import signal
import random as ran


class CNN_LSTM(BaseModel):
    """CNN_LSTM Model Class"""
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.train.batch_size
        self.epoches = self.config.train.epoches
        
        
    def load_data(self):
        """Loads and Preprocess data """
        obj = SampleFile()
        obj.read_hdf("../../BBH_sample_files/default_training_1_sec_10to80_1.hdf")
        self.train_dataset = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

        obj_test_1 = SampleFile()
        obj_test_1.read_hdf("../../Real_events/default_GW170104_O2_PSD_1_sec.hdf")
        df_test_1 = obj_test_1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_2 = SampleFile()
        obj_test_2.read_hdf("../../Real_events/default_GW170729_O2_PSD_1_sec.hdf")
        df_test_2 = obj_test_2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_3 = SampleFile()
        obj_test_3.read_hdf("../../Real_events/default_GW170809_O2_PSD_1_sec.hdf")
        df_test_3 = obj_test_3.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_4 = SampleFile()
        obj_test_4.read_hdf("../../Real_events/default_GW170814_O2_PSD_1_sec.hdf")
        df_test_4 = obj_test_4.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_5 = SampleFile()
        obj_test_5.read_hdf("../../Real_events/default_GW170818_O2_PSD_1_sec.hdf")
        df_test_5 = obj_test_5.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_6 = SampleFile()
        obj_test_6.read_hdf("../../Real_events/default_GW170823_O2_PSD_1_sec.hdf")
        df_test_6 = obj_test_6.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_7 = SampleFile()
        obj_test_7.read_hdf("../../Real_events/default_GW170608_O2_PSD_1_sec.hdf")
        df_test_7 = obj_test_7.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_8 = SampleFile()
        obj_test_8.read_hdf("../../Real_events/default_GW150914_O2_PSD_1_sec.hdf")
        df_test_8 = obj_test_8.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj_test_9 = SampleFile()
        obj_test_9.read_hdf("../../Real_events/default_GW151012_O1_PSD.hdf")
        df_test_9 = obj_test_9.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

        obj_test_10 = SampleFile()
        obj_test_10.read_hdf("../../Real_events/default_GW151226_O1_PSD.hdf")
        df_test_10 = obj_test_10.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

        self.test_dataset = pd.concat([df_test_1, df_test_2, df_test_3, df_test_4, df_test_5, df_test_6, df_test_7, df_test_8, df_test_9, df_test_10], ignore_index= True)


        
        #self.train_dataset = DataLoader().load_train_data(self.config.data)
        #self.test_dataset = DataLoader().load_test_data(self.config.data)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """ Normalizes training and test set signals """
        def normalize_noisy(a):
            new_array = []
            for i in range(len(a)):
                dataset = a[i]
                dataset = dataset/10
                new_array.append(dataset)
            return new_array

        def normalize_pure(a):
            new_array = []
            for i in range(50000):
                dataset = a[i]
                dataset = dataset/1e-22
                new_array.append(dataset)
            return new_array

        def normalize_test_noisy(a):
            new_array = []
            for i in range(10):
                dataset = a[i]
                dataset = dataset/10
                new_array.append(dataset)
            return new_array

        def normalize_test_pure(a):
            new_array = []
            for i in range(10):
                dataset = a[i]
                dataset = dataset/1e-22
                new_array.append(dataset)
            return new_array

        def normalize_new(a):
            new_array = []
            for i in range(50000):
                dataset = a[i]
                dataset = dataset[1536:2048]
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                for j in range(512):
                    if(dataset[j] > 0):
                        dataset[j] = dataset[j]/maximum
                    else:
                        dataset[j] = dataset[j]/minimum
        #        dataset = dataset+gauss
                new_array.append(dataset)
            return new_array

        def normalize_glitch_new(a):
            new_array = []
            for i in range(50000):
                dataset = a[i]
                dataset = dataset[1536:2048]
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                for j in range(512):
                    if(dataset[j] > 0):
                        dataset[j] = dataset[j]/maximum
                    else:
                        dataset[j] = dataset[j]/minimum
            #        dataset = dataset + glitch(10)
                new_array.append(dataset)
            return new_array
        
        def normalize_test_glitch_new(a):
            new_array = []
            for i in range(10):
                dataset = a[i]
                dataset = dataset[1536:2048]
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                for j in range(512):
                    if(dataset[j] > 0):
                        dataset[j] = dataset[j]/maximum
                    else:
                        dataset[j] = dataset[j]/minimum
            #        dataset = dataset+ glitch(10)
                new_array.append(dataset)
            return new_array

        def normalize_test_new(a):
            new_array = []
            for i in range(10):
                dataset = a[i]
                dataset = dataset[1536:2048]
                maximum = np.max(dataset)
                minimum = np.abs(np.min(dataset))
                for j in range(512):
                    if(dataset[j] > 0):
                        dataset[j] = dataset[j]/maximum
                    else:
                        dataset[j] = dataset[j]/minimum
        #        dataset = dataset+gauss
                new_array.append(dataset)
            return new_array
        
#new = df[['h1_signal', 'l1_signal', 'v1_signal']].copy() # Extracting h1_signal, l1_signal and v1_signal columns
                                                         # from hdf file.    
        new = self.train_dataset[['h1_strain']].copy() 
        new_test = self.test_dataset[['h1_strain']].copy()

        new_pure = self.train_dataset[['h1_signal']].copy() 
        new_test_pure = self.test_dataset[['h1_signal']].copy()

        h1 = new.iloc[:,0] #Extracting h1_signal from 'new' dataframe
        #l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe

        #v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
        h1_test = new_test.iloc[:,0]

        h1_pure = new_pure.iloc[:,0] #Extracting h1_signal from 'new' dataframe
        #l1 = new.iloc[:,1] #Extracting l1_signal from 'new' dataframe
        #v1 = new.iloc[:,2] #Extracting v1_signal from 'new' dataframe
        h1_test_pure = new_test_pure.iloc[:,0]

#h1_test = []
#h1_test_pure = []
#for i in range(4000):
#    a = df_test['mass1']
#    b = df_test['mass2']
#    if(a[i] >= 1.7 and a[i] <= 4.0 and b[i] >= 1.7 and b[i] <= 4.0):
#        h1_test.append(df_test['v1_strain'][i])
#        h1_test_pure.append(df_test['v1_signal'][i])
        
#h1_test = np.asarray(h1_test)
#h1_test_pure = np.asarray(h1_test_pure)

        h1_new = normalize_noisy(h1)
     
        h1_test_new = normalize_test_noisy(h1_test)
        h1_pure_new = normalize_pure(h1_pure)
        h1_test_pure_new = normalize_test_pure(h1_test_pure)
        
        

# For no glitch

#h1_glitch_new = h1_new
#h1_test_glitch_new = h1_test_new

# Only for plotting and comparing
#h1_test_glitch_pure_new = h1_test_pure_new


#Uncomment this for glitch:

        self.h1_new = normalize_new(h1_new)
        self.h1_test_new = normalize_test_new(h1_test_new)

        self.h1_pure_new = normalize_new(h1_pure_new)
        self.h1_test_pure_new = normalize_test_new(h1_test_pure_new)
        
        self.reshape_training_set()
        

# Only for plotting and comparing. Uncomment this for glitch
#h1_test_glitch_pure_new = normalize_test_glitch_new(h1_test_pure_new)

# split a univariate sequence into samples
    def split_sequence(self,sequence_noisy,sequence_pure,n_steps):
        X, y = list(), list()
        for i in range(len(sequence_noisy)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence_noisy)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence_noisy[i:end_ix], sequence_pure[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    
    def reshape_training_set(self):
        n_steps = 4
        self.X_train_noisy = []
        self.X_train_pure = []

        for i in range(50000):
            X_noisy = self.h1_new[i]
            X_pure = self.h1_pure_new[i]
            X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
            X_pure = np.pad(X_pure, (4, 4), 'constant', constant_values=(0, 0))
            # split into samples
            X, y = self.split_sequence(X_noisy, X_pure, n_steps)
            self.X_train_noisy.append(X)
            self.X_train_pure.append(y)
    
        self.X_train_noisy = np.asarray(self.X_train_noisy)
        self.X_train_pure = np.asarray(self.X_train_pure)
        
        self.reshape_test_set()
        
    def reshape_test_set(self):
        self.X_test_noisy = []
        self.X_test_pure = []
        n_steps = 4
       
        for i in range(10):
            X_noisy_test = self.h1_test_new[i]
            X_pure_test = self.h1_test_pure_new[i]
            X_noisy_test = np.pad(X_noisy_test, (4, 4), 'constant', constant_values=(0, 0))
            X_pure_test = np.pad(X_pure_test, (4, 4), 'constant', constant_values=(0, 0))
            # split into samples
            X, y = self.split_sequence(X_noisy_test, X_pure_test, n_steps)
            self.X_test_noisy.append(X)
            self.X_test_pure.append(y)
    
        self.X_test_noisy = np.asarray(self.X_test_noisy)
        self.X_test_pure = np.asarray(self.X_test_pure)
        
        self.reshape_for_Keras()
        
    def reshape_for_Keras(self):
        num_time_periods, num_sensors = 4, 1
        num_time_periods_test, num_sensors_test = 4, 1

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
        input_shape = (num_time_periods*num_sensors)
        input_shape_test = (num_time_periods_test*num_sensors_test)

        self.X_train_noisy = self.X_train_noisy.reshape(self.X_train_noisy.shape[0], 516, input_shape)
        self.X_test_noisy = self.X_test_noisy.reshape(self.X_test_noisy.shape[0], 516, input_shape)
        self.X_train_pure = self.X_train_pure.reshape(self.X_train_pure.shape[0], 516, 1)
        self.X_test_pure = self.X_test_pure.reshape(self.X_test_pure.shape[0], 516, 1)
        
        
        print('x_train_noisy shape:', self.X_train_noisy.shape)

        print('input_shape:', input_shape)


        print('x_test_noisy shape:', self.X_test_noisy.shape)

        print('input_shape:', input_shape_test)


        print('x_train_pure shape:', self.X_train_pure.shape)

        print('input_shape:', input_shape)


        print('x_test_pure shape:', self.X_test_pure.shape)
        print('input_shape:', input_shape_test)



        # Convert type for Keras otherwise Keras cannot process the data
        self.X_train_noisy = self.X_train_noisy.astype("float32")

        self.X_test_noisy = self.X_test_noisy.astype("float32")

        # Convert type for Keras otherwise Keras cannot process the data
        self.X_train_pure = self.X_train_pure.astype("float32")

        self.X_test_pure = self.X_test_pure.astype("float32")

        
    from keras import backend as K
    def fractal_tanimoto_loss(self, y_true, y_pred, depth=0, smooth=1e-6):
        x = y_true
        y = y_pred
#    x_norm = K.sum(x)
#    y_norm = K.sum(y)
#    x = x/x_norm
#    y = y/y_norm
        depth = depth+1
        scale = 1./len(range(depth))
    
        def inner_prod(y, x):
            prod = y*x
            prod = K.sum(prod, axis=1)
        
            return prod
    
        def tnmt_base(x, y):

            tpl  = inner_prod(y,x)
            tpp  = inner_prod(y,y)
            tll  = inner_prod(x,x)


            num = tpl + smooth
            denum = 0.0
            result = 0.0
            for d in range(depth):
                a = 2.**d
                b = -(2.*a-1.)

                denum = denum + tf.math.reciprocal( a*(tpp+tll) + b *tpl + smooth)
    #            denum = ( a*(tpp+tll) + b *tpl + smooth)
    #            result = K.mean((result + (num/denum)), axis=0)

            result =  num * denum * scale

            return  result*scale
    
    
        l1 = tnmt_base(x,y)
#        l2 = self.tnmt_base(1.-preds, 1.-labels)

#        result = 0.5*(l1+l2)
        result = l1
    
        return  1. - result

    def build(self):
        """ Builds the Keras model based """
        from keras.models import load_model
 
    # load model
        self.model = load_model('../Models/model_FractalTanimoto.h5', custom_objects={'fractal_tanimoto_loss': self.fractal_tanimoto_loss})
#model = load_model('Models/model_GW170608.h5', custom_objects={'correlation_coefficient_loss': correlation_coefficient_loss})
# summarize model.
        self.model.summary()

    
    def train(self):
        """Compiles and trains the model"""
#        self.model.compile(optimizer=self.config.train.optimizer.type,
#                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.X_train_noisy, self.X_train_pure, epochs=2, batch_size=self.batch_size,
                                       validation_data=(self.X_test_noisy,self.X_test_pure))

        return model_history.history['loss'], model_history.history['val_loss']
                                       
                                       
    def evaluate(self):
        """Predicts resuts for the test dataset"""
        predictions = []
        predictions.append(self.model.predict(self.X_test_noisy))
#decoded_signals = decoded_signals.reshape(-1,512,1)
#decoded_signals = normalize_decoded_new(decoded_signals)

        score = self.model.evaluate(self.X_test_noisy, self.X_test_pure, verbose=1)

        print('\nAccuracy on test data: %0.2f' % score[1])
        print('\nLoss on test data: %0.2f' % score[0])

        return predictions 
    
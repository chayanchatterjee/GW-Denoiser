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
from .base_model import BaseModel
from SampleFileTools1 import SampleFile
from dataloader.dataloader import DataLoader

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
        self.num_train = self.config.train.num_training_samples
        self.num_test = self.config.train.num_test_samples
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epoches
        
        
    def load_data(self):
        """Loads and Preprocess data """
        self.train_dataset, self.test_dataset = DataLoader().load_data(self.config.data)
        
        #self.train_dataset = DataLoader().load_train_data(self.config.data)
        #self.test_dataset = DataLoader().load_test_data(self.config.data)
        
        # Load data
        h1 = self.train_dataset['h1_strain'] 
        h1_test = self.test_dataset['h1_strain']

        h1_pure = self.train_dataset['h1_signal']
        h1_test_pure = self.test_dataset['h1_signal']
        
        # Pre-process data
        self.h1_new = self._preprocess_data(h1)
        self.h1_test_new = self._preprocess_data(h1_test)

        self.h1_pure_new = self._preprocess_data(h1_pure)
        self.h1_test_pure_new = self._preprocess_data(h1_test_pure)
        
        # Reshape data
        self.X_train_noisy, self.X_train_pure = self.reshape_sequences(self.num_train, self.h1_new, self.h1_pure_new)
        self.X_test_noisy, self.X_test_pure = self.reshape_sequences(self.num_test, self.h1_pure_new, self.h1_test_pure_new)
        
        # Reshape data for Keras
        self.reshape_and_print()
        
        
    def _preprocess_data(self, data):
        """ Normalizes training and test set signals """
        arr = []
        for i in range(len(data)):
            samples = data[i]
            samples = samples[1536:2048]
            maximum = np.max(samples)
            minimum = np.abs(np.min(samples))
            for j in range(512):
                if(samples[j] > 0):
                    samples[j] = samples[j]/maximum
                else:
                    samples[j] = samples[j]/minimum
            arr.append(samples)
        return arr
        
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
    
    
    def reshape_sequences(self, num, data_noisy, data_pure):
        n_steps = 4
        arr_noisy = []
        arr_pure = []
        
        for i in range(num):
            X_noisy = data_noisy[i]
            X_pure = data_pure[i]
            X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
            X_pure = np.pad(X_pure, (4, 4), 'constant', constant_values=(0, 0))
            # split into samples
            X, y = self.split_sequence(X_noisy, X_pure, n_steps)
            arr_noisy.append(X)
            arr_pure.append(y)
    
        arr_noisy = np.asarray(arr_noisy)
        arr_pure = np.asarray(arr_pure)
        
        return arr_noisy, arr_pure
    
        
    def reshape_and_print(self):
        
        self.X_train_noisy = self.X_train_noisy.reshape(self.X_train_noisy.shape[0], 516, 4, 1)
        self.X_test_noisy = self.X_test_noisy.reshape(self.X_test_noisy.shape[0], 516, 4, 1)
        self.X_train_pure = self.X_train_pure.reshape(self.X_train_pure.shape[0], 516, 1)
        self.X_test_pure = self.X_test_pure.reshape(self.X_test_pure.shape[0], 516, 1)
        
        print('x_train_noisy shape:', self.X_train_noisy.shape)
        print('x_test_noisy shape:', self.X_test_noisy.shape)
        print('x_train_pure shape:', self.X_train_pure.shape)
        print('x_test_pure shape:', self.X_test_pure.shape)

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
        with strategy.scope():
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32,1, padding='same'),input_shape=(516,4,1)))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(16,1, padding='same')))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
            
            optimizer = tf.keras.optimizers.Adam(lr=0.001)
            self.model.compile(optimizer=optimizer,loss='mse',metrics=['accuracy'])
    
        self.model.summary()
        
           
    def train(self):
        """Compiles and trains the model"""
    #        self.model.compile(optimizer=self.config.train.optimizer.type,
    #                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.X_train_noisy, self.X_train_pure, epochs=self.epochs, batch_size=self.batch_size,
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

    
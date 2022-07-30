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

from dataloader.dataloader import DataLoader

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

import h5py


class CNN_LSTM(BaseModel):
    """CNN_LSTM Model Class"""
    def __init__(self, config):
        super().__init__(config)
        self.num_train = self.config.train.num_training_samples
        self.num_test = self.config.train.num_test_samples
        self.n_samples = self.config.train.n_samples_per_signal
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epoches
        self.det = self.config.train.detector
        self.depth = self.config.train.depth
        self.lr = self.config.learning_rate
        
        self.cnn_filters_1 = self.config.layers.CNN_layer_1
        self.cnn_filters_2 = self.config.layers.CNN_layer_2
        self.lstm_1 = self.config.layers.lstm_1
        self.lstm_2 = self.config.layers.lstm_2
        self.lstm_3 = self.config.layers.lstm_3
        self.kernel_size= self.config.layers.kernel_size
        self.pool_size = self.config.layers.pool_size
        self.train_from_checkpoint = self.config.train.train_from_checkpoint
        self.checkpoint_path = self.config.train.checkpoint_path
        
    def load_data(self):
        """Loads and Preprocess data """
                
        # Load training data
        self.strain_train, self.signal_train = DataLoader(self.det, 'train').load_data(self.config.data)
        self.strain_test, self.signal_test = DataLoader(self.det, 'test').load_data(self.config.data)
        
        # Pre-process data
        self.strain_train = self._preprocess_data(self.strain_train, self.num_train, self.n_samples)
        self.signal_train = self._preprocess_data(self.signal_train, self.num_train, self.n_samples)
        
        self.strain_test = self._preprocess_data(self.strain_test, self.num_test, self.n_samples)
        self.signal_test = self._preprocess_data(self.signal_test, self.num_test, self.n_samples)

        # Reshape data
        self.X_train_noisy, self.X_train_pure = self.reshape_sequences(self.num_train, self.strain_train, self.signal_train)
        self.X_test_noisy, self.X_test_pure = self.reshape_sequences(self.num_test, self.signal_test, self.signal_test)
        
        # Reshape data for Keras
        self.reshape_and_print()
        
        
    def _preprocess_data(self, data, num, samples):
        """ Normalizes training and test set signals """
        
        new_array = np.zeros((num,samples))

        for i in range(num):
            new_array[i][np.where(data[i]>0)] = data[i][data[i]>0]/(np.max(data, axis=1)[i])
            new_array[i][np.where(data[i]<0)] = data[i][data[i]<0]/abs(np.min(data, axis=1)[i])
        
        return new_array
        
        
# Split a univariate sequence into samples
    def split_sequence(self,sequence_noisy,sequence_pure,n_steps):
        X = [] 
        y = []
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
        
#        self.X_train_noisy = self.X_train_noisy.reshape(self.X_train_noisy.shape[0], 516, 4, 1)
#        self.X_test_noisy = self.X_test_noisy.reshape(self.X_test_noisy.shape[0], 516, 4, 1)
#        self.X_train_pure = self.X_train_pure.reshape(self.X_train_pure.shape[0], 516, 1)
#        self.X_test_pure = self.X_test_pure.reshape(self.X_test_pure.shape[0], 516, 1)

        self.X_train_noisy = self.X_train_noisy[:,:,:,None]
        self.X_test_noisy = self.X_test_noisy[:,:,:,None]
        self.X_train_pure = self.X_train_pure[:,:,None]
        self.X_test_pure = self.X_test_pure[:,:,None]
        
        print('x_train_noisy shape:', self.X_train_noisy.shape)
        print('x_test_noisy shape:', self.X_test_noisy.shape)
        print('x_train_pure shape:', self.X_train_pure.shape)
        print('x_test_pure shape:', self.X_test_pure.shape)

        self.X_train_noisy = self.X_train_noisy.astype("float32")
        self.X_test_noisy = self.X_test_noisy.astype("float32")

        self.X_train_pure = self.X_train_pure.astype("float32")
        self.X_test_pure = self.X_test_pure.astype("float32")

        
    from keras import backend as K
    def fractal_tanimoto_loss(self, y_true, y_pred, depth=self.depth, smooth=1e-6):
        x = y_true
        y = y_pred
        
        depth = self.depth+1
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
                
            result =  num * denum * scale

            return  result*scale
    
    
        result = tnmt_base(x,y)
        
        return  1. - result

    def build(self):
        """ Builds and compiles the model """
        with strategy.scope():
            self.model = tf.keras.Sequential()
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_1,self.kernel_size, padding='same'),input_shape=(self.X_train_noisy.shape[1:])))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size, padding='same')))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.cnn_filters_2, self.kernel_size, padding='same')))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_1, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_2, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
            
            self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_3, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
            
            self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
            
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
            self.model.compile(optimizer=optimizer,loss=self.fractal_tanimoto_loss,metrics=['accuracy'])
    
        self.model.summary()
        
        self.train(checkpoint)
           
    def train(self, checkpoint):
        """Trains the model"""
        
        # initialize checkpoints
        dataset_name = "/fred/oz016/Chayan/GW-Denoiser/checkpoints/Saved_checkpoint"
        checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        
        # load best model with min validation loss
        if(self.train_from_checkpoint == True):
            checkpoint.restore(self.checkpoint_path)

        model_history = self.model.fit(self.X_train_noisy, self.X_train_pure, epochs=self.epochs, batch_size=self.batch_size,
                                       validation_data=(self.X_test_noisy,self.X_test_pure))
        
        checkpoint.save(file_prefix=checkpoint_prefix)
        
        self.model.save("model/trained_model.h5")
#        print("Saved model to disk")

        self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])
        
    
    def plot_loss_curves(self, loss, val_loss):
    
        # summarize history for accuracy and loss
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on test data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        plt.savefig("evaluation/Accuracy_curve.png", dpi=200)
                                       
                                       
    def evaluate(self):
        """Predicts resuts for the test dataset"""
#        predictions = []
        predictions = self.model.predict(self.X_test_noisy)
        
        score = self.model.evaluate(self.X_test_noisy, self.X_test_pure, verbose=1)
        
        f1 = h5py.File('evaluation/results_snr_20.hdf', 'w')
        f1.create_dataset('denoised_signals', data=predictions)
        f1.create_dataset('pure_signals', data=self.X_test_pure)
        
        print('\nAccuracy on test data: %0.2f' % score[1])
        print('\nLoss on test data: %0.2f' % score[0])

    
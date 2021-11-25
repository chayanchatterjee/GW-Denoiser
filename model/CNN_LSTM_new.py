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

from __future__ import print_function
from matplotlib import pyplot as plt
plt.switch_backend('agg')
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

#from sklearn import metrics
#from sklearn.metrics import classification_report

#from sklearn import preprocessing
#from sklearn.preprocessing import OneHotEncoder


#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-47a85f0a-67f6-b4ec-c7ce-3cb01dc30fc4, GPU-a19ae4ae-43c6-c5bb-4b7a-d053b52e5c92"
#os.environ['MKL_NUM_THREADS'] = '24'
#os.environ['GOTO_NUM_THREADS'] = '24'
#os.environ['OMP_NUM_THREADS'] = '24'
#os.environ['openmp'] = 'True'

#os.environ['KERAS_BACKEND'] = 'tensorflow'

#import keras
#import keras.backend as K
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D,Input, LSTM, RepeatVector, Multiply, Lambda, Bidirectional, Permute
#from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D, TimeDistributed
#from keras.utils import np_utils
#from keras.layers import Activation
#from keras.regularizers import l2

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
from SampleFileTools1 import SampleFile



#from attention_decoder import AttentionDecoder
#from attention_new import attention

#from Fractal_Tanimoto import ftnmt_loss


from scipy import signal
import random as ran

def glitch(gauss_amp):
    sigma = ran.uniform(0.2,2)
    amp = ran.uniform(-gauss_amp,gauss_amp)
    gauss=(signal.gaussian(512, std=sigma)*amp)
    
    #find the start and end indicies of the gaussian curve
    trigger=0
    for ex,y in enumerate(gauss,start=0):
        if y != 0 and trigger==0:
            trigger=1
            start=ex-1
        if trigger==1 and y==0:
            stop=ex
            break
            
    # find the range of the gaussian curve		
    ragnar = stop-start
    # create the glitch array
    glitch = np.ones(512)-1
    # randomise injection point
    begin=ran.randint(0,512-ragnar)
    # make the glitch
    glitch[begin:begin+ragnar]=gauss[start:stop]
    
    return glitch


obj = SampleFile()
obj.read_hdf("BBH_sample_files/default_training_1_sec_10to80_1.hdf")
df = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#obj2 = SampleFile()
#obj2.read_hdf("../BBH_sample_files/default_training_1_sec_10to80_2.hdf")
#df2 = obj2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#obj3 = SampleFile()
#obj3.read_hdf("../BBH_sample_files/default_snr.hdf")
#df3 = obj3.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

#df = pd.concat([df1,df2], ignore_index= True)


#obj_test_1 = SampleFile()
#obj_test_1.read_hdf("../Real_events/default_GW170104_O2_PSD_1_sec.hdf")
#df_test_1 = obj_test_1.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_2 = SampleFile()
#obj_test_2.read_hdf("../Real_events/default_GW170729_O2_PSD_1_sec.hdf")
#df_test_2 = obj_test_2.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_3 = SampleFile()
#obj_test_3.read_hdf("../Real_events/default_GW170809_O2_PSD_1_sec.hdf")
#df_test_3 = obj_test_3.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_4 = SampleFile()
#obj_test_4.read_hdf("../Real_events/default_GW170814_O2_PSD_1_sec.hdf")
#df_test_4 = obj_test_4.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_5 = SampleFile()
#obj_test_5.read_hdf("../Real_events/default_GW170818_O2_PSD_1_sec.hdf")
#df_test_5 = obj_test_5.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_6 = SampleFile()
#obj_test_6.read_hdf("../Real_events/default_GW170823_O2_PSD_1_sec.hdf")
#df_test_6 = obj_test_6.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_7 = SampleFile()
#obj_test_7.read_hdf("../Real_events/default_GW170608_O2_PSD_1_sec.hdf")
#df_test_7 = obj_test_7.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#obj_test_8 = SampleFile()
#obj_test_8.read_hdf("../Real_events/default_GW150914_O2_PSD_1_sec.hdf")
#df_test_8 = obj_test_8.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

#df_test = pd.concat([df_test_1, df_test_2, df_test_3, df_test_4, df_test_5, df_test_6, df_test_7, df_test_8], ignore_index= True)

obj_test = SampleFile()
obj_test.read_hdf("BBH_sample_files/default_injection_run_Gaussian.hdf")
df_test = obj_test.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.



# Same labels will be reused throughout the program
#LABELS = ['1',
#          '2',
#          '3',
#          '4',
#          '5',
#          '6',
#          '7',
#          '8']
# The number of steps within one time segment
#TIME_PERIODS = 512
TIME_PERIODS = 512
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
#STEP_DISTANCE = 512
STEP_DISTANCE = 512

#Extracting seconds_before_event from 'info.csv' file

#dataset = pd.read_csv('info.csv')
#sbe = dataset.iloc[:,1].values


#Creating Timestamp column
timestamp = []
event_time = 1234567936
count = 0
grid = []
grid_test = []
for i in range(50000):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid.append(np.linspace(event_time - 0.20, event_time + 0.05, 512))
for i in range(8):
#    grid.append(np.linspace(event_time - sbe[i], event_time + (2.0 - sbe[i]), int(2048 * 0.4)))
    grid_test.append(np.linspace(event_time - 0.20, event_time + 0.05, 512))
    
timestamp = np.hstack(grid)#timestamp is now the array representing the required Timestamp column of the datastructure
timestamp_test = np.hstack(grid_test)

def normalize_noisy(a):
    new_array = []
    for i in range(50000):
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
    for i in range(4000):
        dataset = a[i]
        dataset = dataset/10
        new_array.append(dataset)
    return new_array

def normalize_test_pure(a):
    new_array = []
    for i in range(4000):
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
    for i in range(4000):
        dataset = a[i]
#        dataset = dataset[1536:2048]
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
    for i in range(4000):
        dataset = a[i]
#        dataset = dataset[1536:2048]
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
new = df[['l1_strain']].copy() 
new_test = df_test[['l1_strain']].copy()

new_pure = df[['l1_signal']].copy() 
new_test_pure = df_test[['l1_signal']].copy()

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

h1_new = normalize_new(h1_new)
h1_test_new = normalize_test_new(h1_test_new)

h1_pure_new = normalize_new(h1_pure_new)
h1_test_pure_new = normalize_test_new(h1_test_pure_new)

# Only for plotting and comparing. Uncomment this for glitch
#h1_test_glitch_pure_new = normalize_test_glitch_new(h1_test_pure_new)


from numpy import array
# split a univariate sequence into samples
def split_sequence(sequence_noisy,sequence_pure,n_steps):
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

# choose a number of time steps
n_steps = 4
X_train_noisy = []
X_train_pure = []

for i in range(50000):
    X_noisy = h1_new[i]
    X_pure = h1_pure_new[i]
    X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
    X_pure = np.pad(X_pure, (4, 4), 'constant', constant_values=(0, 0))
    # split into samples
    X, y = split_sequence(X_noisy, X_pure, n_steps)
    X_train_noisy.append(X)
    X_train_pure.append(y)
    
X_train_noisy = np.asarray(X_train_noisy)
X_train_pure = np.asarray(X_train_pure)

# choose a number of time steps
n_steps = 4
X_test_noisy = []
X_test_pure = []

for i in range(4000):
    X_noisy = h1_test_new[i]
    X_pure = h1_test_pure_new[i]
    X_noisy = np.pad(X_noisy, (4, 4), 'constant', constant_values=(0, 0))
    X_pure = np.pad(X_pure, (4, 4), 'constant', constant_values=(0, 0))
    # split into samples
    X, y = split_sequence(X_noisy, X_pure, n_steps)
    X_test_noisy.append(X)
    X_test_pure.append(y)
    
X_test_noisy = np.asarray(X_test_noisy)
X_test_pure = np.asarray(X_test_pure)

##### print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train_noisy shape: ', X_train_noisy.shape)
print('x_test_noisy shape: ', X_test_noisy.shape)
# Displays (20869, 40, 3)
#print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect x data
print('x_train_pure shape: ', X_train_pure.shape)
print('x_test_pure shape: ', X_test_pure.shape)

# Displays (20869, 40, 3)
#print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Set input & output dimensions

num_time_periods, num_sensors = 4, 1

num_time_periods_test, num_sensors_test = 4, 1

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods*num_sensors)
input_shape_test = (num_time_periods_test*num_sensors_test)

X_train_noisy = X_train_noisy.reshape(X_train_noisy.shape[0], 516, input_shape)
X_test_noisy = X_test_noisy.reshape(X_test_noisy.shape[0], 516, input_shape)
X_train_pure = X_train_pure.reshape(X_train_pure.shape[0], 516, 1)
X_test_pure = X_test_pure.reshape(X_test_pure.shape[0], 516, 1)


print('x_train_noisy shape:', X_train_noisy.shape)

print('input_shape:', input_shape)


print('x_test_noisy shape:', X_test_noisy.shape)

print('input_shape:', input_shape_test)


print('x_train_pure shape:', X_train_pure.shape)

print('input_shape:', input_shape)


print('x_test_pure shape:', X_test_pure.shape)
print('input_shape:', input_shape_test)



# Convert type for Keras otherwise Keras cannot process the data
X_train_noisy = X_train_noisy.astype("float32")

X_test_noisy = X_test_noisy.astype("float32")

# Convert type for Keras otherwise Keras cannot process the data
X_train_pure = X_train_pure.astype("float32")

X_test_pure = X_test_pure.astype("float32")

#from keras import backend as K
def fractal_tanimoto_loss(y_true, y_pred, depth=0, smooth=1e-6):
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
        prod = tf.math.reduce_sum(prod, axis=1)
        
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
#            result = tf.math.reduce_mean((result + (num/denum)), axis=0)

        result =  num * denum * scale
#        result = result
        return  result
    
    def mean_squared_error_loss(y_true, y_pred):
        
        a = tf.reduce_mean(tf.square(y_pred - y_true))
        
        return a
    
    l1 = tnmt_base(x,y)
#    l1 = weighted_tanimoto(x,y)
    l2 = mean_squared_error_loss(x,y)
#        l2 = self.tnmt_base(1.-preds, 1.-labels)

#        result = 0.5*(l1+l2)
    result = l2 - l1
    
    
    return  result

with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((516,4,1), input_shape=(516,4)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32,1, padding='same'),input_shape=(516,4,1)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(16,1, padding='same')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Activation('tanh')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh',kernel_initializer='glorot_normal', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,loss='mse',metrics=['accuracy'])
    
model.summary()

history = model.fit(X_train_noisy, X_train_pure,
                epochs=800,
                shuffle=True,
                batch_size=500,
                validation_data=(X_test_noisy, X_test_pure))

model.save("Models/model_FractalTanimoto_FT_MSE.h5")
print("Saved model to disk")

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
#plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
#plt.plot(history.history['val_acc'], "g", label="Accuracy of test data")
plt.plot(history.history['loss'], "r--", label="Loss on training data")
plt.plot(history.history['val_loss'], "r", label="Loss on test data")
#plt.title('Model Accuracy and Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.ylim(0)
#plt.legend()
plt.savefig("Accuracy_plots/Accuracy_TimeDistributed_FractalTanimoto_FT_MSE.png", dpi=150)
np.savetxt("Accuracy_plots/Training_loss_FT_MSE.txt", np.transpose(history.history['loss']))
np.savetxt("Accuracy_plots/Test_loss_FT_MSE.txt", np.transpose(history.history['val_loss']))
#plt.show()

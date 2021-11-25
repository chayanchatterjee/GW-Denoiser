# GW-Denoiser: A Deep Learning Model for Gravitational Wave Denoising
This repository is the implementation of the paper [Extraction of Binary Black Hole Gravitational Wave Signals in Detector Data Using Deep Learning](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.064046) by Chayan Chatterjee, Linqing Wen, Foivos Diakogiannis and Kevin Vinsen.  

The aim of this work is to implement a neural network model, called denoising autoencoder, to extract binary black hole gravitational wave signals from stationary Gaussian and real detector noise. A denoising autoencoder is a deep learning architecture that consists of two neural networks, called an encoder and a decoder. Together, these two networks produce clean, noise-free versions of corrupted or noisy input data. The encoder compresses noisy data into a low-dimensional feature vector and the decoder reconstructs the pure input data, with noise removed from the compressed feature vector.
We have used a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) as our encoder and a [Bi-directional Long Short Term Memory Network](https://en.wikipedia.org/wiki/Long_short-term_memory) as our decoder. The architecture of the model is shown below:  
![Model architecture](https://github.com/chayanchatterjee/GW-Denoiser/figures/CNN-LSTM_model_architecture.png?raw=true)


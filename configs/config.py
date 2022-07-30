# -*- coding: utf-8 -*-
# Author: Chayan Chatterjee
# Last modified: 26th November 2021

"""Model config in json format"""

CFG = {
    "data": {
        "path_train": "../BBH_sample_files/default_snr.hdf",
        "path_test": "../BBH_sample_files/default_snr-20_test.hdf",
        
        
    },
    "train": {
        "num_training_samples": 100000,
        "num_test_samples": 25,
        "detector": 'Hanford', # 'Hanford'/'Livingston'/'Virgo'
        "n_samples_per_signal": 512,
        "batch_size": 1000,
        "epoches": 100,
        "depth": 0,
        "train_from_checkpoint": False,
        "checkpoint_path": '/fred/oz016/Chayan/GW-Denoiser/checkpoints/Saved_checkpoint/tmp_0x6513d638/ckpt-1', # if train_from_checkpoint == True
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
        "input": [516,4],
        "layers": {
            "CNN_layer_1": 32,
            "CNN_layer_2": 16,
            "LSTM_layer_1": 50,
            "LSTM_layer_2": 50,
            "LSTM_layer_3": 50,
            "Output_layer": 1,
            "kernel_size": 1,
            "pool_size": 2
        },
        "learning_rate": 1e-3
    }
}

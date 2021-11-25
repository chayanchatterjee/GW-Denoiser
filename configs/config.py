# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path_train": "../../BBH_sample_files/default_training_1_sec_10to80_1.hdf",
        "path_test_1": "../../Real_events/default_GW170104_O2_PSD_1_sec.hdf",
        "path_test_2": "../../Real_events/default_GW170729_O2_PSD_1_sec.hdf",
        "path_test_3": "../../Real_events/default_GW170809_O2_PSD_1_sec.hdf",
        "path_test_4": "../../Real_events/default_GW170814_O2_PSD_1_sec.hdf",
        "path_test_5": "../../Real_events/default_GW170818_O2_PSD_1_sec.hdf",
        "path_test_6": "../../Real_events/default_GW170823_O2_PSD_1_sec.hdf",
        "path_test_7": "../../Real_events/default_GW170608_O2_PSD_1_sec.hdf",
        "path_test_8": "../../Real_events/default_GW150914_O2_PSD_1_sec.hdf",
        "path_test_9": "../../Real_events/default_GW151012_O1_PSD.hdf",
        "path_test_10": "../../Real_events/default_GW151226_O1_PSD.hdf"
        
    },
    "train": {
        "num_training_samples": 50000,
        "num_test_samples": 10,
        "n_samples_per_signal": 2048,
        "batch_size": 1000,
        "epoches": 100,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
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
            "kernel_size": 1
        },
        "learning_rate": 1e-3
    }
}
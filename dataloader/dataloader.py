# -*- coding: utf-8 -*-
"""Data Loader"""

from SampleFileTools1 import SampleFile
import pandas as pd

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        
        obj = SampleFile()
        obj.read_hdf(data_config.path_train)
        train_dataset = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.
        
        obj.read_hdf(data_config.path_test_1)
        df_test_1 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_2)
        df_test_2 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_3)
        df_test_3 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_4)
        df_test_4 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_5)
        df_test_5 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_6)
        df_test_6 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_7)
        df_test_7 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_8)
        df_test_8 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file

        obj.read_hdf(data_config.path_test_9)
        df_test_9 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

        obj.read_hdf(data_config.path_test_10)
        df_test_10 = obj.as_dataframe(True,True,True,False) #creating the dataframe from the hdf file.

        test_dataset = pd.concat([df_test_1, df_test_2, df_test_3, df_test_4, df_test_5, df_test_6, df_test_7, df_test_8, df_test_9, df_test_10], ignore_index= True)

        
        
        return train_dataset, test_dataset
    
    
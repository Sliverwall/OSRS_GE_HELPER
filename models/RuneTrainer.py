'''Module for trainer methods for the RuneLSTM'''
import torch
import torchvision
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class RuneTrainer():
    def __init__(self, 
                 model, 
                 input_features:list,
                 target_features:list,
                 loss_fn=nn.MSELoss(),
                 lr:float=1e-3,
                 time_feature:str= "timestamp",
                 device:str="cuda:0"):
        
        # Model config
        self.model = model
        self.device = device

        # Data config
        self.input_features = input_features
        self.target_features = target_features
        self.time_feature = time_feature

        # Data adjustment
        self.scaler = MinMaxScaler()

        # Training config
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    '''Create dataset methods'''
    def create_sequences(self, input_array,target_array,sequence_length):
        '''
        Method used to create time series inputs and targets
        Order the data in acending order to ensure proper sequence alighment
        '''
        xs, ys = [], []

        for i in range(len(input_array) - sequence_length):
            x = input_array[i:i+sequence_length]
            y = target_array[i+sequence_length]  # predict next point (regression)
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    def create_data_loader(self, 
                        data:pd.DataFrame, 
                        sequence_length:int=10, 
                        batch_size:int=32):
        # Sort data by the time feature to ensure proper sequence alighment
        data = data.sort_values(self.time_feature)
        # Handle missing data points using a linear interpolation
        data = data.interpolate(method='linear', limit_direction='both')

        # Extract input_data and target_data from general dataframe
        input_data = data[self.input_features]
        target_data = data[self.target_features]

        # Use trainer's scaler method to input scale features
        input_data[self.input_features] = self.scaler.fit_transform(data[self.input_features])

        # Convert inputs and targets to numpy arrays
        input_array = input_data.values()
        target_array = target_data.values()

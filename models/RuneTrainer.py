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
                 num_epochs:int=100,
                 batch_size:int=32,
                 seq_length:int=10,
                 loss_fn=nn.MSELoss(),
                 lr:float=1e-3,
                 time_feature:str= "timestamp",
                 device:str="cuda:0"):
        
        # Model config
        self.device = device
        self.model = model.to(self.device)

        # Data config
        self.input_features = input_features
        self.target_features = target_features
        self.time_feature = time_feature
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seq_length = seq_length

        # Data adjustment
        self.scaler = MinMaxScaler()

        # Training config
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    '''Create dataset methods'''
    def create_sequences(self, 
                         input_array,
                         target_array,
                         sequence_length:int) -> tuple[torch.Tensor,torch.Tensor]:
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
                        sequence_length:int=10):
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

        # Check if input_array length and target_array length align
        if len(input_array) != len(target_array):
            print("ERROR at create_data_loader.... input feature length does not match target feature length")
            return
        
        # Create sequenced data arrays
        X, y = self.create_sequences(input_array=input_array, target_array=target_array,sequence_length=sequence_length)

        # Wrap sequenced data into a tensor dataloader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch_size)
        
        return loader
    
    '''Set up training function'''
    def fit(self, 
            data_loader:DataLoader):
        # Set to train mode
        self.model.train()
        # train using batches
        for epoch in range(self.num_epochs):
            for batch_inputs, batch_targets in data_loader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

'''Module for trainer methods for the RuneLSTM'''
import torch
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
        self.time_feature = time_feature
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seq_length = seq_length

        # Data adjustment
        self.input_scaler = MinMaxScaler()

        # Training config
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    '''Create dataset methods'''
    def create_sequences(self, 
                         input_array:np.ndarray) -> tuple[torch.Tensor,torch.Tensor]:
        '''
        Method used to create time series inputs and targets
        Order the data in acending order to ensure proper sequence alighment
        '''
        xs, ys = [], []

        for i in range(len(input_array) - self.seq_length):
            x = input_array[i:i+self.seq_length]
            y = input_array[i+self.seq_length]  # predict next point (regression)
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

    def create_data_loader(self, 
                        data:pd.DataFrame):
        # Sort data by the time feature to ensure proper sequence alighment
        data = data.sort_values(self.time_feature)
        # Handle missing data points using a linear interpolation
        data = data.interpolate(method='linear', limit_direction='both')

        # 2) extract arrays
        inputs = data[self.input_features].values  # shape (T, n_feats)
        scaled = self.input_scaler.fit_transform(inputs)

        # Create sequenced data arrays. Input needs to be the same size as the output
        X, y = self.create_sequences(input_array=scaled)

        # Get unscaled targets for plot labels
        y_unscaled = inputs[self.seq_length :]

        # Wrap sequenced data into a tensor dataloader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
        
        # Return loader for fitting, X for autoregressive predictions, and unscaled targets for plotting
        return loader, X, y_unscaled
    
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

    def autoregressive_pred(self, input_seq: torch.Tensor, n_steps: int=10) -> np.ndarray:
        """
        input_seq: Tensor of shape (1, seq_length, n_feats)
        returns:  (n_steps, n_feats) unscaled predictions
        """
        self.model.eval()
        # ensure batch dim
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(0)

        preds = []
        
        with torch.no_grad():
            for _ in range(n_steps):
                # predict next time-step (still scaled)
                next_pred = self.model(input_seq.to(self.device))  
                # next_pred: (1, n_feats)
                next_pred_step = next_pred.unsqueeze(1).to(self.device)           # (1,1,n_feats)

                # slide window: drop oldest, append newest
                input_seq = torch.cat(
                    [input_seq[:, 1:, :].to(self.device), next_pred_step],
                    dim=1
                )

                preds.append(next_pred_step)

        # combine and unscale
        preds_numpy = torch.cat(preds, dim=1).squeeze(0).cpu().numpy()
        # preds_tensor.shape == (n_steps, n_feats)
        preds_unscaled = self.input_scaler.inverse_transform(preds_numpy)
        return preds_unscaled
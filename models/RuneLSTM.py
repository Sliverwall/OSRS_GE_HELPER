
import torch
from torch import nn
from torch.nn import functional as F

'''Construct a model with a RNN layer + a MLP to process the regression'''
class RuneLSTM(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, output_size, device:str="cuda:0"):
        super(RuneLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # use torch.nn LSTM layer to handle LSTM logic. Expects: (t, batch_size, features)
        self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers, batch_first=False)
        

        # Network to handle regression task
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Dropout layer after LSTM to regularize between the layers
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (batch_size, t, features)
        # Make input match (t, batch_size, features)
        x = x.transpose(0,1)
        
        # Initialize hidden state with shape (num_layers, batch_size, features)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)  # Initialize cell state
        
        # Pass to lstm
        # output shape: (t, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))  # output shape: (t, batch_size, hidden_size)
        
        # Apply dropout after LSTM output
        output = self.dropout(output)
        
        # Use the last time step's output for regression
        output = self.net(output[-1])
        return output


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
                 time_feature:str= "timestamp"):
        
        # Model config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                        data:pd.DataFrame,
                        val_split:float = 0.2):
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

        # Optional validation split
        val_size = int(len(X) * val_split)
        train_size = len(X) - val_size

        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        val_dataset   = TensorDataset(X[train_size:], y[train_size:])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, X, y_unscaled
    
    '''Set up training function'''
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            load_checkpoint=True,
            save_checkpoint=True):
        
        # Load checkpoint if checked
        if load_checkpoint:
            self.load_checkpoint()
        # Set to train mode
        self.model.train()
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Evaluate on validation set
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_inputs, val_targets in val_loader:
                        val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
                        val_outputs = self.model(val_inputs)
                        val_loss += self.loss_fn(val_outputs, val_targets).item()
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                self.model.train()

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}] "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}] Train Loss: {avg_train_loss:.4f}")

        if save_checkpoint:      
            # Save checkpoint once done
            checkpoint = {
                'epoch': epoch,                          # e.g. current epoch number
                'model_state_dict': self.model.state_dict(),  # the learnable params
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            # Set checkpoint_path
            checkpoint_path = 'models/checkpoints/current_model.pth'
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
        return train_losses, val_losses if val_loader else None

    def load_checkpoint(self, checkpoint_path:str = "models/checkpoints/current_model.pth"):
        '''
        Method to load a checkpoint in
        '''
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        # Load parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def autoregressive_pred(self, input_seq: torch.Tensor, n_steps: int=10) -> np.ndarray:
        """
        input_seq: Tensor of shape (1, seq_length, n_feats)
        returns:  (n_steps, n_feats) unscaled predictions
        """
        print("Loading checkpoint...")
        self.load_checkpoint()

        # Set to eval mode
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
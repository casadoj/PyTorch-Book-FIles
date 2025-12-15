import time
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

class TimeSeriesMLP(nn.Module):
    def __init__(self, input_size):
        super(TimeSeriesMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 10), 
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.network(x)
    
class TimeSeriesCNN1D(nn.Module):
    def __init__(self, input_size):
        super(TimeSeriesCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=3,
            padding=1
            )

        conv_output_size = input_size  # Same padding maintains input size

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * conv_output_size, 28)
        self.dense2 = nn.Linear(28, 10)
        self.dense3 = nn.Linear(10, 1)

    def forward(self, x):
        # Transpose input from [batch_size, sequence_length] to [batch_size, 1, sequence_length]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    

class TimeSeriesMultilayerCNN1D(nn.Module):
    def __init__(self, input_size, num_conv_layers, conv_channels, kernel_size, dense_sizes):
        super(TimeSeriesMultilayerCNN1D, self).__init__()

        # Conv layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_channels[i],
                kernel_size=kernel_size,
                padding='same'  # This maintains the sequence length
            ))
            self.conv_layers.append(nn.ReLU())
            in_channels = conv_channels[i]

        # Calculate flattened size after convolutions
        # Since we use 'same' padding, the sequence length remains input_size
        # The channels become the last conv layer's out_channels
        flattened_size = input_size * conv_channels[-1]

        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_size = flattened_size

        for size in dense_sizes:
            self.dense_layers.append(nn.Linear(prev_size, size))
            self.dense_layers.append(nn.ReLU())
            prev_size = size

        # Final output layer
        self.output_layer = nn.Linear(prev_size, 1)

        # Flatten layer
        self.flatten = nn.Flatten()

        print(f"Model architecture:")
        print(f"Input size: {input_size}")
        print(f"Conv channels: {conv_channels}")
        print(f"Flattened size: {flattened_size}")
        print(f"Dense sizes: {dense_sizes}")

    def forward(self, x):
        # Print shape at each step for debugging
        # print(f"Input shape: {x.shape}")

        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        # print(f"After reshape: {x.shape}")

        # Conv layers
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            # print(f"After conv layer {i//2}: {x.shape}")

        # Flatten
        x = self.flatten(x)
        # print(f"After flatten: {x.shape}")

        # Dense layers
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            # print(f"After dense layer {i//2}: {x.shape}")

        # Output layer
        x = self.output_layer(x)
        # print(f"Final output: {x.shape}")
        
        return x


class TimeSeriesRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1, dropout_rate=0.3):
        super(TimeSeriesRNN, self).__init__()

        self.rnn1 = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  # Add dropout to RNN

        self.rnn2 = nn.RNN(input_size=hidden_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          dropout=dropout_rate)  # Add dropout to RNN

        self.dropout = nn.Dropout(dropout_rate)  # Additional dropout layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out1, _ = self.rnn1(x)
        out2, _ = self.rnn2(out1)
        last_out = out2[:, -1, :]
        last_out = self.dropout(last_out)  # Add dropout before final layer
        output = self.linear(last_out)
        return output
    

class TimeSeriesGatedRNN(nn.Module):
    def __init__(
            self,
            type: Literal['LSTM', 'GRU'],
            input_size=1, 
            hidden_size=100, 
            output_size=1, 
            dropout_rate=0.1, 
            bidirectional=False
            ):
        super(TimeSeriesGatedRNN, self).__init__()

        # size of the output of the GRNN
        grnn_output_size = hidden_size * 2 if bidirectional else hidden_size

        if type.upper() == 'LSTM':
            self.grnn1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional
                )
            self.grnn2 = nn.LSTM(
                input_size=grnn_output_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional
                )
        elif type.upper() == 'GRU':
            self.grnn1 = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional
                )
            self.grnn2 = nn.GRU(
                input_size=grnn_output_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional
                )
        else:
            raise ValueError(f'"type" must be either "LSTM" or "GRU". {type} was introduced.')

        # Additional layers
        self.fc1 = nn.Linear(grnn_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.grnn1(x)
        out, _ = self.grnn2(out)
        out = out[:, -1, :]

        # Additional processing
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        output = self.linear(out)
        
        return output
    

def train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs=100,
        scheduler=None,
        device=None, 
        patience: int = None,
        verbose: Optional[int] = 100
        ):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # training loop
    train_losses, val_losses, learning_rates = [], [], [] # training outputs
    best_loss, best_model, counter = float('inf'), None, 0 # early stopping
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # If using OneCycleLR, step the scheduler here
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step the scheduler
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # For ReduceLROnPlateau
        elif isinstance(scheduler, lr_scheduler.StepLR):
            scheduler.step()  # For other schedulers

        # Record the learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if isinstance(verbose, int):
            if (epoch + 1) % verbose == 0:
                print(f'Epoch [{epoch + 1} / {epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}')
            
        # Early stopping
        if val_loss < best_loss:
            best_loss, best_model, counter = val_loss, deepcopy(model), 0
        else:
            counter += 1
        if patience:
            if counter >= patience:
                print(f'Early stopping tiggered after {epoch} epochs')
                break

    # save losses and learning rates as methods
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rates': learning_rates
    }
    
    return best_model, best_loss, history


def predict(model, loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Make predictions
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            predictions.append(outputs.cpu().numpy())
            targets.append(y.numpy())

    predictions = torch.tensor(np.concatenate(predictions))
    targets = torch.tensor(np.concatenate(targets))
    
    return predictions, targets


def neural_architecture_search(train_loader, val_loader, input_size=29):
    """Perform neural architecture search for CNN architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the search space
    num_conv_layers_options = [1, 2]  # Reduced for initial testing
    conv_channels_options = [
        [32],
        [64],
        [32, 16],
        [64, 32],
    ]
    kernel_sizes = [3, 5]
    dense_sizes_options = [
        [16],
        [32, 16],
        [64, 32],
    ]
    learning_rates = [0.001, 0.0001]

    best_mae = float('inf')
    best_config = None
    best_model = None
    results = []

    # Generate valid configurations
    configurations = []
    for num_conv_layers in num_conv_layers_options:
        for channels in conv_channels_options:
            if len(channels) == num_conv_layers:  # Only use channel configs that match layer count
                for kernel_size in kernel_sizes:
                    for dense_sizes in dense_sizes_options:
                        for lr in learning_rates:
                            configurations.append({
                                'num_conv_layers': num_conv_layers,
                                'conv_channels': channels,
                                'kernel_size': kernel_size,
                                'dense_sizes': dense_sizes,
                                'learning_rate': lr
                            })

    print(f"Total configurations to try: {len(configurations)}")
    start_time = time.time()

    for idx, config in enumerate(configurations):
        print(f"\nTrying configuration {idx + 1}/{len(configurations)}:")
        print(config)

        try:
            model = TimeSeriesMultilayerCNN1D(
                input_size=input_size,
                num_conv_layers=config['num_conv_layers'],
                conv_channels=config['conv_channels'],
                kernel_size=config['kernel_size'],
                dense_sizes=config['dense_sizes']
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

            trained_model, val_loss, history = train(
                model, train_loader, val_loader, criterion, optimizer,
                epochs=100, device=device, patience=10
            )

            # Evaluate model
            val_predictions, val_targets = predict(trained_model, val_loader)
            val_mae = torch.mean(torch.abs(val_predictions - val_targets))

            config_results = {
                **config,
                'val_mae': val_mae,
                'val_loss': val_loss
            }
            results.append(config_results)

            if val_mae < best_mae:
                best_mae = val_mae
                best_config = config_results
                best_model = deepcopy(trained_model)
                print(f"New best MAE: {val_mae:.4f}")

        except Exception as e:
            print(f"Error with configuration {idx + 1}: {str(e)}")
            continue

    total_time = time.time() - start_time
    results.sort(key=lambda x: x['val_mae'])

    return best_model, results, total_time
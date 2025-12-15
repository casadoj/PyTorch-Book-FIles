import numpy as np
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
    

def train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs=100,
        scheduler=None,
        device=None, 
        verbose=True
        ):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
        
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
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
        val_loss = 0
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

        if verbose & ((epoch + 1) % 100 == 0):
            print(f'Epoch [{epoch + 1} / {epochs}], '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}')
            
    return train_losses, val_losses, learning_rates


def predict(model, dataloader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        # Get predictions for validation set
        predictions = []
        targets = []
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(y.numpy())

    predictions = torch.tensor(np.array(predictions))
    targets = torch.tensor(np.array(targets))
    
    return predictions, targets
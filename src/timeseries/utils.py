import numpy as np
import matplotlib.pyplot as plt
import torch

def create_sliding_windows(data, window_size, shift=1):
    """
    Creates sliding windows from a sequence, splitting each window into features and target.

    Args:
        data: Input sequence (can be list, numpy array, or torch tensor)
        window_size: Size of each window
        shift: Number of elements to shift each window (stride)

    Returns:
        tuple: (features, targets) where:
            - features is a tensor of shape (num_windows, window_size-1)
            - targets is a tensor of shape (num_windows, 1)
    """
    # Convert input to tensor if it isn't already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    # Create windows using unfold
    windows = data.unfold(0, window_size, shift)

    # Split each window into features (all but last element) and target (last element)
    features = windows[:, :-1]  # All elements except the last
    targets = windows[:, -1:]   # Just the last element

    return features, targets


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.4,
        np.cos(season_time * 2 * np.pi),
        1 / np.exp(3 * season_time)
        )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def normalize_series(data, missing_value=999.9):
    # Convert to numpy array if not already
    data = np.array(data, dtype=np.float64)

    # Create mask for valid values (not NaN and not missing_value)
    valid_mask = (data != missing_value) & (~np.isnan(data))

    # Keep only valid values
    clean_data = data[valid_mask]

    # Normalize using only valid values
    mean = np.mean(clean_data)
    std = np.std(clean_data)
    normalized = (clean_data - mean) / std

    return normalized


def plot_training(train_losses, val_losses, learning_rates, **kwargs):

    # Plot training progress and learning rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs.get('figsize', (12, 4)))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot learning rate
    ax2.plot(learning_rates, label='Learning Rate', color='r')
    ax2.set_title('Learning Rate Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.legend()
    ax2.grid(True)

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])

    plt.tight_layout()
    plt.show()


def plot_prediction(predictions, targets, **kwargs):
    # Plot predictions vs actual for validation set
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 4)))

    ax.plot(targets, label='Actual', color="lightgrey")
    ax.plot(predictions, label='Predicted', color="red")

    # performance
    mae = torch.mean(torch.abs(predictions - targets))
    ax.text(.025, .9, f'MAE = {mae:.3f}', transform=ax.transAxes)

    ax.set(
        title='Predictions vs Actual Values (Validation Set)',
        xlabel='Time',
        ylabel='Value'
        )
    ax.legend(frameon=False)
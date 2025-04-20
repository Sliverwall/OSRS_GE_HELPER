'''Module to create needed matplotlib plots'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
def plot_price_prediction(y, preds):
    """
    Returns a matplotlib figure comparing actual and predicted high/low prices.

    Parameters:
    - y: ndarray of actual values (shape: [timesteps, 2])
    - preds: ndarray of predicted values (shape: [1, timesteps, 2])

    Returns:
    - fig: matplotlib figure object
    """

    # Combine actual + predicted values
    combined_high = np.concatenate([y[:, 0], preds[:, 0]])
    combined_low = np.concatenate([y[:, 1], preds[:, 1]])

    time_range = np.arange(len(combined_high))
    split_index = len(y)

    # Compute smoothed actual values using running average with window=10
    smooth_high = pd.Series(combined_high[:split_index]).rolling(window=10, min_periods=1).mean()
    smooth_low = pd.Series(combined_low[:split_index]).rolling(window=10, min_periods=1).mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot actual and predicted high prices
    ax.plot(time_range[:split_index], y[:, 0], label='Actual avgHighPrice', alpha=0.5)
    ax.plot(time_range[split_index:], preds[:, 0], label='Predicted avgHighPrice', linestyle='--')

    # Plot actual and predicted low prices
    ax.plot(time_range[:split_index], y[:, 1], label='Actual avgLowPrice', alpha=0.5)
    ax.plot(time_range[split_index:], preds[:, 1], label='Predicted avgLowPrice', linestyle='--')

    # Plot smoothed actual high/low prices
    ax.plot(time_range[:split_index], smooth_high, label='Smoothed avgHighPrice', linewidth=2)
    ax.plot(time_range[:split_index], smooth_low, label='Smoothed avgLowPrice', linewidth=2)

    ax.set_title('Price Prediction (avgHighPrice & avgLowPrice)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Price')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig
'''Module to create needed matplotlib plots'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot actual and predicted high prices
    ax.plot(time_range[:split_index], combined_high[:split_index], label='Actual avgHighPrice')
    ax.plot(time_range[split_index:], combined_high[split_index:], label='Predicted avgHighPrice', linestyle='--')

    # Plot actual and predicted low prices
    ax.plot(time_range[:split_index], combined_low[:split_index], label='Actual avgLowPrice')
    ax.plot(time_range[split_index:], combined_low[split_index:], label='Predicted avgLowPrice', linestyle='--')

    ax.set_title('Price Prediction (avgHighPrice & avgLowPrice)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Price')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig
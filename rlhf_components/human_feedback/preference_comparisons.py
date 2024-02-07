import pandas as pd

class HumanFeedback:
    def __init__(self, df):
        """
        Initialize the HumanFeedback class.

        Parameters:
        - df: pd.DataFrame
            DataFrame containing historical data.
        """
        # Store the input DataFrame containing historical data
        self.data = df

    def trading_strategy(self):
        """
        Define a trading strategy based on historical data.

        Returns:
        - pd.DataFrame
            DataFrame containing trading signals based on the strategy.
        """
        # Calculate the 7-day moving average of the closing prices
        self.data['7_day_mean'] = self.data['Close'].shift(1).rolling(window=7).mean()

        # Drop rows with NaN values introduced by the rolling mean
        self.data = self.data.dropna()

        # Create a DataFrame to store trading signals
        signals = pd.DataFrame(index=self.data.index)

        # Generate trading signals based on the strategy (Buy when the price is above the 7-day moving average)
        signals['signal'] = self.data['Close'] > self.data['7_day_mean']

        # Drop any remaining NaN values from the signals DataFrame
        signals = signals.dropna()

        print(f'Length of signals from strategy: {len(signals)}')
        
        # Return the DataFrame containing trading signals
        return pd.DataFrame(signals)

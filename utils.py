import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Function to load financial data using Yahoo Finance API (yfinance)
def load_financial_data(symbol, start_date, end_date):
    """Load financial data using Yahoo Finance API (yfinance)."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    return pd.DataFrame(data)

# Function to display information about the loaded data
def display_data_info(data):
    """Display information about the loaded data."""
    print(f'Data shape: {data.shape}')
    print(f'The columns are: {data.columns} ')
    print(f'Printing dataframe:\n{data.head()}')

# Function to test the trading agent using Deep Q-Network (DQN)
def test_agent(agent, environment, max_steps, human_feedback):
    """Test the trading agent using Deep Q-Network (DQN)."""
    print('Testing start')
    agent.test(max_steps, human_feedback)
    print(f'Final portfolio value: {environment.portfolio_value}')
    print('Testing completed ')

# Function for reward shaping based on human feedback
def reward_shaping_human_feedback(action, human_answer, i):
    """
    Shape reward based on human feedback.

    Args:
        action (str): The action taken by the agent.
        human_answer (list): List of human feedback for each time step.
        i (int): Index representing the current time step.

    Returns:
        int: The shaped reward based on human feedback.
    """
    # Check if the agent's action matches human feedback
    if human_answer[i] == action:
        reward = 1  # Positive reward for a correct action
    else:
        reward = -1  # Negative reward for an incorrect action

    return reward


# Function for reward shaping based on profit/loss
def reward_shaping(profit_loss):
    """
    Shape reward based on profit/loss.

    Args:
        profit_loss (float): The profit or loss from a trading action.

    Returns:
        float: The shaped reward based on the profit or loss.
    """
    # Check if there's a loss
    #if profit_loss < 0:
     #   penalty_1 = 0.00  # Adjust the penalty value as needed
    return profit_loss  



# Function to calculate precision, recall, F1 score, and support for classification
def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F1 score, and support for each class and weighted averages.

    Parameters:
    - y_true (list or array): True labels
    - y_pred (list or array): Predicted labels 

    Returns:
    - precision (array): Precision scores for each class
    - recall (array): Recall scores for each class
    - f1_score (array): F1 scores for each class
    - support (array): The number of occurrences of each label in y_true.
    - weighted_avg_precision (float): Weighted average precision
    - weighted_avg_recall (float): Weighted average recall
    - weighted_avg_f1 (float): Weighted average F1 score
    """
    # Calculate precision, recall, F1 score, and support for each class
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels=['buy', 'short', 'hold'], average=None)

    # Display metrics for each class
    for label, prec, rec, f1, sup in zip(['buy', 'short', 'hold'], precision, recall, fscore, support):
        print(f"Class: {label}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall: {rec:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Support: {sup}")
        print("\n")

    # Calculate weighted averages
    weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print("Weighted Average Precision:", weighted_avg_precision)
    print("Weighted Average Recall:", weighted_avg_recall)
    print("Weighted Average F1 Score:", weighted_avg_f1)

    return precision, recall, fscore, support, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1

# Function to plot training results, including rewards, actions, portfolio values, and losses
def plot_results(list_of_reward, list_of_actions, list_of_portfolio, losses):
    """
    Plots the training results, including episode rewards and losses.

    Args:
        list_of_reward (list): List of rewards for each episode.
        list_of_actions (list): List of actions taken in each episode.
        list_of_portfolio (list): List of portfolio values over time.
        losses (list): List of losses over training steps.
    """
    # Create a figure with a specified size
    plt.figure(figsize=(12, 6))

    # Plot histogram of all rewards
    plt.subplot(2, 2, 1)
    colors = {'Positive_Reward': 'green', 'Negative_Reward': 'red', 'Zero_Reward': 'blue'}
    positive_values = [value for value in list_of_reward if value > 0]
    zero_values = [value for value in list_of_reward if value == 0]
    negative_values = [value for value in list_of_reward if value < 0]
    counts = [len(positive_values), len(zero_values), len(negative_values)]
    plt.bar(['Positive_Reward', 'Zero_Reward', 'Negative_Reward'], height=counts, color=[colors[key] for key in ['Positive_Reward', 'Zero_Reward', 'Negative_Reward']], label=['Positive_Reward', 'Zero_Reward', 'Negative_Reward'])
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Positive, Zero, and Negative Values of Reward')
    plt.legend()

    # Plot bar graph of actions
    plt.subplot(2, 2, 2)
    buy_values = [1 if action == 'buy' else 0 for action in list_of_actions]
    short_values = [1 if action == 'short' else 0 for action in list_of_actions]
    hold_values = [1 if action == 'hold' else 0 for action in list_of_actions]
    colors = {'buy': 'green', 'short': 'red', 'hold': 'blue'}
    categories = ['Buy', 'Short', 'Hold']
    plt.bar(categories, [sum(buy_values), sum(short_values), sum(hold_values)], color=[colors[cat.lower()] for cat in categories], label=['Buy', 'Short', 'Hold'])
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title('Number of Buy, Short, and Hold Actions')
    plt.show()

    # Plot portfolio values over time
    plt.subplot(2, 2, 3)
    plt.plot(list_of_portfolio, label='Portfolio Value', color='blue', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()

    # Plot losses
    plt.subplot(2, 2, 4)
    plt.plot(losses)
    plt.title(f'Loss Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # Adjust layout for better presentation
    plt.tight_layout()

    # Show the plot
    plt.show()

# End of the code

import pandas as pd
import numpy as np
from rlhf_components.environment.state_constructer import StatesConstruction
from utils import reward_shaping,reward_shaping_human_feedback

class RLEnvironment:
    def __init__(self, df, initial_amount, State_data):
        """
        Initialize the RLHFEnvironment.

        Parameters:
        - df: A DataFrame containing historical data.
        - initial_amount: Initial capital for trading.
        - State_data: A DataFrame containing precomputed states for the environment(every day).
            (size reduce by x days as computing x day mean, here we have 7 day means length reduce (n-7) if there is n days of data )
        """
        self.df = df
        self.done = False
        self.portfolio_value = [initial_amount]
        self.trades = [] 
        self.states_data = State_data

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - states: The initial state of the environment.
        """
        # Combine the initial state from precomputed states and the current portfolio value
        self.states = self.states_data.iloc[0, :].to_list() + [self.portfolio_value[-1]]
        return np.array(self.states)

    def step(self, action, i):
        """
        Take an action in the environment and return the next state, reward, and done flag.

        Parameters:
        - actions: The action.
        - i: The current time step.

        Returns:
        - new_states: The new state after taking the action.
        - reward: The reward for the action.
        - done: True if the episode is done, else False.
        """
        if self.states is None:
            raise ValueError("Call reset to initialize the environment.")
        #new_states, reward, done = self.transition(action, i)
        
        if action == 'buy':
            profit_loss =  self.normal_trade(i)
            #reward = self.normal_trade(i)
        elif action == 'short':
            profit_loss = self.short_trade(i)
            #reward = self.short_trade(i)
        elif action == 'hold':
            profit_loss = 0
            self.trades.append(('hold', self.df.index[i], self.df['Open'][i]))
        else:
            raise ValueError("Invalid action")
        
                
        # Update next state
        new_states = self.states_data.iloc[i+1, :].to_list()
        next_states = new_states + [self.portfolio_value[-1]]
        reward = reward_shaping(profit_loss)
        self.portfolio_value.append(self.portfolio_value[-1]+profit_loss)
        self.done = True if i == len(self.states_data) - 1 else False
       
        return np.array(next_states), reward,  self.done



    def normal_trade(self, i):
        """
        Perform a normal trading strategy: Buy low (today's open) and sell high (next day's open).

        Args:
        i (int): Index representing the current day.

        Returns:
        float: Profit or loss from the trade.
        """
    
        # Buy Low (today's open)
        self.num_of_shares = self.portfolio_value[-1] / self.df['Open'][i]
        self.trades.append(('BUY', self.df.index[i], self.df['Open'][i]))
        buy_cost = self.df['Open'][i] * self.num_of_shares

        # Sell High (next day's open)
        sell_price = self.df['Open'][i+1] * self.num_of_shares
        profit_loss = sell_price - buy_cost

        return profit_loss


    def short_trade(self, i):
        """
        Perform a short trading strategy: Sell high (today's open) and buy low to cover (next day's open).

        Args:
            i (int): Index representing the current day.

        Returns:
            float: Profit or loss from the short trade.
        """
    
        # Sell High (today's open)
        self.num_of_short_stocks = self.portfolio_value[-1] / self.df['Open'][i]
        cash_from_sale = self.df['Open'][i] * self.num_of_short_stocks
        self.trades.append(('SELL', self.df.index[i], self.df['Open'][i]))

        # Buy Low to Cover (next day's open)
        buy_price = self.df['Open'][i+1] * self.num_of_short_stocks

        # Calculate Profit or Loss
        profit_loss = cash_from_sale - buy_price

        return profit_loss

    
    
    
    
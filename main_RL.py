import numpy as np
import pandas as pd
import yfinance as yf
from rlhf_components.environment.RL_Environment import RLEnvironment
from rlhf_components.agents.DQN_Agent import DQNAgent
from rlhf_components.environment.state_constructer import StatesConstruction
from utils import display_data_info,load_financial_data
from datetime import datetime

def main():
    # Load financial data
    symbol = 'qqq'
    start_date_train = '2018-07-01'
    end_date_train = '2020-08-01'
    start_date_test = '2020-09-01'
    end_date_test = '2021-09-01'
    
    
    # Define predefined values
    initial_amount = 1000
    num_episodes_train = 5
    
    financial_train_data = load_financial_data(symbol, start_date_train, end_date_train)
    financial_test_data = load_financial_data(symbol, start_date_test, end_date_test)

    # Display information about the loaded data
    print('---------------------------------------------')
    print('Training Data Information:')
    display_data_info(financial_train_data)
    print('---------------------------------------------')
    print('Testing Data Information:')
    display_data_info(financial_test_data)
    print('---------------------------------------------')
    
    
    

    # Define state constructor to extract features for the environment
    StatesConstruction_ = StatesConstruction(financial_train_data,financial_test_data)
    state_constructor_train,state_constructor_test = StatesConstruction(financial_train_data,financial_test_data).get_state()
 
    print(f'State training data shape: {state_constructor_train.shape}')
    print(f'State training data shape: {state_constructor_test.shape}')
    
    
    
    # Instantiate the trading environment
    env_train = RLEnvironment(financial_train_data, initial_amount, state_constructor_train)
    
    env_test  = RLEnvironment(financial_test_data, initial_amount, state_constructor_test)
    print('Trading environments executed ')
    
   

    max_steps_train = state_constructor_train.shape[0]
    print(f'Max Training steps: {max_steps_train}')
    # Train the trading agent
    agent = DQNAgent(env_train, env_test)
    # train agent 
    agent.train(max_steps_train,num_episodes_train)
    # save agent 
    #agent.save_agent(filename= f'{datetime.date()} DQN.pkl')
    # load agent 
    #agent.load_agent(filename= f'{datetime.date()} DQN.pkl')
    #Testing of agent 
    print('Testing Start')
    #Define Max_steps for testing (always number of rows of test states)
    max_steps_test = state_constructor_test.shape[0]
    print(f'Max testing steps: {max_steps_test}')
    
    print('Testing start')
    agent.test(max_steps_test)
    print('Testing completed ')
    
    print(f'Final portfolio value: {env_test.portfolio_value[-1]}')
  
    
    #Monitor the agent's performance for training by plotting training results
    #print('Training Result and agent Evaluation ')
    agent.plot_train_result()
    

    #Monitor the agent's performance for testing by plotting testing results
    print('Testing Result')
    agent.plot_test_result()
    
    
    #evaluate the agent performance for train an test
    print(len(StatesConstruction_.y_true_train))
    print(len(StatesConstruction_.y_true_test))
   
    
    print('All action during Train')
    print(agent.action_train)
    
    print('All action during test')
    print(agent.action_test)
    
    print('Agent Evluation')
    agent.evaluation(StatesConstruction_.y_true_train,StatesConstruction_.y_true_test,agent.action_train,agent.action_test)
    
   
if __name__ == "__main__":
    main()

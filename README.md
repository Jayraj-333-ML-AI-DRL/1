# RLHF Trading Library

This Python library implements a reinforcement learning-based trading system using a Deep Q-Network (DQN) agent. The system is designed to trade financial instruments based on historical price data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)


## Installation
To run this project, you'll need to have the following dependencies installed:

- Python 3.x
- pandas
- TensorFlow
- yfinance
- numpy
- matplotlib
- keras
- scikit-learn


- To install the all dependencies , you can use 
 ```python
 pip install -r read requirements.txt
```
# Usage

To train and evaluate the trading agent, follow these steps:


Before using the library, make sure to import the required modules:

  -
    ```python
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from rlhf_components.environment.RL_Environment import RLEnvironment
    from rlhf_components.agents.DQN_Agent import DQNAgent
    from rlhf_components.environment.state_constructer import StatesConstruction
    from utils import display_data_info, train_agent, load_financial_data, test_agent
    ```
1. **Load Financial Data:**
   - Set the symbol, start date, and end date for the financial data.
   - Example:
     ```python
     symbol = 'qqq'
     start_date_train = '2018-07-01'
     end_date_train = '2020-08-01'
     start_date_test = '2020-09-01'
     end_date_test = '2021-09-01'
     financial_train_data = load_financial_data(symbol, start_date_train, end_date_train)
     financial_test_data = load_financial_data(symbol, start_date_test, end_date_test)
     ```

2. **Display Information About Loaded Data:**
   - Print information about the loaded training and testing data.
   - Example:
     ```python
     display_data_info(financial_train_data)
     display_data_info(financial_test_data)
     ```

3. **Define Predefined Values:**
   - Set initial trading amount and the number of training episodes.
   - Example:
     ```python
     initial_amount = 1000
     num_episodes_train = 3
     ```

4. **Define State Constructor:**
   - Create a state constructor to extract features for the environment.
   - Example:
     ```python
     StatesConstruction_ = StatesConstruction(financial_train_data, financial_test_data)
     state_constructor_train, state_constructor_test = StatesConstruction_.get_state()
     ```

5. **Instantiate Trading Environment:**
   - Create instances of the trading environment for training and testing.
   - Example:
     ```python
     env_train = RLEnvironment(financial_train_data, initial_amount, state_constructor_train)
     env_test = RLEnvironment(financial_test_data, initial_amount, state_constructor_test)
     ```

6. **Train the Trading Agent:**
   - Configure the agent with the training environment and train it.
   - Example:
     ```python
      max_steps_train = state_constructor_train.shape[0]
      
      # Train the trading agent
      agent = DQNAgent(env_train, env_test)
      # train agent 
      agent.train(max_steps_train,num_episodes_train)
     ```

7. **Test the Agent:**
   - Test the trained agent on the testing environment.
   - Example:
     ```python
     max_steps_test = state_constructor_test.shape[0]
     test_agent(agent, env_test, max_steps_test)
     ```

8. **Monitor Agent Performance:**
   - Visualize the agent's performance during training and testing.
   - Example:
     ```python
    
     agent.plot_train_result()
   
     agent.plot_test_result()
     ```

9. **Evaluate Agent Performance:**
   - Evaluate the agent's performance for both training and testing data.
   - Example:
     ```python
       agent.evaluation(StatesConstruction_.y_true_train,StatesConstruction_.y_true_test,agent.action_train,agent.action_test)
     ```

10. **Save and Load the Agent:**
   - Save the trained agent to a file and load it back for future use. Add the following code to your `main.py`:

     ```python
     # Save the trained agent
     agent.save_agent(filename=f'{datetime.date()} DQN.pkl')
     
     # Load the agent
     agent.load_agent(filename=f'{datetime.date()} DQN.pkl')
     ```
11. **Retrain the Agent:**
   - Retrain the agent by providing a saved agent path. Add the following code to your `main.py`
   
   :

     ```python
     # Create a new agent
     new_agent = DQNAgent(new_env_train, new_env_test)
     
     # Train the new agent using the saved agent from the specified path
     new_agent.train(max_steps_train, num_episodes_train, load_agent_filename='old_save_agent.pkl')
     ```
## Demonstration

To run a demonstration of training and evaluating the trading agent using the provided script `main_RL.py`, follow these steps:

1. Open a terminal and navigate to the root directory of the library.

2. Run the following command:

   ```bash
   python main_RL.py
   
   ```
This script contains the complete code for loading financial data, defining parameters, training the agent, testing the agent, and evaluating its performance.

Observe the output in the terminal for information on the training and testing process, as well as visualizations of the agent's performance.

Note: Ensure that you have the necessary dependencies installed before running the demonstration script. You can install the required packages by running:

```bash
pip install -r requirements.txt
```


## Credits

- **Author:** Jayraj Rajput
- **Email:** jayrajput1997@gmail.com
- **GitHub:** https://github.com/Jayraj-333-ML-AI-DRL

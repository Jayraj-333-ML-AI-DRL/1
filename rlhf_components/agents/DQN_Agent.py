import random
import numpy as np
import tensorflow as tf
import pickle
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import calculate_metrics, plot_results
from sklearn.metrics import precision_recall_fscore_support
from collections import deque

class DQNAgent:

    def __init__(self, environment_train, environment_test):
        """
        Initializes the DQNAgent.

        Args:
            environment_train: The training environment for the agent.
            environment_test: The testing environment for the agent.
        """
        self.environment_train = environment_train
        self.environment_test = environment_test
        self.state_dim = environment_train.states_data.shape[1]
        self.actions = ['buy', 'short', 'hold']
        self.action_dim = len(self.actions)
        self.epsilon = 0.5  # Exploration-exploitation trade-off
        self.gamma = 0.99  # Discount factor for future rewards

        # Initialize Q-network and optimizer
        self.q_network = self.build_q_network()
        self.optimizer = Adam(learning_rate=0.001)

        # Lists to store training and testing results
        self.episode_rewards_train = []
        self.losses_train = []
        self.episode_rewards_test = []
        self.losses_test = []
        self.action_train = []
        self.action_test = []
        self.all_reward_train = []
        self.all_reward_test = []

        # Replay memory to store experiences
        self.memory = deque(maxlen=20000)

    def build_q_network(self):
        """
        Builds the Q-network model.

        Returns:
            The Q-network model.
        """
        q_network = Sequential([
            Dense(32, activation='relu', input_shape=(self.state_dim + 1,)),
            Dense(8, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        q_network.compile(optimizer="Adam", loss='mean_squared_error')
        return q_network

    def remember(self, state, action_index, reward, next_state, done):
        """
        Store the experience in the replay memory.

        Args:
            state: Current state.
            action_index: Index of the selected action.
            reward: Obtained reward.
            next_state: Next state.
            done: Flag indicating the end of the episode.
        """
        self.memory.append((state, action_index, reward, next_state, done))

    def replay(self, batch_size):
        """
        Replay a batch of experiences from the replay memory.

        Args:
            batch_size: Number of experiences to replay.
        """
        minibatch = random.sample(self.memory, batch_size)

        for state, action_index, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Update Q-values using the Bellman equation
                q_values = self.q_network.predict(state[None, :])[0]
                next_q_values = self.q_network.predict(next_state[None, :])[0]
                target = reward + self.gamma * np.max(next_q_values)

                q_target = q_values.copy()
                q_target[action_index] = target

                # Train the Q-network
                loss = self.q_network.train_on_batch(state[None, :], q_target.reshape(1, -1))
                self.losses_train.append(loss)

    def train(self, max_steps, num_episodes_train, batch_size=216, load_agent_filename=None):
        """
        Train the DQNAgent.

        Args:
            num_episodes_train: Number of episodes to train.
            max_steps: Maximum number of steps per episode.
            batch_size: Size of the replay memory batch.
            load_agent_filename: Filename to load the agent from.
        """
        if load_agent_filename:
            # Load the agent and its Q-network weights
            loaded_agent = DQNAgent.load_agent(load_agent_filename)
            self.q_network.set_weights(loaded_agent.q_network.get_weights())
            print(f"Q-network weights loaded from {load_agent_filename}")

        for episode in range(num_episodes_train):
            state = self.environment_train.reset()
            print(f'Initial state: {state}')
            episode_reward = 0

            for i in range(max_steps - 1):
                print(f'Current state: {state}')
                action, action_index = self.select_action(state)
                print(f'Random action: {action}')
                self.action_train.append(action)
                next_state, reward, done = self.environment_train.step(action, i)
                self.all_reward_train.append(reward)

                print(f'Next state: {next_state}')
                print(f'Reward: {reward}')
                # Add the experience to the replay memory
                self.remember(state, action_index, reward, next_state, done)

                episode_reward += reward
                state = next_state
                print(f'Step number: {i}')
                if done:
                    break

            # Sample a batch from the replay memory and perform a replay step
            if len(self.memory) > batch_size:
                self.replay(batch_size)

            self.epsilon = max(0.1, self.epsilon - 0.1)  # Ensure epsilon doesn't go below 0.1
            self.episode_rewards_train.append(episode_reward)
            print(f"Training Episode {episode + 1}/{num_episodes_train}, Reward: {episode_reward}")

    def select_action(self, state):
        """
        Select an action using epsilon-greedy strategy.

        Args:
            state: Current state.

        Returns:
            chosen_action: Selected action.
            action_index: Index of the selected action.
        """
        if random.random() < self.epsilon:
            chosen_action = random.choice(self.actions)
            return chosen_action, self.actions.index(chosen_action)
        else:
            q_values = self.q_network.predict(state[None, :])[0]
            action_index = np.argmax(q_values)
            return self.actions[action_index], action_index

    def select_action_test(self, state):
        """
        Select an action for testing.

        Args:
            state: Current state.

        Returns:
            chosen_action: Selected action.
        """
        q_values = self.q_network.predict(state[None, :])[0]
        action_index = np.argmax(q_values)
        return self.actions[action_index]

    def test(self, max_test_steps, num_episodes_test=1):
        """
        Test the DQNAgent.

        Args:
            num_episodes_test: Number of episodes to test, always 1.
            max_test_steps: Maximum number of steps per episode.
        """
        for episode in range(num_episodes_test):
            state = self.environment_test.reset()
            episode_reward = 0

            for i in range(max_test_steps - 1):
                action = self.select_action_test(state)
                self.action_test.append(action)

                next_state, reward, done = self.environment_test.step(action, i)
                self.all_reward_test.append(reward)

                episode_reward += reward
                state = next_state

                if done:
                    break

            self.episode_rewards_test.append(episode_reward)
            print(f"Test Episode {episode + 1}/{num_episodes_test}, Reward: {episode_reward}")

    def plot_train_result(self):
        """
        Plot training results.
        """
        plot_results(self.all_reward_train, self.action_train, self.environment_train.portfolio_value, self.losses_train)

    def plot_test_result(self):
        """
        Plot testing results.
        """
        plot_results(self.all_reward_test, self.action_test, self.environment_test.portfolio_value, self.losses_test)

    def evaluation(self, y_true_train, y_true_test, agent_action_train, agent_actions_test):
        """
        Evaluate the performance of the agent.

        Args:
            y_true_train: True labels for training.
            y_true_test: True labels for testing.
            agent_action_train: Predicted actions for training.
            agent_actions_test: Predicted actions for testing.
        """
        y_true_test = y_true_test[:-1]  # Adjust if needed
        print('Testing Evaluation ')
        calculate_metrics(y_true_test, agent_actions_test)

    def save_agent(self, filename='agent.pkl'):
        """
        Save the agent, including Q-network weights.

        Args:
            filename: Filename to save the agent to.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Agent saved to {filename}")

    @staticmethod
    def load_agent(filename='agent.pkl'):
        """
        Load the agent, including Q-network weights.

        Args:
            filename: Filename to load the agent from.

        Returns:
            Loaded agent.
        """
        with open(filename, 'rb') as file:
            loaded_agent = pickle.load(file)
        print(f"Agent loaded from {filename}")
        return loaded_agent

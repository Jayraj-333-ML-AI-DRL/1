import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

class StatesConstruction:
    
    def __init__(self, train_df, test_df):
        """
        Constructor for the StatesConstruction class.

        Parameters:
        - train_df (pd.DataFrame): Training DataFrame containing stock data.
        - test_df (pd.DataFrame): Testing DataFrame containing stock data.
        """
        self.train_df = train_df
        self.test_df = test_df
        self.train_states = self.get_initial_states(train_df)
        self.test_states = self.get_initial_states(test_df)
        self.best_model, self.predictions_X, self.modal_accuracy = self.train_and_evaluate(self.train_states)
        self.train_state_final = self.get_train_state()
        self.test_state_final = self.get_test_state()
        self.y_true_train = self.generate_y_true(self.train_states)
        self.y_true_test = self.generate_y_true(self.test_states)
    
        print(f'best_model: {self.best_model}')
        print(f'modal_Mean_squared_error: {self.modal_accuracy}')
      
    def get_initial_states(self, df):
        """"
        Generates a new DataFrame with additional financial state columns.
    
        Args:
            df (pd.DataFrame): Input DataFrame containing financial data with 'Close' and 'Open' columns.

        Returns:
            pd.DataFrame: New DataFrame with added financial state columns.
            The resulting DataFrame will have reduced rows based on the maximum window size used.
        """
        # Create a new DataFrame with the 'Open' column
        new_dataframe = pd.DataFrame(df['Open'])
    
        # Add columns representing closing prices from previous days
        new_dataframe['Close_Yesterday_1'] = df['Close'].shift(1)
        new_dataframe['Close_Yesterday_2'] = df['Close'].shift(2)
        new_dataframe['Close_Yesterday_3'] = df['Close'].shift(3)
    
        # Add columns representing rolling mean values from previous days
        new_dataframe['Rolling_Mean_3_previous_days'] = df['Close'].shift(1).rolling(window=3).mean()
        new_dataframe['Rolling_Mean_5_previous_days'] = df['Close'].shift(1).rolling(window=5).mean()
        new_dataframe['Rolling_Mean_7_previous_days'] = df['Close'].shift(1).rolling(window=7).mean()
    
        # Add a column representing the next day's opening value
        new_dataframe['next_day_value'] = df['Open'].shift(-1)
    
        # Drop rows with NaN values
        new_dataframe = new_dataframe.dropna()
    
        return new_dataframe

    
    def get_train_state(self):
        train_state_final = self.train_states.drop('next_day_value', axis=1)
        train_state_final['predicted_next_day_value'] = self.predictions_X
        return train_state_final
    
    def get_test_state(self):
        test_states_final = self.test_states.drop('next_day_value', axis=1)
        test_states_final['predicted_next_day_value'] = self.best_model.predict(test_states_final)
        return test_states_final
    
    def get_state(self):
        return self.train_state_final, self.test_state_final
    
    def train_and_evaluate(self, df):
        """
        Train and evaluate an XGBoost model.

        Parameters:
        - df (pd.DataFrame): DataFrame for training and evaluation.

        Returns:
        - tuple: (best_model, predictions_X, modal_accuracy)
        """
        target_column = 'next_day_value'
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300],
            # Add more hyperparameters as needed
        }

        model = xgb.XGBRegressor()
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions_X = best_model.predict(X)
        predictions_test = best_model.predict(X_test)
        modal_accuracy = mean_squared_error(y_test, predictions_test)

        return best_model, predictions_X, modal_accuracy
    
    def generate_y_true(self, df):
        """
        Generate y_true array based on the logic:
        - If tomorrow's open price is higher than today's, label as 'buy'
        - If tomorrow's open price is lower than today's, label as 'short'
        - Otherwise, label as 'hold'

        Parameters:
        - df (pd.DataFrame): DataFrame containing open prices for today and next_day_value

        Returns:
        - list: Array of labels ('buy', 'short', 'hold') based on the logic
        """
        y_true = []
        df['action'] = np.select([df['next_day_value'] > df['Open'], df['next_day_value'] < df['Open']],
                                 ['buy', 'short'], default='hold')
        y_true = df['action'].tolist()
        return y_true

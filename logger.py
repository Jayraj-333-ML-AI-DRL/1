import logging 
import os
from datetime import datetime

# Set up logging configuration
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%m_%S')}.log"

# Create a directory for storing log files
log_path = os.path.join(os.getcwd(), "logging", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the log file path
    format='[%(asctime)s ] %(levelname)-8s %(name)-15s %(message)s',  # Define the log message format
    level=logging.INFO  # Set the logging level to INFO
)

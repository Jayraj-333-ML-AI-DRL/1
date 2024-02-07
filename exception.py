from RL4Trading.logger import logging
import os
import sys

def error_msg_details(error, error_details: sys):
    """
    Create a custom error message with details.

    Parameters:
    - error: Exception
        The exception that occurred.
    - error_details: sys
        The details of the error.

    Returns:
    - str
        Custom error message with file name, line number, and error message.
    """
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_msg

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        """
        Initialize a custom exception.

        Parameters:
        - error_message: str
            The error message.
        - error_details: sys
            The details of the error.
        """
        super().__init__(error_message)
        self.error_message = error_msg_details(error_message, error_details)
        
    def __str__(self):
        """
        Return the custom error message as a string.

        Returns:
        - str
            Custom error message with file name, line number, and error message.
        """
        return self.error_message

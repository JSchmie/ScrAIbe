"""
global_var.py

This module stores global variables for the app.

Global variables:
    REQUEST_QUEUE (multiprocessing.Queue): A queue to store audio file paths as strings.
    RESPONSE_QUEUE (multiprocessing.Queue): A queue to store transcriptions as strings.
    LAST_ACTIVE_TIME (multiprocessing.Value): A value to store the time of the last activity.
    LOADED_EVENT (multiprocessing.Event): An event to indicate when the model is loaded.
    RUNNING_EVENT (multiprocessing.Event): An event to indicate when the model is running.
    MODEL_PARAMS (Optional[dict]): A dictionary to store the model parameters.
    MODEL_PROCESS (Optional[multiprocessing.Process]): A process to handle the model globally.
    LAST_USED (float): A float to track the time of the last user activity.
    TIMEOUT (Optional[int]): An integer to store the timeout in seconds.
    DEFAULT_APP_CONIFG_PATH (str): A string to store the default path to the app configuration file.
"""

import multiprocessing
import os
import time
from typing import Optional

REQUEST_QUEUE: multiprocessing.Queue = multiprocessing.Queue()  # audio file path as string 
RESPONSE_QUEUE: multiprocessing.Queue = multiprocessing.Queue()  # transcription as string
LAST_ACTIVE_TIME: multiprocessing.Value = multiprocessing.Value('d', time.time())  # time of last activity
LOADED_EVENT: multiprocessing.Event = multiprocessing.Event()  # model loaded event
RUNNING_EVENT: multiprocessing.Event = multiprocessing.Event()  # model running event

MODEL_PARAMS: Optional[dict] = None  # model parameters
MODEL_PROCESS: Optional[multiprocessing.Process] = None  # model process to handle globally

# Global variable to track user activity
LAST_USED: float = time.time()
TIMEOUT: Optional[int] = None  # seconds

DEFAULT_APP_CONIFG_PATH: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml")
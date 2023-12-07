"""
Stores global variables for the app.
"""

# Global variable to store the model
import multiprocessing
import os
import time
import yaml

REQUEST_QUEUE = multiprocessing.Queue() # audio file path as string 
RESPONSE_QUEUE = multiprocessing.Queue() # transcription as string
LAST_ACTIVE_TIME = multiprocessing.Value('d', time.time()) # time of last activity
LOADED_EVENT = multiprocessing.Event() # model loaded event
RUNNING_EVENT = multiprocessing.Event() # model running event

MODEL_PARAMS = None # model parameters
MODEL_PROCESS = None # model process to handle globally

# Global variable to track user activity
LAST_USED = time.time()
TIMEOUT = None #seconds

DEFAULT_APP_CONIFG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yml")

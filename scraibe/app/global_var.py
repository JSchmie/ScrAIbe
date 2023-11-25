"""
Stores global variables for the app.
"""

# Global variable to store the model
from threading import Event

import time


MODEL = None
MODEL_THREAD_PARAMS = None
MODEL_THREAD = None

# Global variable to track user activity
LAST_USED = time.time()
TIMEOUT = 30 #seconds
TRANSCRIBE_ACTIVE = Event()
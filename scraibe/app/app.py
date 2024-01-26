"""
Gradio App.
--------------------------------

This module provides an interface to transcribe audio files using the 
Scraibe model. Users can either upload an audio file or record their speech 
live for transcription. The application supports multiple languages and provides 
options to specify the number of speakers and the language of the audio.

Attributes:
    LANGUAGES (list): A list of supported languages for transcription.

Usage:
    Run this script to start the Gradio web interface for audio transcription.
    
"""


####
# Gradio Interface
####

from threading import Thread

import scraibe.app.global_var as gv
from .interface import gradio_Interface
from .multi import *
from .utils import *


def app(config : str = None, **kwargs):
    """
    Launches the Gradio interface for audio transcription.
    
    Args:
        interface_params (dict): A dictionary of parameters for the Gradio interface.
        queue_params (dict): A dictionary of parameters for the queue.
        launch_params (dict): A dictionary of parameters for launching the interface.
    
    Returns:
        None
    
    """
    
    # Load the configuration
    
    config = AppConfig.load_config(config, **kwargs)
    
    
    gv.MODEL_PROCESS = start_model_worker(gv.MODEL_PARAMS,
                                        gv.REQUEST_QUEUE,
                                        gv.LAST_ACTIVE_TIME,
                                        gv.RESPONSE_QUEUE,
                                        gv.LOADED_EVENT,
                                        gv.RUNNING_EVENT)
    
    timer = Thread(target=timer_thread, args=(gv.REQUEST_QUEUE,
                                            gv.LAST_ACTIVE_TIME,
                                            gv.LOADED_EVENT,
                                            gv.RUNNING_EVENT,
                                            gv.TIMEOUT), daemon=True)
    
    layout = config.get_layout()
    
    timer.start()
    
    print("Starting Gradio Web Interface")
    
    gradio_Interface(layout).queue(**config.queue).launch(**config.launch)

    timer.join()
    gv.MODEL_PROCESS.join()
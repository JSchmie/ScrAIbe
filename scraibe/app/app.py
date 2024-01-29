"""
Gradio App
----------

This module provides an interface to transcribe audio files using the 
Scraibe model. Users can either upload an audio file or record their speech 
live for transcription. The application supports multiple languages and provides 
options to specify the number of speakers and the language of the audio. It also 
enables efficient management of resources by loading and unloading AI models 
based on usage.

The configuration is managed via a 'config.yml' file, which allows customization
of various aspects of the application, including the Gradio interface, queue
management, and model parameters.

Configuration Sections in 'config.yml':
- launch: Settings for launching the interface, such as server port, authentication, SSL configuration.
- queue: Configuration for managing request handling and concurrency.
- layout: Customization options for the interface layout, like headers, footers, and logos.
- model: Specifications for different AI models used in transcription.
- advanced: Advanced settings, including session timeout duration.

Note: 
    The .queue function of the Gradio interface is currently experiencing issues 
    and might not work as expected. 

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

    Initializes the Gradio web interface with settings from a YAML configuration file
    and/or keyword arguments. The function manages AI models, handling their loading 
    into RAM and unloading after a session or specified timeout.

    The `kwargs` are used to override or supplement values from the `config.yml` file.
    They should follow the structure of `config.yml`, which includes sections like 
    'launch', 'queue', 'layout', 'model', and 'advanced'.

    Args:
        config (str): Path to the YAML configuration file. Default settings are used 
                      if not provided.
        **kwargs: Keyword arguments corresponding to the configuration sections. Each 
                  argument should be a dictionary reflecting the structure of its 
                  respective section in `config.yml`.

    Returns:
        None
    """

    # Load and override configuration from the YAML file with kwargs
    
    config = AppConfig.load_config(config, **kwargs)
    
    
    gv.MODEL_PROCESS = start_model_worker(gv.MODEL_PARAMS,
                                        gv.REQUEST_QUEUE,
                                        gv.LAST_ACTIVE_TIME,
                                        gv.RESPONSE_QUEUE,
                                        gv.LOADED_EVENT,
                                        gv.RUNNING_EVENT)
    
    # Set the timer thread to manage model loading and unloading
    timer = Thread(target=timer_thread, args=(gv.REQUEST_QUEUE,
                                            gv.LAST_ACTIVE_TIME,
                                            gv.LOADED_EVENT,
                                            gv.RUNNING_EVENT,
                                            gv.TIMEOUT), daemon=True)
    
    # Set the layout for the Gradio interface
    layout = config.get_layout()
    
    # start the timer thread
    timer.start()
    
    print("Starting Gradio Web Interface")
    
    # Launch the Gradio interface
    gradio_Interface(layout).queue(**config.queue).launch(**config.launch)

    # Wait for the timer thread to finish
    timer.join()
    gv.MODEL_PROCESS.join()
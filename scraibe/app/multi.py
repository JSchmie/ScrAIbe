"""
This module contains functions for managing and optimizing the resource usage of the application.

The functions in this module monitor the application's usage and make adjustments to improve efficiency. 
This includes managing the loading and unloading of the model based on the application's activity.
This dynamic management of resources helps to ensure that the application uses only the resources it needs,
improving overall performance and reducing unnecessary resource consumption.

Functions:
    clear_queue(queue): Clears all items from the queue.
    model_worker(model_params, request_queue, last_active_time,
                response_queue, loaded_event, running_event, *args, **kwargs): Manages the model worker process.

Modules:
    time: Provides various time-related functions.
    gc: Provides an interface to the garbage collector.
    multiprocessing: Provides support for parallel execution of code.
    torch: Provides tensor computation and deep learning functionality.
    gradio: Provides a simple way to create interactive UIs for Python functions.
    scraibe.autotranscript: Provides automatic transcription functionality.
    .stg: Contains the GradioTranscriptionInterface class.
"""


import time
import gc
from typing import Union, Any
import multiprocessing
import torch

from gradio import Warning
from scraibe.autotranscript import Scraibe
from .stg import GradioTranscriptionInterface 

def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except queue.Empty:
            continue
        
def model_worker(model_params : Union[Scraibe, dict],
                 request_queue: multiprocessing.Queue,
                 last_active_time: multiprocessing.Value,
                 response_queue: multiprocessing.Queue,
                 loaded_event: multiprocessing.Event,
                 running_event: multiprocessing.Event,
                 *args: Any, **kwargs: Any) -> None:
    """
    Manages the model worker process.

    The model worker process is responsible for running the model and returning the results.

    Args:
        model_params (Union[Scraibe, dict]): The parameters for the Scraibe model.
        request_queue (multiprocessing.Queue): The queue for incoming requests.
        last_active_time (multiprocessing.Value): The last time the model was active.
        response_queue (multiprocessing.Queue): The queue for outgoing responses.
        loaded_event (multiprocessing.Event): An event that signals when the model is loaded.
        running_event (multiprocessing.Event): An event that signals when the model is running.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.
    """
    
    loaded_event.set()
    
    if model_params is None:
        _model = Scraibe()
    elif type(model_params) is Scraibe:
        _model = model_params
    elif type(model_params) is dict:
        _model = Scraibe(**model_params)
    else:
        raise TypeError("model must be of type Scraibe, or dict")

    model = GradioTranscriptionInterface(_model)
    
    while True:
        
        req = request_queue.get()

        if req == "STOP":
            
            break
        elif type(req) is dict:
            runner = model.get_task_from_str(req.pop("task"))
            running_event.set()
            transcription = runner(**req)
            running_event.clear()
            response_queue.put(transcription)
            last_active_time.value = time.time()
        else:
            raise TypeError("request must be of type dict")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    clear_queue(request_queue)
    clear_queue(response_queue)
    loaded_event.clear()

def start_model_worker(model_params: Union[Scraibe, dict],
                       request_queue: multiprocessing.Queue,
                       last_active_time: multiprocessing.Value,
                       response_queue: multiprocessing.Queue,
                       loaded_event: multiprocessing.Event,
                       running_event: multiprocessing.Event,
                       *args: Any, **kwargs: Any) -> multiprocessing.Process:
    """
    Starts the model worker process.

    Args:
        model_params (Union[Scraibe, dict]): The parameters for the Scraibe model.
        request_queue (multiprocessing.Queue): The queue for incoming requests.
        last_active_time (multiprocessing.Value): The last time the model was active.
        response_queue (multiprocessing.Queue): The queue for outgoing responses.
        loaded_event (multiprocessing.Event): An event that signals when the model is loaded.
        running_event (multiprocessing.Event): An event that signals when the model is running.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        multiprocessing.Process: The model worker process.
    """
    context = multiprocessing.get_context('spawn')
    model_process = context.Process(target=model_worker, args=(model_params, request_queue, last_active_time, response_queue,loaded_event, running_event, *args), kwargs=kwargs)
    model_process.start()
    return model_process

def timer_thread(request_queue: multiprocessing.Queue,
                 last_active_time: multiprocessing.Value,
                 loaded_event: multiprocessing.Event,
                 running_event: multiprocessing.Event,
                 timeout: int) -> None:
    """
    Monitors the model worker process and stops it after a period of inactivity.

    Args:
        request_queue (multiprocessing.Queue): The queue for incoming requests.
        last_active_time (multiprocessing.Value): The last time the model was active.
        loaded_event (multiprocessing.Event): An event that signals when the model is loaded.
        running_event (multiprocessing.Event): An event that signals when the model is running.
        timeout (int): The period of inactivity after which the model worker process is stopped.
    """
    while True:
        time.sleep(timeout)
        
        if time.time() - last_active_time.value > timeout and loaded_event.is_set() and not running_event.is_set():
            print(f"No activity for the last {timeout} seconds. Stopping the model worker.", flush=True)
            request_queue.put("STOP")
            Warning("Model worker stopped due to inactivity.")
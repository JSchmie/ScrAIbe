"""
This file contains the functions which are related to monitoring the actual app usage. 
Therefore, the app is to be more efficient in the usage of the resources. 
By for example, unloading or reloading the model.
"""



import time
import gc
from typing import Union
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
                 request_queue,
                 last_active_time,
                 response_queue,
                 loaded_event,
                 running_event,
                 *args, **kwargs):
    
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

def start_model_worker(model_params, request_queue, last_active_time, response_queue,loaded_event, running_event, *args, **kwargs):
    context = multiprocessing.get_context('spawn')
    model_process = context.Process(target=model_worker, args=(model_params, request_queue, last_active_time, response_queue,loaded_event, running_event, *args), kwargs=kwargs)
    model_process.start()
    return model_process

def timer_thread(request_queue, last_active_time,loaded_event, running_event, timeout=30):
    while True:
        time.sleep(timeout)
        
        if time.time() - last_active_time.value > timeout and loaded_event.is_set() and not running_event.is_set():
            print(f"No activity for the last {timeout} seconds. Stopping the model worker.", flush=True)
            request_queue.put("STOP")
            Warning("Model worker stopped due to inactivity.")
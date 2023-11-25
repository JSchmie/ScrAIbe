"""
This file contains the functions which are related to monitoring the actual app usage. 
Therefore, the app is to be more efficient in the usage of the resources. 
By for example, unloading or reloading the model.
"""

import time
import gc
from typing import Union
import torch

import scraibe.app.global_var as gv
from scraibe.autotranscript import Scraibe 


def load_model_thread(model : Union[Scraibe, dict] = None):
    if model is None:
        gv.MODEL = Scraibe()
    elif type(model) is Scraibe:
        gv.MODEL = model
    elif type(model) is dict:
        gv.MODEL = Scraibe(**model)
    else:
        raise TypeError("model must be of type Scraibe, or dict")

    gv.LAST_USED = time.time()

# Create a thread to monitor user activity
def delete_unused_model():
    while True:
        
        _unload_porperty = (not gv.TRANSCRIBE_ACTIVE.is_set() and (time.time() - gv.LAST_USED > gv.TIMEOUT) and gv.MODEL is not None)

        if _unload_porperty:
            
            del gv.MODEL
            gv.MODEL = None
            
            gc.collect()
            torch.cuda.empty_cache()

            gv.MODEL_THREAD.join()
            
        time.sleep(int(gv.TIMEOUT/5))

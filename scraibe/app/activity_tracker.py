"""
This file contains the functions which are related to monitoring the actual app usage. 
Therefore, the app is to be more efficient in the usage of the resources. 
By for example, unloading or reloading the model.
"""
import time
import threading
import torch
import gc
import gradio as gr


timeout = 30 #seconds
USER_ACTIVE = True
user_active_lock = threading.Lock() # dummy for now

# Create a thread to monitor user activity
def monitor_activity(model, pipe, timeout=timeout):
    global USER_ACTIVE
    
    while True:
        time.sleep(timeout)  # Check user activity every second
        with user_active_lock:
            
            if not USER_ACTIVE:
                del model
                del pipe
                
                gc.collect()
                torch.cuda.empty_cache()
                
                
                
                print("Model deleted empty memory")
                gr.Warning("Model unloaded due to inactivity. Please reload the model to continue.")
                break
            USER_ACTIVE = False 
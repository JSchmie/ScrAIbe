import time

from scraibe import Scraibe
import threading
import torch
import gc

model = None
last_used = time.time()
transcribe_active = threading.Event()

def transcribe_thread(audio):
    
    global model 
    transcribe_active.set()
    print(model.autotranscribe(audio))
    transcribe_active.clear()
    
def model_thread():
    global model, last_used
    model = Scraibe(dia_model= "models/pyannote/config.yaml")
    last_used = time.time()
    
def interaction_thread():
    global model
    while True:
        command = input("Enter a command ('q' to quit, 'reload' to reload model): ")
        print("Command entered:", command, command.lower() == 'reload')
        if command.lower() == 'q':
            break
        elif command.lower() == 'reload':
            print("Reloading model...", model)
            if model is None:
                model_runner = threading.Thread(target=model_thread)
                model_runner.start()
                model_runner.join()
            else:
                print("Model is already loaded.")
        else:
            transcribe = threading.Thread(target=transcribe_thread, args=(command,))
            transcribe.start()
            transcribe.join()

def delete_unused_model(model_runner):
    global model, last_used, transcribe_active
    while True:
        if not transcribe_active.is_set() and (time.time() - last_used > 30) and model is not None:
            
            del model
            model = None
            
            gc.collect()
            torch.cuda.empty_cache()

            model_runner.join()
            print("Model deleted", threading.active_count())
        time.sleep(1)

if __name__ == "__main__":
    
    lock = threading.Lock()
    
    interaction = threading.Thread(target=interaction_thread)
    model_runner = threading.Thread(target=model_thread)
    model_deleter = threading.Thread(target=delete_unused_model, args=(model_runner,))
    
    model_runner.start()
    model_deleter.start()

    # Ensure the model is initialized before starting the interaction
    model_runner.join()
    interaction.start()
    interaction.join()
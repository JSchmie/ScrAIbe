import multiprocessing
import os
import threading
import queue
import time
import torch
from scraibe import Scraibe

def input_thread(input_queue, processed_event):
    while True:
        processed_event.wait()  # Wait for the previous input to be processed
        processed_event.clear()  # Clear the event for the next input
        inp = input("Enter the path to the audio file ('q' to quit, 'reload' to reload model): ")
        input_queue.put(inp)

def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except queue.Empty:
            continue

def model_worker(request_queue, last_active_time, response_queue,loaded_event, running_event):
    
    loaded_event.set()
    
    model = Scraibe(dia_model="models/pyannote/config.yaml")
    
    while True:
        audio_path = request_queue.get()
        if audio_path == "STOP":
            break
        running_event.set()
        transcription = model.autotranscribe(audio_path)
        running_event.clear()
        response_queue.put(transcription)
        last_active_time.value = time.time()

    del model
    torch.cuda.empty_cache()
    clear_queue(request_queue)
    clear_queue(response_queue)
    loaded_event.clear()
    

def start_model_worker(request_queue, last_active_time, response_queue,loaded_event, running_event):
    model_process = multiprocessing.Process(target=model_worker, args=(request_queue, last_active_time, response_queue,loaded_event, running_event))
    model_process.start()
    return model_process

def timer_thread(request_queue, last_active_time,loaded_event, running_event, timeout=30):
    while True:
        time.sleep(timeout)
        
        if time.time() - last_active_time.value > timeout and loaded_event.is_set() and not running_event.is_set():
            print(f"No activity for the last {timeout} seconds. Stopping the model worker.", flush=True)
            request_queue.put("STOP")

if __name__ == "__main__":
    request_queue = multiprocessing.Queue()
    response_queue = multiprocessing.Queue()
    input_queue = queue.Queue()
    last_active_time = multiprocessing.Value('d', time.time())
    loaded_event = multiprocessing.Event()
    running_event = multiprocessing.Event()
    
    processed_event = multiprocessing.Event()
    processed_event.set()  # Initially set to allow the first input

    model_process = start_model_worker(request_queue, last_active_time, response_queue,loaded_event ,running_event)
    timer = threading.Thread(target=timer_thread, args=(request_queue, last_active_time, loaded_event, running_event), daemon=True)
    input_handler = threading.Thread(target=input_thread, args=(input_queue,processed_event))

    timer.start()
    input_handler.start()

    while True:
        
        audio_file_path = input_queue.get()  # Get input from the input thread
        print(audio_file_path)
        
        if audio_file_path.lower() == 'q':
            request_queue.put("STOP")
            model_process.join()
            break
        elif audio_file_path.lower() == 'reload':
            if loaded_event.is_set():
                request_queue.put("STOP")
                model_process.join()
            model_process = start_model_worker(request_queue, last_active_time, response_queue, loaded_event, running_event)
            print("Model reloaded.")
        elif not os.path.exists(audio_file_path):
            print("File does not exist.")
        else:
            if not loaded_event.is_set():
                model_process = start_model_worker(request_queue, last_active_time, response_queue, loaded_event, running_event)
            request_queue.put(audio_file_path)
            transcription = response_queue.get()
            print(transcription)
        
        processed_event.set()  # Signal that the input has been processed

    model_process.join()
    timer.join()
    input_handler.join()

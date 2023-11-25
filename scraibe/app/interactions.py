"""
This file contains ervery function that will be called when the user interacts with the 
UI like pressing a button or uploading a file.
"""

import time
import gradio as gr 
import scraibe.app.global_var as gv
from scraibe import Transcript
from scraibe.app.stg import GradioTranscriptionInterface
import threading

def select_task(choice):
        # tell the app that it is still in use
    if choice == 'Auto Transcribe':
        
        return (gr.update(visible = True),
                gr.update(visible = True),
                gr.update(visible = True))
                
        
    elif choice == 'Transcribe':
        
        return (gr.update(visible = False),
                gr.update(visible = True),
                gr.update(visible = True))

        
    elif choice == 'Diarisation':
        
        return (gr.update(visible = True),
                gr.update(visible = False),
                gr.update(visible = False))
        
def select_origin(choice):
        
    # tell the app that it is still in use
    if choice == "Upload Audio":
        
        return (gr.update(visible = True),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None))
    
    elif choice == "Record Audio":
        
        return (gr.update(visible = False, value = None),
                gr.update(visible = True),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None))

    elif choice == "Upload Video":
        
        return (gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = True),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None))
    
    elif choice == "Record Video":
        
        return (gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = True),
                gr.update(visible = False, value = None))
        
    elif choice == "File or Files":
        
        return (gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = False, value = None),
                gr.update(visible = True))
        
def run_scraibe(task,
                num_speakers,
                translate,
                language,
                audio1,
                audio2,
                video1,
                video2,
                file_in,
                progress = gr.Progress(track_tqdm= True)):
    
    # get *args which are not None 
    
    if gv.MODEL is None and gv.MODEL_THREAD_PARAMS is not None:
        progress(0, desc='Model was not loaded to conserve resources. Loading model...')
        time.sleep(1)
        gv.MODEL_THREAD = threading.Thread(**gv.MODEL_THREAD_PARAMS)
        gv.MODEL_THREAD.start()
        gv.MODEL_THREAD.join()
      
    pipe = GradioTranscriptionInterface()
        
    progress(0.1, desc='Starting task...')
    source = audio1 or audio2 or video1 or video2 or file_in
    
    if isinstance(source, list):
        source = [s.name for s in source]
        if len(source) == 1:
            source = source[0]

    if task == 'Auto Transcribe':

        out_str , out_json = pipe.auto_transcribe(source = source,
                            num_speakers = num_speakers,
                            translation = translate,
                            language = language)
        
        if isinstance(source, str):
            return (gr.update(value = out_str, visible = True),
                    gr.update(value = out_json, visible = True),
                    gr.update(visible = True),
                    gr.update(visible = True))      
        else:
            return (gr.update(value = out_str, visible = True),
                    gr.update(value = out_json, visible = True),
                    gr.update(visible = False),
                    gr.update(visible = False))  
        
    elif task == 'Transcribe':
        
        out = pipe.transcribe(source = source,
                            translation = translate,
                            language = language)
        
        return (gr.update(value = out, visible = True),
                gr.update(value = None, visible = False),
                gr.update(visible = False),
                gr.update(visible = False))
        
    elif task == 'Diarisation':
        
        out = pipe.perform_diarisation(source = source,
                            num_speakers = num_speakers)
        
        return (gr.update(value = None, visible = False),
                gr.update(value = out, visible = True),
                gr.update(visible = False),
                gr.update(visible = False))
    
def annotate_output(annoation : str, out_json : dict):
    # get *args which are not None
    
    trans = Transcript.from_json(out_json)
    trans = trans.annotate(*annoation.split(","))

    return gr.update(value = str(trans)),gr.update(value = trans.get_json())


"""
stg - scraibe to gradio interface

This file contains the code for the scraibe to gradio interface.
It makes adds gradio interactions to the scraibe class in the back.

"""

import json
import gradio as gr
from tqdm import tqdm

import scraibe.app.global_var as gv


class GradioTranscriptionInterface:
    """
    Interface handling the interaction between Gradio UI and the Audio Transcription system.
    """

    def __init__(self, model):
        """
        Initializes the GradioTranscriptionInterface with a transcription model.

        Args:
            model (Scraibe): Model responsible for audio transcription tasks.
        """
        self.model = model 

    def autotranscribe(self, source,
                        num_speakers : int,
                        translate : bool,
                        language : str,*args ,**kwargs):
        """
        Shortcut method for the Scraibe task.

        Returns:
            tuple: Transcribed text (str), JSON output (dict)
        """
        
        _kwargs = {
            "num_speakers": num_speakers if num_speakers != 0 else None,
            "language": language if language != "None" else None,
            "task": 'translate' if translate else None
        }
        if isinstance(source, str):
            try:
                result = self.model.autotranscribe(source, **_kwargs)
            except ValueError:
                raise gr.Error("Couldn't detect any speech in the provided audio. \
                        Please try again!")
                
            return str(result), result.get_json()
        
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Transcribing audio files"):
                try:
                    res = self.model.autotranscribe(s, **_kwargs)
                except ValueError:
                    _name = s.split("/")[-1]
                    res = f"NO TRANSCRIPT FOUND FOR {_name}"
                    gr.Warning(f"Couldn't detect any speech in {_name} will skip this file.")
                result.append(res)
            
            out = ''
            out_dict = {}
            for i, r in enumerate(result):
                out += f"TRANSCRIPT FOR {source_names[i]}:\n\n"
                out += str(r)
                out += "\n\n"
                
                if isinstance(r, str):
                    out_dict[source_names[i]] = r
                else:
                    out_dict[source_names[i]] = r.get_dict()
              
            return out, json.dumps(out_dict, indent=4)
        
        else:
            raise gr.Error("Please provide a valid audio file.")


    def transcribe(self, source, translate, language,*args ,**kwargs):
        """
        Shortcut method for the Transcribe task.

        Returns:
            str: Transcribed text.
        """

        _kwargs = {
            "language": language if language != "None" else None,
            "task": 'translate' if translate == "Yes" else None
        }
    
        if isinstance(source, str):
            result = self.model.transcribe(source, **_kwargs)

            return str(result)
        
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Transcribing audio files"):
                res = self.model.transcribe(s, **_kwargs)
                result.append(res)
            
            out = ''
            for i, res in enumerate(result):
                out += f"TRANSCRIPT FOR {source_names[i]}:\n\n"
                out += str(res)
                out += "\n\n"
            
            return out
        
        else:
            raise gr.Error("Please provide a valid audio file.")

    def diarisation(self, source, num_speakers, *args ,**kwargs):
        """
        Shortcut method for the Diarisation task.

        Returns:
            str: JSON output of diarisation result.
        """
        
        _kwargs = {
            "num_speakers": num_speakers if num_speakers != 0 else None,
        }
        
        if isinstance(source, str):
            try:
                result = self.model.diarization(source, **_kwargs)
            except ValueError:
                raise gr.Error("Couldn't detect any speech in the provided audio. \
                        Please try again!")
                
            return json.dumps(result, indent=2)
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Performing diarisation"):
                try:
                    res = self.model.diarization(s, **_kwargs)
                except ValueError:
    
                    res = f"NO DIARISATION FOUND FOR {s}"
                    gr.Warning(f"Couldn't detect any speech in {s} will skip this file.")
                result.append(res)
            
            out = {}
            
            for i, res in enumerate(result):
                out[source_names[i]] = res
            
            return json.dumps(out, indent=4)
        
        else:
            gr.Error("Please provide a valid audio file.")
    
    def get_task_from_str(self, task):
        """
        Returns the coresponing task function based on the task string.
        
        params:
            task (str): Task string. Can be one of the following:
                - 'Auto Transcribe'
                - 'Transcribe'
                - 'Diarisation'
        """ 
        
        if task == 'Auto Transcribe':
            return self.autotranscribe
        elif task == 'Transcribe':
            return self.transcribe
        elif task == 'Diarisation':
            return self.diarisation
        else:
            raise ValueError("Invalid task string.")
        
               

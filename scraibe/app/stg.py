"""
stg - Scraibe to Gradio Interface

This module provides an interface between the Scraibe transcription system and the Gradio user interface.
It defines a class, GradioTranscriptionInterface, that wraps the Scraibe model and provides methods for performing transcription tasks through the Gradio UI.

Modules:
    json: Used for encoding and decoding JSON data.
    gradio as gr: Used for creating the Gradio UI.
    tqdm: Used for displaying progress bars.
    scraibe.app.global_var as gv: Contains global variables for the Scraibe app.
"""
import json
import gradio as gr
from tqdm import tqdm
from typing import Any, Dict, Union, Tuple, List




class GradioTranscriptionInterface:
    """
    A class that provides an interface between the Gradio UI and the Scraibe transcription system.

    This class wraps the Scraibe model and provides methods for performing transcription tasks through the Gradio UI. 
    These tasks include auto transcription, transcription, and diarisation.

    Attributes:
        model (Scraibe): The Scraibe model for performing transcription tasks.
    """

    def __init__(self, model) -> None:
        """
        Initializes the GradioTranscriptionInterface with a Scraibe model.

        Args:
            model (Scraibe): The Scraibe model for performing transcription tasks.
            *args (Any): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.
        """
            
        self.model = model

    def autotranscribe(self, source: Union[str, List[str]],
                        num_speakers: int,
                        translate: bool,
                        language: str,
                        *args: Any, **kwargs: Dict[str, Any]) -> Tuple[str, Union[str, dict]]:
        """
        Performs auto transcription on the given source.

        Args:
            source (Union[str, List[str]]): The source to transcribe. This can be a string representing a single source,
                                    or a list of strings representing multiple sources.
            num_speakers (int): The number of speakers in the source.
            translate (bool): Whether to translate the transcription.
            language (str): The language of the source.
            *args (Any): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            Tuple[str, Union[str, dict]]: A tuple containing the transcribed text (str) and the JSON output (str or dict).
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


    def transcribe(self, source: Union[str, List[str]], 
                   translate: bool,
                   language: str,
                   *args: Any, **kwargs: Dict[str, Any]) -> str:
        """
        Performs transcription on the given source.

        Args:
            source (Union[str, List[str]]): The source to transcribe.
            This can be a string representing a single source, or a list of strings representing multiple sources.
            translate (bool): Whether to translate the transcription.
            language (str): The language of the source.
            *args (Any): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            str: The transcribed text.
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

    def diarisation(self, source: Union[str, List[str]],
                    num_speakers: int,
                    *args: Any, **kwargs: Dict[str, Any]) -> str:
        """
        Performs diarisation on the given source.

        Args:
            source (Union[str, List[str]]): The source to perform diarisation on.
            This can be a string representing a single source, 
            or a list of strings representing multiple sources.
            num_speakers (int): The number of speakers in the source.
            *args (Any): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            str: The JSON output of the diarisation result.
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
    
    def get_task_from_str(self, task: str) -> callable:
        """
        Returns the corresponding task function based on the given task string.

        Args:
            task (str): The task string. This can be one of the following: 'Auto Transcribe', 'Transcribe', 'Diarisation'.

        Returns:
            callable: The corresponding task function.
        """
        
        if task == 'Auto Transcribe':
            return self.autotranscribe
        elif task == 'Transcribe':
            return self.transcribe
        elif task == 'Diarisation':
            return self.diarisation
        else:
            raise ValueError("Invalid task string.")
        
               

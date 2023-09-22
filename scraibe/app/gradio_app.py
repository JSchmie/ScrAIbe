"""
Gradio Audio Transcription App.
--------------------------------

This module provides an interface to transcribe audio files using the 
Scraibe model. Users can either upload an audio file or record their speech 
live for transcription. The application supports multiple languages and provides 
options to specify the number of speakers and the language of the audio.

Attributes:
    LANGUAGES (list): A list of supported languages for transcription.

Usage:
    Run this script to start the Gradio web interface for audio transcription.
    
"""

"""
Gradio Audio Transcription App.
--------------------------------

This module provides an interface to transcribe audio files using the 
Scraibe model. Users can either upload an audio file or record their speech 
live for transcription. The application supports multiple languages and provides 
options to specify the number of speakers and the language of the audio.

Attributes:
    LANGUAGES (list): A list of supported languages for transcription.

Usage:
    Run this script to start the Gradio web interface for audio transcription.
    
"""

import json
import os
from tkinter import CURRENT

import gradio as gr
from tqdm import tqdm

from scraibe import Scraibe, Transcript

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue='orange',
    neutral_hue="gray",  
)

LANGUAGES = [
    "Afrikaans", "Arabic", "Armenian", "Azerbaijani", "Belarusian",
    "Bosnian", "Bulgarian", "Catalan", "Chinese", "Croatian",
    "Czech", "Danish", "Dutch", "English", "Estonian",
    "Finnish", "French", "Galician", "German", "Greek",
    "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian",
    "Italian", "Japanese", "Kannada", "Kazakh", "Korean",
    "Latvian", "Lithuanian", "Macedonian", "Malay", "Marathi",
    "Maori", "Nepali", "Norwegian", "Persian", "Polish",
    "Portuguese", "Romanian", "Russian", "Serbian", "Slovak",
    "Slovenian", "Spanish", "Swahili", "Swedish", "Tagalog",
    "Tamil", "Thai", "Turkish", "Ukrainian", "Urdu",
    "Vietnamese", "Welsh"
]

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class GradioTranscriptionInterface:
    """
    Interface handling the interaction between Gradio UI and the Audio Transcription system.
    """

    def __init__(self, model: Scraibe):
        """
        Initializes the GradioTranscriptionInterface with a transcription model.

        Args:
            model (Scraibe): Model responsible for audio transcription tasks.
        """
        self.model = model

    def auto_transcribe(self, source,
                        num_speakers : int,
                        translation : bool,
                        language : str):
        """
        Shortcut method for the Scraibe task.

        Returns:
            tuple: Transcribed text (str), JSON output (dict)
        """
        
        kwargs = {
            "num_speakers": num_speakers if num_speakers != 0 else None,
            "language": language if language != "None" else None,
            "task": 'translate' if translation else None
        }
        if isinstance(source, str):
            try:
                result = self.model.autotranscribe(source, **kwargs)
            except ValueError:
                raise gr.Error("Couldn't detect any speech in the provided audio. \
                        Please try again!")
            
            return str(result), result.get_json()
        
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Transcribing audio files"):
                try:
                    res = self.model.autotranscribe(s, **kwargs)
                except ValueError:
                    _name = s.split("/")[-1]
                    res = f"NO TRANSCRIPT FOUND FOR {_name}"
                    gr.Warning(f"Couldn't detect any speech in {_name} will skip this file.")
                result.append(res)
            
            out = ''
            out_dict = {}
            for i, r in enumerate(result):
                out += f"TRANSCRIPT {i} FOR ({source_names[i]}):\n\n"
                out += str(r)
                out += "\n\n"
                
                if isinstance(r, str):
                    out_dict[source_names[i]] = r
                else:
                    out_dict[source_names[i]] = r.get_dict()
              
            return out, json.dumps(out_dict, indent=4)
        
        else:
            raise gr.Error("Please provide a valid audio file.")


    def transcribe(self, source, translation, language):
        """
        Shortcut method for the Transcribe task.

        Returns:
            str: Transcribed text.
        """
        kwargs = {
            "language": language if language != "None" else None,
            "task": 'translate' if translation == "Yes" else None
        }
        
        if isinstance(source, str):
            result = self.model.transcribe(source, **kwargs)
            
            return str(result)
        
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Transcribing audio files"):
                res = self.model.transcribe(s, **kwargs)
                result.append(res)
            
            out = ''
            for i, res in enumerate(result):
                out += f"TRANSCRIPT {i} FOR ({source_names[i]}):\n\n"
                out += str(res)
                out += "\n\n"
            
            return out
        
        else:
            raise gr.Error("Please provide a valid audio file.")

    def perform_diarisation(self, source, num_speakers):
        """
        Shortcut method for the Diarisation task.

        Returns:
            str: JSON output of diarisation result.
        """
        kwargs = {
            "num_speakers": num_speakers if num_speakers != 0 else None,
        }
        
        if isinstance(source, str):
            try:
                result = self.model.diarization(source, **kwargs)
            except ValueError:
                raise gr.Error("Couldn't detect any speech in the provided audio. \
                        Please try again!")
        
            return json.dumps(result, indent=2)
        elif isinstance(source, list):
            source_names = [s.split("/")[-1] for s in source]
            result = []
            for s in tqdm(source, total=len(source),desc = "Performing diarisation"):
                try:
                    res = self.model.diarization(s, **kwargs)
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
            

####
# Gradio Interface
####

def gradio_Interface(model : Scraibe = None):
    
    if model is None:
        model = Scraibe()
        
    pipe = GradioTranscriptionInterface(model)

    def select_task(choice):
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

    def run_scribe(task,
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
        progress(0, desc='Starting task...')
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
        
        
    with gr.Blocks(theme=theme,title='ScrAIbe: Automatic Audio Transcription') as demo:
            
        # Define components
        hname = os.path.join(CURRENT_PATH, "header.html")
        header = open(hname, "r").read()
        gr.HTML(header, visible= True, show_label=False)
        
        with gr.Row():
            
            with gr.Column():
            
                task = gr.Radio(["Auto Transcribe", "Transcribe", "Diarisation"], label="Task",
                                value= 'Auto Transcribe')
                
                num_speakers = gr.Number(value=0, label= "Number of speakers (optional)", 
                                info = "Number of speakers in the audio file. If you don't know,\
                                    leave it at 0.", visible= True)
                
                translate = gr.Checkbox(label="Translation", choices=[True, False], value = False,
                                info="Select 'Yes' to have the output translated into English.",
                                visible= True)
                
                language = gr.Dropdown(LANGUAGES,
                                label="Language (optional)", value = "None",
                                info="Language of the audio file. If you don't know,\
                                    leave it at None.", visible= True)
                
                input = gr.Radio(["Upload Audio", "Record Audio", "Upload Video","Record Video" 
                                    ,"File or Files"], label="Input Type", value="Upload Audio")
                
                audio1 = gr.Audio(source="upload", type="filepath", label="Upload Audio",
                                    interactive= True, visible= True)
                audio2 = gr.Audio(source="microphone", label="Record Audio", type="filepath",
                                    interactive= True, visible= False)
                video1 = gr.Video(source="upload", type="filepath", label="Upload Video",
                                    interactive= True, visible= False)
                video2 = gr.Video(source="webcam", label="Record Video", type="filepath",
                                    interactive= True, visible= False)
                file_in = gr.Files(label="Upload File or Files", interactive= True, visible= False)
                
                submit = gr.Button()
            
            with gr.Column():
                
                out_txt = gr.Textbox(label="Output",
                                        visible= True, show_copy_button=True)
                
                out_json = gr.JSON(label="JSON Output",
                                    visible= False, show_copy_button=True)
                
                annoation = gr.Textbox(label="Name your speaker's",
                                    info= "Please provide a list of the speakers arranged \
                                    in the order in which they appear in the input. Use comma ',' \
                                    as a seperator. Be aware that the first name is given \
                                        to SPEAKER_00 the second to SPEAKER_01 and so on.",
                                    visible= False, interactive= True)
                
                annotate = gr.Button(value="Annotate", visible= False, interactive= True)
            
        # Define usage of components
        input.change(fn=select_origin, inputs=[input],
                        outputs=[audio1, audio2, video1, video2, file_in])
        
        task.change(fn=select_task, inputs=[task],
                    outputs=[num_speakers, translate, language])
        
        translate.change(fn= lambda x : gr.update(value = x),
                            inputs=[translate], outputs=[translate])
        num_speakers.change(fn= lambda x : gr.update(value = x),
                            inputs=[num_speakers], outputs=[num_speakers])
        language.change(fn= lambda x : gr.update(value = x), 
                        inputs=[language], outputs=[language])
        
        submit.click(fn = run_scribe, 
                        inputs=[task, num_speakers, translate, language, audio1,
                                audio2, video1, video2, file_in],
                        outputs=[out_txt, out_json, annoation, annotate])
        
        annotate.click(fn = annotate_output, inputs=[annoation, out_json],
                        outputs=[out_txt, out_json])
        
    return demo

    
if __name__ == "__main__":
    
    gradio_Interface().queue().launch()
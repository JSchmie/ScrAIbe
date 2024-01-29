"""
This module contains the gradio Interface which is used to interact with the user.

The interface is themed with a soft color scheme, with primary colors of green and orange, and a neutral color of gray.

A list of languages is also defined in this module, which may be used elsewhere in the application.

Classes:
    Soft: A class from the gradio library used to theme the interface.

Variables:
    theme (gr.themes.Soft): The theme for the gradio interface.
    LANGUAGES (list of str): A list of languages supported by the application.
"""

import gradio as gr

from .interactions import *
from .stg import *

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




def gradio_Interface(layout = None,):
    """
    Creates a gradio interface for audio transcription.

    The interface includes options for the user to select the task, number of speakers, translation, language, and input type.
    It also provides options for the user to upload or record audio/video, or upload files.
    The output of the transcription is displayed in a textbox, and the JSON output in a JSON viewer.
    The user can also annotate the output by naming the speakers.

    Args:
        layout (dict, optional): A dictionary containing layout information. Defaults to None.

    Returns:
        gr.Blocks: A gradio Blocks object representing the interface.
    """
    with gr.Blocks(theme=theme,title='ScrAIbe: Automatic Audio Transcription') as demo:
            
            # Define components
            

            if layout.get('header') is not None:            
                gr.HTML(layout.get('header'), visible= True, show_label=False)
            
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
                    video2 = gr.Video(source="webcam", label="Record Video", type="filepath",include_audio= True,
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
            
            if layout.get('footer') is not None:            
                gr.HTML(layout.get('footer'), visible= True, show_label=False)
                 
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
            
            submit.click(fn = run_scraibe, 
                            inputs=[task, num_speakers, translate, language, audio1,
                                    audio2, video1, video2, file_in],
                            outputs=[out_txt, out_json, annoation, annotate])
            
            annotate.click(fn = annotate_output, inputs=[annoation, out_json],
                            outputs=[out_txt, out_json])
            
    return demo
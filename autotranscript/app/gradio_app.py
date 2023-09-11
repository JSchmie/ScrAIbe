"""
Gradio Audio Transcription App.
--------------------------------

This module provides an interface to transcribe audio files using the 
AutoTranscribe model. Users can either upload an audio file or record their speech 
live for transcription. The application supports multiple languages and provides 
options to specify the number of speakers and the language of the audio.

Attributes:
    LANGUAGES (list): A list of supported languages for transcription.

Usage:
    Run this script to start the Gradio web interface for audio transcription.
    
"""

from autotranscript import AutoTranscribe
import gradio as gr

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


def gradio_server(model : AutoTranscribe):
    """
    Sets up and launches the Gradio interface for audio transcription.

    Args:
        model (AutoTranscribe): An instance of the AutoTranscribe model for transcription.
    """
    def transcribe(audio, microphone, number_of_speakers, language):
        """
        Transcribes the provided audio input based on the given parameters.

        Args:
            audio (str): Filepath to the uploaded audio file.
            microphone (str): Filepath to the recorded audio.
            number_of_speakers (int): Number of speakers in the audio.
            language (str): Language of the audio content.

        Returns:
            tuple: Transcribed text (str), JSON output (dict)
        """
        kwargs = {}
        if number_of_speakers != 0:
            kwargs["num_speakers"] = number_of_speakers
        if language != "None":
            kwargs["language"] = language
            
        print()
        
        if audio is not None:
            out = model.transcribe(audio, **kwargs)
        elif microphone is not None:
            out = model.transcribe(microphone , **kwargs)
        else:
            out = "Please upload an audio file or record one."
        
        return str(out), out.get_json(), out.get_md()

    gr.Interface(
        fn=transcribe, 
        inputs=[
            gr.Audio(source= "upload", type="filepath", label="Upload Your Audio File",
                     interactive=True),
            gr.Audio(source= "microphone", type="filepath", label="Record Your Audio",
                     interactive=True, container= False),
            gr.Number(value=0, label= "Number of speakers (optional)", 
                      info = "Number of speakers in the audio file. If you don't know, leave it at 0."), 
            gr.Dropdown(LANGUAGES,
                        label="Language (optional)", value = "None",
                        info="Language of the audio file. If you don't know, leave it at None.")
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.JSON(label="Raw Output", container= False),
        ],
        title="Audio Transcription",
        description="Upload an audio file to transcribe its content. Powered by AutoTranscribe!",
        theme="soft",       # Example of a more modern theme
        server_port=7860,
        server_name="0.0.0.0",   
    ).queue().launch(server_port=7860, server_name="0.0.0.0") 
    
    
if __name__ == "__main__":
    
    model = AutoTranscribe()
    gradio_server(model)
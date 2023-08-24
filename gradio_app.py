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

    def transcribe(audio, microphone, number_of_speakers, language):
        kwargs = {}
        if number_of_speakers != 0:
            kwargs["num_speakers"] = number_of_speakers
        if language != "None":
            kwargs["language"] = language
        
        if audio is not None:
            out = model.transcribe(audio, **kwargs)
        elif microphone is not None:
            out = model.transcribe(microphone , **kwargs)
        else:
            out = "Please upload an audio file or record one."
        
        
        return str(out)

    gr.Interface(
        fn=transcribe, 
        inputs=[
            gr.Audio(source= "upload", type="filepath", label="Upload Your Audio File", interactive=True),
            gr.Audio(source= "microphone", type="filepath", label="Record Your Audio", interactive=True),
            gr.Number(value=0, label= "Number of speakers", 
                      info = "Number of speakers in the audio file. If you don't know, leave it at 0."), 
            # gr.Number(value=0, label= "Minimal number of speakers", 
            #           info = "Minimal number of speakers in the audio file. If you don't know or you have specified Numspeakers, leave it at 0."),
            gr.Dropdown(LANGUAGES,
                        label="Languages", default="None",
                        info="Language of the audio file. If you don't know, leave it at None.")
        ],
        outputs=[
            "text"
        ],
        title="Audio Transcription",
        thumbnail = "Logo_KIDA.png",
        description="Upload an audio file to transcribe its content. Powered by AutoTranscribe!",
        theme="soft",       # Example of a more modern theme
    ).launch(share=True)
    
    
if __name__ == "__main__":
    
    model = AutoTranscribe()
    gradio_server(model)
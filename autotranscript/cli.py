"""
Command-Line Interface (CLI) for the AutoTranscribe class,
allowing for user interaction to transcribe and diarize audio files. 
The function includes arguments for specifying the audio files, model paths,
output formats, and other options necessary for transcription.
"""
import os 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from turtle import st

from .transcriber import WHISPER_DEFAULT_PATH
from .diarisation import PYANNOTE_DEFAULT_PATH
from .autotranscript import AutoTranscribe

from whisper import available_models
from whisper.utils import get_writer
from whisper.tokenizer import LANGUAGES , TO_LANGUAGE_CODE
from torch.cuda import is_available
from torch import set_num_threads


def cli():
    """
    Command-Line Interface (CLI) for the AutoTranscribe class, allowing for user interaction to transcribe 
    and diarize audio files. The function includes arguments for specifying the audio files, model paths, 
    output formats, and other options necessary for transcription.

    This function can be executed from the command line to perform transcription tasks, providing a 
    user-friendly way to access the AutoTranscribe class functionalities.
    """
 
    def str2bool(string):
        str2val = {"True": True, "False": False}
        if string in str2val:
            return str2val[string]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    
    parser.add_argument("-f","--audio_files", nargs="+", type=str, default=None,
                        help="List of audio files to transcribe.")
    
    group.add_argument('--start_server', action='store_true',
                        help='Start the Gradio app.')
    
    parser.add_argument("--port", type=int, default= None,
                        help="Port to run the Gradio app on.")
    
    parser.add_argument("--server_name", type=str, default= "autotranscript",
                        help="Name of the Gradio app.")
                        
    parser.add_argument("--whisper_model_name", default="medium",
                        help="Name of the Whisper model to use.")

    parser.add_argument("--whisper_model_directory", type=str, default= None,
                        help="Path to save Whisper model files; defaults to ./models/whisper.")

    parser.add_argument("--diarization_directory", type=str, default= None,
                        help="Path to the diarization model directory.")

    parser.add_argument("--huggingface_token", default= None, type=str,
                        help="HuggingFace token for private model download.")

    parser.add_argument("--allow_download", type=str2bool, default=True,
                        help="Allow model download if not found locally.")

    parser.add_argument("--inference_device",
                        default="cuda" if is_available() else "cpu",
                        help="Device to use for PyTorch inference.")

    parser.add_argument("--num_threads", type=int, default=0,
                        help="Number of threads used by torch for CPU inference; overrides MKL_NUM_THREADS/OMP_NUM_THREADS.")

    parser.add_argument("--output_directory", "-o", type=str, default=".",
                        help="Directory to save the transcription outputs.")

    parser.add_argument("--output_format", "-of", type=str, default="txt",
                        choices=["txt", "json", "md", "html"],
                        help="Format of the output file; defaults to txt.")

    parser.add_argument("--verbose_output", type=str2bool, default=True,
                        help="Enable or disable progress and debug messages.")

    parser.add_argument("--task", type=str, default= None, # unifinished code
                        choices=["autoranscribe", "diarize", "autotranscribe+translate", "translate"],
                        help="Choose to perform transcription, diarization, or translation. \
                        If set to translate, the language argument must be specified.")

    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="Language spoken in the audio. Specify None to perform language detection.")

    args = parser.parse_args()
    
    arg_dict = vars(args)

    # configure output

    os.makedirs(arg_dict.pop("output_directory"), exist_ok=True)

    out_format = arg_dict.pop("output_format")
    
    # seup server arg: 
    start_server = arg_dict.pop("start_server")
    
    
    if args.num_threads > 0:
        set_num_threads(arg_dict.pop("num_threads"))

    class_kwargs = dict()
    
    for k, v in arg_dict.items():
        if v is not None:
            class_kwargs[k] = v
  


    model = AutoTranscribe(**class_kwargs)
    
    # if transcription_task == "transcribe":
    #     for audio in audio_files:
    #         out = model.transcribe(audio, language=spoken_language)
    #         basename = audio.split("/")[-1].split(".")[0]
    #         spath = f"{output_directory}/{basename}.{output_format}"
    #         out.save(spath)

    # # ... include other tasks here ...
    # elif transcription_task == "diarize":
    #     # diarize code here
    #     pass
    # elif transcription_task == "wtranscribe":
    #     # wtranscribe code here
    #     pass
    
    # if start_server: # unfinished code
    #     from .gradio_app import gradio_app
    #     gradio_app(model)

if __name__ == "__main__":
    cli()
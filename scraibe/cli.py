"""
Command-Line Interface (CLI) for the Scraibe class,
allowing for user interaction to transcribe and diarize audio files. 
The function includes arguments for specifying the audio files, model paths,
output formats, and other options necessary for transcription.
"""
import os 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

from .autotranscript import Scraibe
from .app.gradio_app import gradio_Interface

from whisper.tokenizer import LANGUAGES , TO_LANGUAGE_CODE
from torch.cuda import is_available
from torch import set_num_threads


def cli():
    """
    Command-Line Interface (CLI) for the Scraibe class, allowing for user interaction to transcribe 
    and diarize audio files. The function includes arguments for specifying the audio files, model paths, 
    output formats, and other options necessary for transcription.

    This function can be executed from the command line to perform transcription tasks, providing a 
    user-friendly way to access the Scraibe class functionalities.
    """
 
    def str2bool(string):
        str2val = {"True": True, "False": False}
        if string in str2val:
            return str2val[string]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    
    parser.add_argument("-f","--audio-files", nargs="+", type=str, default=None,
                        help="List of audio files to transcribe.")
    
    group.add_argument('--start-server', action='store_true',
                        help='Start the Gradio app.')
    
    parser.add_argument("--port", type=int, default= None,
                        help="Port to run the Gradio app on. Defaults to 7860.")
    
    parser.add_argument("--server-name", type=str, default= None,
                        help="Name of the Gradio app. If empty 127.0.0.1 or 0.0.0.0 will be used.")
                        
    parser.add_argument("--whisper-model-name", default="medium",
                        help="Name of the Whisper model to use.")

    parser.add_argument("--whisper-model-directory", type=str, default= None,
                        help="Path to save Whisper model files; defaults to ./models/whisper.")

    parser.add_argument("--diarization-directory", type=str, default= None,
                        help="Path to the diarization model directory.")

    parser.add_argument("--hf-token", default= None, type=str,
                        help="HuggingFace token for private model download.")

    parser.add_argument("--inference-device",
                        default="cuda" if is_available() else "cpu",
                        help="Device to use for PyTorch inference.")

    parser.add_argument("--num-threads", type=int, default=0,
                        help="Number of threads used by torch for CPU inference; overrides MKL_NUM_THREADS/OMP_NUM_THREADS.")

    parser.add_argument("--output-directory", "-o", type=str, default=".",
                        help="Directory to save the transcription outputs.")

    parser.add_argument("--output-format", "-of", type=str, default="txt",
                        choices=["txt", "json", "md", "html"],
                        help="Format of the output file; defaults to txt.")

    parser.add_argument("--verbose-output", type=str2bool, default=True,
                        help="Enable or disable progress and debug messages.")

    parser.add_argument("--task", type=str, default= 'autotranscribe', # unifinished code
                        choices=["autotranscribe", "diarization",
                                 "autotranscribe+translate", "translate", 'transcribe'],
                        help="Choose to perform transcription, diarization, or translation. \
                        If set to translate, the output will be translated to English.")

    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="Language spoken in the audio. Specify None to perform language detection.")

    args = parser.parse_args()
    
    arg_dict = vars(args)
    
    # configure output
    out_folder = arg_dict.pop("output_directory")
    os.makedirs(out_folder, exist_ok=True)

    out_format = arg_dict.pop("output_format")
    
    # seup server arg: 
    start_server = arg_dict.pop("start_server")
    
    task = arg_dict.pop("task")
    
    if args.num_threads > 0:
        set_num_threads(arg_dict.pop("num_threads"))
    
    class_kwargs = {'whisper_model' : arg_dict.pop("whisper_model_name"),
                    'dia_model': arg_dict.pop("diarization_directory"),
                    'use_auth_token' : arg_dict.pop("hf_token")}
    
    if arg_dict["whisper_model_directory"]:
        class_kwargs["download_root"] = arg_dict.pop("whisper_model_directory")

    model = Scraibe(**class_kwargs)

    
    if arg_dict["audio_files"]:
        audio_files = arg_dict.pop("audio_files")
        
        if task == "autotranscribe" or task == "autotranscribe+translate":
            for audio in audio_files:
                if task == "autotranscribe+translate":
                    task = "translate"
                else:
                    task = "transcribe"
                    
                out = model.autotranscribe(audio,task = task, language=arg_dict.pop("language"), verbose = arg_dict.pop("verbose_output"))
                basename = audio.split("/")[-1].split(".")[0]
                print(f'Saving {basename}.{out_format} to {out_folder}')
                out.save(os.path.join(out_folder, f"{basename}.{out_format}"))
                
        elif task == "diarization":
            for audio in audio_files:
                if arg_dict.pop("verbose_output"):
                    print(f"Verbose not implemented for diarization.")
                    
                out = model.diarization(audio)
                basename = audio.split("/")[-1].split(".")[0]
                path = os.path.join(out_folder, f"{basename}.{out_format}")
                
                print(f'Saving {basename}.{out_format} to {out_folder}')
                
                with open(path, "w") as f:
                    json.dump(json.dumps(out, indent= 1), f)

        elif task == "transcribe" or task == "translate":
            
            for audio in audio_files:
 
                out = model.transcribe(audio, task = task,
                                       language= arg_dict.pop("language"),
                                       verbose = arg_dict.pop("verbose_output"))
                basename = audio.split("/")[-1].split(".")[0]
                path = os.path.join(out_folder, f"{basename}.{out_format}")
                with open(path, "w") as f:
                    f.write(out)  
                
    
    if start_server: # unfinished code
        
        gradio_Interface(model).queue().launch(server_port=args.port, server_name=args.server_name)
        

if __name__ == "__main__":
    cli()
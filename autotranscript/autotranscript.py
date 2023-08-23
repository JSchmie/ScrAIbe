"""
AutoTranscribe Class
--------------------

This class serves as the core of the transcription system, responsible for handling
transcription and diarization of audio files. It leverages pretrained models for
speech-to-text (such as Whisper) and speaker diarization (such as pyannote.audio),
providing an accessible interface for audio processing tasks such as transcription,
speaker separation, and timestamping.

By encapsulating the complexities of underlying models, it allows for straightforward
integration into various applications, ranging from transcription services to voice assistants.

Available Classes:
- AutoTranscribe: Main class for performing transcription and diarization.
                  Includes methods for loading models, processing audio files,
                  and formatting the transcription output.

Usage:
    from .autotranscribe import AutoTranscribe

    model = AutoTranscribe(whisper_model="path/to/whisper/model", dia_model="path/to/diarisation/model")
    transcript = model.transcribe("path/to/audiofile.wav")
"""

# Standard Library Imports
import argparse
import os
from glob import iglob
from subprocess import run
from typing import TypeVar, Union
from warnings import warn

# Third-Party Imports
import torch
from numpy import ndarray
from tqdm import trange

# Application-Specific Imports
from .audio import AudioProcessor
from .diarisation import Diariser
from .transcriber import Transcriber, whisper
from .transcript_exporter import Transcript

DiarisationType = TypeVar('DiarisationType')


class AutoTranscribe:
    """
    AutoTranscribe is a class responsible for managing the transcription and diarization of audio files.
    It serves as the core of the transcription system, incorporating pretrained models
    for speech-to-text (such as Whisper) and speaker diarization (such as pyannote.audio),
    allowing for comprehensive audio processing.

    Attributes:
        transcriber (Transcriber): The transcriber object to handle transcription.
        diariser (Diariser): The diariser object to handle diarization.
    
    Methods:
        __init__: Initializes the AutoTranscribe class with appropriate models.
        transcribe: Transcribes an audio file using the whisper model and pyannote diarization model.
        remove_audio_file: Removes the original audio file to avoid disk space issues or ensure data privacy.
        get_audio_file: Gets an audio file as an AudioProcessor object.
    """
    def __init__(self,
                whisper_model: Union[bool, str, whisper] = None,
                dia_model : Union[bool, str, DiarisationType] = None,
                **kwargs) -> None:
        """Initializes the AutoTranscribe class.

        Args:
            whisper_model (Union[bool, str, whisper], optional): 
                                Path to whisper model or whisper model itself.
            diarisation_model (Union[bool, str, DiarisationType], optional): 
                                Path to pyannote diarization model or model itself.
            **kwargs: Additional keyword arguments for whisper
                        and pyannote diarization models.
        """
        
        if whisper_model is None:
            self.transcriber = Transcriber.load_model("medium")    
        elif isinstance(whisper_model, str):
            self.transcriber = Transcriber.load_model(whisper_model, **kwargs)
        else:
            self.transcriber = whisper_model

        if dia_model is None:
            self.diariser = Diariser.load_model()
        elif isinstance(dia_model, str):
            self.diariser = Diariser.load_model(dia_model, **kwargs)
        else:
            self.diariser = dia_model

        print("AutoTranscribe initialized all models successfully loaded.")
            
    def transcribe(self, audio_file : Union[str, torch.Tensor, ndarray],
                   remove_original : bool = False,
                   **kwargs) -> Transcript:
        """
        Transcribes an audio file using the whisper model and pyannote diarization model.

        Args:
            audio_file (Union[str, torch.Tensor, ndarray]): 
                            Path to audio file or a tensor representing the audio.
            remove_original (bool, optional): If True, the original audio file will
                                                be removed after transcription.
            *args: Additional positional arguments for diarization and transcription.
            **kwargs: Additional keyword arguments for diarization and transcription.

        Returns:
            Transcript: A Transcript object containing the transcription,
                        which can be exported to different formats.
        """
        
        # Get audio file as an AudioProcessor object
        audio_file = self.get_audio_file(audio_file)
        
        # Prepare waveform and sample rate for diarization
        dia_audio = {
            "waveform" : audio_file.waveform.reshape(1,len(audio_file.waveform)), 
            "sample_rate": audio_file.sr
            }
       
        print("Starting diarisation.")
        
        diarisation = self.diariser.diarization(dia_audio,
                                                *args , **kwargs)
        
        print("Diarisation finished. Starting transcription.")
        
        audio_file.sr = torch.Tensor([audio_file.sr]).to(audio_file.waveform.device)
        
        # Transcribe each segment and store the results
        final_transcript = dict()
        
        for i in trange(len(diarisation["segments"]), desc= "Transcribing"):
            
            seg = diarisation["segments"][i]
            
            audio = audio_file.cut(seg[0], seg[1])
            
            transcript = self.transcriber.transcribe(audio, *args , **kwargs)
            
            final_transcript[i] = {"speaker" : diarisation["speakers"][i],
                                   "segment" : seg,
                                   "text" : transcript}
        
        # Remove original file if needed 
        if remove_original:
            if kwargs.get("shred") is True:
                self.remove_audio_file(audio_file, shred=True)
            else:
                self.remove_audio_file(audio_file, shred=False)
            
        return Transcript(final_transcript)

    @staticmethod
    def remove_audio_file(audio_file : str,
                          shred : bool = False) -> None:
        """
        Removes the original audio file to avoid disk space issues or ensure data privacy.

        Args:
            audio_file_path (str): Path to the audio file.
            shred (bool, optional): If True, the audio file will be shredded,
                                    not just removed.
        """
        if not os.path.exists(audio_file):
            raise ValueError(f"Audiofile {audio_file} does not exist.")
        
        if shred:
            
            warn("Shredding audiofile can take a long time.", RuntimeWarning)
            
            gen = iglob(f'{audio_file}', recursive=True)
            cmd = ['shred', '-zvu', '-n', '10', f'{audio_file}']
            
            if os.path.isdir(audio_file):
                raise ValueError(f"Audiofile {audio_file} is a directory.")
            
            for file in gen:
                print(f'shredding {file} now\n')
                
                run(cmd , check=True)

        else:
            os.remove(audio_file)
            print(f"Audiofile {audio_file} removed.")
        
        
    
    @staticmethod
    def get_audio_file(audio_file : Union[str, torch.Tensor, ndarray],
                        *args, **kwargs) -> AudioProcessor:
        """Gets an audio file as TorchAudioProcessor.

        Args:
            audio_file (Union[str, torch.Tensor, ndarray]): Path to the audio file or 
                                                        a tensor representing the audio.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            AudioProcessor: An object containing the waveform and sample rate in
                            torch.Tensor format.
        """
        
        if isinstance(audio_file, str):
            audio_file = AudioProcessor.from_file(audio_file)   
        
        elif isinstance(audio_file, torch.Tensor):
            audio_file = AudioProcessor(audio_file[0], audio_file[1])
        elif isinstance(audio_file, ndarray):
            audio_file = AudioProcessor(torch.Tensor(audio_file[0]),
                                       audio_file[1])
            
        if not isinstance(audio_file, AudioProcessor):
            raise ValueError(f'Audiofile must be of type AudioProcessor,' \
                             f'not {type(audio_file)}')     
        return audio_file


def cli():
    """
    Command-Line Interface (CLI) for the AutoTranscribe class, allowing for user interaction to transcribe 
    and diarize audio files. The function includes arguments for specifying the audio files, model paths, 
    output formats, and other options necessary for transcription.

    This function can be executed from the command line to perform transcription tasks, providing a 
    user-friendly way to access the AutoTranscribe class functionalities.
    """
    from whisper import available_models
    from whisper.utils import get_writer
    from whisper.tokenizer import LANGUAGES , TO_LANGUAGE_CODE
    from .transcriber import WHISPER_DEFAULT_PATH
    from .diarisation import PYANNOTE_DEFAULT_PATH
    def str2bool(string):
        str2val = {"True": True, "False": False}
        if string in str2val:
            return str2val[string]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("audio_files", nargs="+", type=str,
                        help="List of audio files to transcribe.")

    parser.add_argument("--whisper_model_name", default="medium",
                        help="Name of the Whisper model to use.")

    parser.add_argument("--whisper_model_directory", type=str, default=WHISPER_DEFAULT_PATH,
                        help="Path to save Whisper model files; defaults to ./models/whisper.")

    parser.add_argument("--diarization_directory", type=str, default=PYANNOTE_DEFAULT_PATH,
                        help="Path to the diarization model directory.")

    parser.add_argument("--huggingface_token", default="", type=str,
                        help="HuggingFace token for private model download.")

    parser.add_argument("--allow_download", type=str2bool, default=False,
                        help="Allow model download if not found locally.")

    parser.add_argument("--inference_device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for PyTorch inference.")

    parser.add_argument("--num_threads", type=int, default=0,
                        help="Number of threads used by torch for CPU inference; overrides MKL_NUM_THREADS/OMP_NUM_THREADS.")

    parser.add_argument("--output_directory", "-o", type=str, default=".",
                        help="Directory to save the transcription outputs.")

    parser.add_argument("--output_format", "-f", type=str, default="txt",
                        choices=["txt", "json", "md", "html"],
                        help="Format of the output file; defaults to txt.")

    parser.add_argument("--verbose_output", type=str2bool, default=True,
                        help="Enable or disable progress and debug messages.")

    parser.add_argument("--transcription_task", type=str, default="transcribe",
                        choices=["transcribe", "diarize", "wtranscribe"],
                        help="Choose to perform transcription, diarization, or Whisper transcription.")

    parser.add_argument("--spoken_language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="Language spoken in the audio. Specify None to perform language detection.")

    args = parser.parse_args()

    output_directory = args.output_directory
    num_threads = args.num_threads
    whisper_model_directory = args.whisper_model_directory
    allow_download = args.allow_download
    inference_device = args.inference_device
    whisper_model_name = args.whisper_model_name
    diarization_directory = args.diarization_directory
    huggingface_token = args.huggingface_token
    transcription_task = args.transcription_task
    audio_files = args.audio_files
    spoken_language = args.spoken_language
    output_format = args.output_format

    os.makedirs(output_directory, exist_ok=True)

    if num_threads > 0:
        torch.set_num_threads(num_threads)

    whisper_kwargs = {
        "download_root": whisper_model_directory,
        "local": allow_download,
        "device": inference_device
    }

    diarisation_kwargs = {
        "local": allow_download,
        "token": huggingface_token
    }

    model = AutoTranscribe(whisper_model=whisper_model_name,
                           whisper_kwargs=whisper_kwargs,
                           dia_model=diarization_directory,
                           dia_kwargs=diarisation_kwargs)

    if transcription_task == "transcribe":
        for audio in audio_files:
            out = model.transcribe(audio, language=spoken_language)
            basename = audio.split("/")[-1].split(".")[0]
            spath = f"{output_directory}/{basename}.{output_format}"
            out.save(spath)

    # ... include other tasks here ...
    elif transcription_task == "diarize":
        # diarize code here
        pass
    elif transcription_task == "wtranscribe":
        # wtranscribe code here
        pass

if __name__ == "__main__":
    cli()
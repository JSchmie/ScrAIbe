"""
Scraibe Class
--------------------

This class serves as the core of the transcription system, responsible for handling
transcription and diarization of audio files. It leverages pretrained models for
speech-to-text (such as Whisper) and speaker diarization (such as pyannote.audio),
providing an accessible interface for audio processing tasks such as transcription,
speaker separation, and timestamping.

By encapsulating the complexities of underlying models, it allows for straightforward
integration into various applications, ranging from transcription services to voice assistants.

Available Classes:
- Scraibe: Main class for performing transcription and diarization.
                  Includes methods for loading models, processing audio files,
                  and formatting the transcription output.

Usage:
    from scraibe import Scraibe

    model = Scraibe()
    transcript = model.autotranscribe("path/to/audiofile.wav")
"""

# Standard Library Imports
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
from .transcriber import Transcriber, load_transcriber, whisper
from .transcript_exporter import Transcript
from .misc import SCRAIBE_TORCH_DEVICE


DiarisationType = TypeVar('DiarisationType')


class Scraibe:
    """
    Scraibe is a class responsible for managing the transcription and diarization of audio files.
    It serves as the core of the transcription system, incorporating pretrained models
    for speech-to-text (such as Whisper) and speaker diarization (such as pyannote.audio),
    allowing for comprehensive audio processing.

    Attributes:
        transcriber (Transcriber): The transcriber object to handle transcription.
        diariser (Diariser): The diariser object to handle diarization.

    Methods:
        __init__: Initializes the Scraibe class with appropriate models.
        transcribe: Transcribes an audio file using the whisper model and pyannote diarization model.
        remove_audio_file: Removes the original audio file to avoid disk space issues or ensure data privacy.
        get_audio_file: Gets an audio file as an AudioProcessor object.
    """

    def __init__(self,
                 whisper_model: Union[bool, str, whisper] = None,
                 whisper_type: str = "whisper",
                 dia_model: Union[bool, str, DiarisationType] = None,
                 **kwargs) -> None:
        """Initializes the Scraibe class.

        Args:
            whisper_model (Union[bool, str, whisper], optional): 
                                Path to whisper model or whisper model itself.
            whisper_type (str):
                                Type of whisper model to load. "whisper" or "faster-whisper".
            diarisation_model (Union[bool, str, DiarisationType], optional): 
                                Path to pyannote diarization model or model itself.
            **kwargs: Additional keyword arguments for whisper
                        and pyannote diarization models.
                    e.g.:

                    - verbose: If True, the class will print additional information.
                    - save_kwargs: If True, the keyword arguments will be saved
                                    for autotranscribe. So you can unload the class and reload it again.
        """

        if whisper_model is None:
            self.transcriber = load_transcriber(
                "medium", whisper_type, **kwargs)
        elif isinstance(whisper_model, str):
            self.transcriber = load_transcriber(
                whisper_model, whisper_type, **kwargs)
        else:
            self.transcriber = whisper_model

        if dia_model is None:
            self.diariser = Diariser.load_model(**kwargs)
        elif isinstance(dia_model, str):
            self.diariser = Diariser.load_model(dia_model, **kwargs)
        else:
            self.diariser: Diariser = dia_model

        if kwargs.get("verbose"):
            print("Scraibe initialized all models successfully loaded.")
            self.verbose = True
        else:
            self.verbose = False

        # Save kwargs for autotranscribe if you want to unload the class and load it again.
        if kwargs.get('save_setup'):
            self.params = dict(whisper_model=whisper_model,
                               dia_model=dia_model,
                               **kwargs)
        else:
            self.params = {}
            
        self.device = kwargs.get(
            "device", SCRAIBE_TORCH_DEVICE)

    def autotranscribe(self, audio_file: Union[str, torch.Tensor, ndarray],
                       remove_original: bool = False,
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
        if kwargs.get("verbose"):
            self.verbose = kwargs.get("verbose")
        # Get audio file as an AudioProcessor object
        audio_file: AudioProcessor = self.get_audio_file(audio_file)

        # Prepare waveform and sample rate for diarization
        dia_audio = {
            "waveform": audio_file.waveform.reshape(1, len(audio_file.waveform)).to(self.device),
            "sample_rate": audio_file.sr
        }
        
        if self.verbose:
            print("Starting diarisation.")

        diarisation = self.diariser.diarization(dia_audio, **kwargs)

        if not diarisation["segments"]:
            print("No segments found. Try to run transcription without diarisation.")

            transcript = self.transcriber.transcribe(
                audio_file.waveform, **kwargs)

            final_transcript = {0: {"speakers": 'SPEAKER_01',
                                    "segments": [0, len(audio_file.waveform)],
                                    "text": transcript}}

            return Transcript(final_transcript)

        if self.verbose:
            print("Diarisation finished. Starting transcription.")


        # Transcribe each segment and store the results
        final_transcript = dict()

        for i in trange(len(diarisation["segments"]), desc="Transcribing", disable=not self.verbose):

            seg = diarisation["segments"][i]

            audio = audio_file.cut(seg[0], seg[1])

            transcript = self.transcriber.transcribe(audio, **kwargs)

            final_transcript[i] = {"speakers": diarisation["speakers"][i],
                                   "segments": seg,
                                   "text": transcript}

        # Remove original file if needed
        if remove_original:
            if kwargs.get("shred") is True:
                self.remove_audio_file(audio_file, shred=True)
            else:
                self.remove_audio_file(audio_file, shred=False)

        return Transcript(final_transcript)

    def diarization(self, audio_file: Union[str, torch.Tensor, ndarray],
                    **kwargs) -> dict:
        """
        Perform diarization on an audio file using the pyannote diarization model.

        Args:
            audio_file (Union[str, torch.Tensor, ndarray]):
                The audio source which can either be a path to the audio file or a tensor representation.
            **kwargs: 
                Additional keyword arguments for diarization.

        Returns:
            dict: 
                A dictionary containing the results of the diarization process.
        """

        # Get audio file as an AudioProcessor object
        audio_file: AudioProcessor = self.get_audio_file(audio_file)

        # Prepare waveform and sample rate for diarization
        dia_audio = {
            "waveform": audio_file.waveform.reshape(1, len(audio_file.waveform)).to(self.device),
            "sample_rate": audio_file.sr
        }

        print("Starting diarisation.")

        diarisation = self.diariser.diarization(dia_audio, **kwargs)

        return diarisation

    def transcribe(self, audio_file: Union[str, torch.Tensor, ndarray],
                   **kwargs):
        """
            Transcribe the provided audio file.

            Args:
                audio_file (Union[str, torch.Tensor, ndarray]):
                    The audio source, which can either be a path or a tensor representation.
                **kwargs: 
                    Additional keyword arguments for transcription.

            Returns:
                str:
                    The transcribed text from the audio source.
        """
        audio_file: AudioProcessor = self.get_audio_file(audio_file)

        return self.transcriber.transcribe(audio_file.waveform, **kwargs)

    def update_transcriber(self, whisper_model: Union[str, whisper], **kwargs) -> None:
        """
        Update the transcriber model.

        Args:
            whisper_model (Union[str, whisper]):
                The new whisper model to use for transcription.
            **kwargs:
                Additional keyword arguments for the transcriber model.

            Returns:
                None
        """
        _old_model = self.transcriber.model_name

        if isinstance(whisper_model, str):
            self.transcriber = load_transcriber(whisper_model, **kwargs)
        elif isinstance(whisper_model, Transcriber):
            self.transcriber = whisper_model
        else:
            warn(
                f"Invalid model type. Please provide a valid model. Fallback to old {_old_model} Model.", RuntimeWarning)

        return None

    def update_diariser(self, dia_model: Union[str, DiarisationType], **kwargs) -> None:
        """
        Update the diariser model.

        Args:
            dia_model (Union[str, DiarisationType]):
                The new diariser model to use for diarization.
            **kwargs:
                Additional keyword arguments for the diariser model.

            Returns:
                None
        """
        if isinstance(dia_model, str):
            self.diariser = Diariser.load_model(dia_model, **kwargs)
        elif isinstance(dia_model, Diariser):
            self.diariser = dia_model
        else:
            warn("Invalid model type. Please provide a valid model. Fallback to old Model.", RuntimeWarning)

        return None

    @staticmethod
    def remove_audio_file(audio_file: str,
                          shred: bool = False) -> None:
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

                run(cmd, check=True)

        else:
            os.remove(audio_file)
            print(f"Audiofile {audio_file} removed.")

    @staticmethod
    def get_audio_file(audio_file: Union[str, torch.Tensor, ndarray]) -> AudioProcessor:
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
            raise ValueError(f'Audiofile must be of type AudioProcessor,'
                             f'not {type(audio_file)}')

        return audio_file

    def __repr__(self):
        return f"Scraibe(transcriber={self.transcriber}, diariser={self.diariser})"

"""
Transcriber Module
------------------

This module provides the Transcriber class, a comprehensive tool for working with Whisper models.
The Transcriber class offers functionalities such as loading different Whisper models, transcribing audio files,
and saving transcriptions to text files. It acts as an interface between various Whisper models and the user,
simplifying the process of audio transcription.

Main Features:
    - Loading different sizes and versions of Whisper models.
    - Transcribing audio in various formats including str, Tensor, and nparray.
    - Saving the transcriptions to the specified paths.
    - Adaptable to various language specifications.
    - Options to control the verbosity of the transcription process.
    
Constants:
    WHISPER_DEFAULT_PATH: Default path for downloading and loading Whisper models.

Usage:
    >>> from your_package import Transcriber
    >>> transcriber = Transcriber.load_model(model="medium")
    >>> transcript = transcriber.transcribe(audio="path/to/audio.wav")
    >>> transcriber.save_transcript(transcript, "path/to/save.txt")
"""

from whisper import Whisper
from whisper import load_model as whisper_load_model
from whisperx.asr import WhisperModel
from whisperx import load_model as whisperx_load_model
from typing import TypeVar, Union, Optional
from torch import Tensor, device
from torch.cuda import is_available as cuda_is_available
from numpy import ndarray
from inspect import signature
from abc import abstractmethod
import warnings

from .misc import WHISPER_DEFAULT_PATH
whisper = TypeVar('whisper')


class Transcriber:
    """
    Transcriber Class
    -----------------

    The Transcriber class serves as a wrapper around Whisper models for efficient audio
    transcription. By encapsulating the intricacies of loading models, processing audio,
    and saving transcripts, it offers an easy-to-use interface
    for users to transcribe audio files.

    Attributes:
        model (whisper): The Whisper model used for transcription.

    Methods:
        transcribe: Transcribes the given audio file.
        save_transcript: Saves the transcript to a file.
        load_model: Loads a specific Whisper model.
        _get_whisper_kwargs: Private method to get valid keyword arguments for the whisper model.

    Examples:
        >>> transcriber = Transcriber.load_model(model="medium")
        >>> transcript = transcriber.transcribe(audio="path/to/audio.wav")
        >>> transcriber.save_transcript(transcript, "path/to/save.txt")

    Note:
        The class supports various sizes and versions of Whisper models. Please refer to
        the load_model method for available options.
    """

    def __init__(self, model: whisper, model_name: str) -> None:
        """
        Initialize the Transcriber class with a Whisper model.

        Args:
            model (whisper): The Whisper model to use for transcription.
            model_name (str): The name of the model.
        """

        self.model = model

        self.model_name = model_name

    @abstractmethod
    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """
        pass

    @staticmethod
    def save_transcript(transcript: str, save_path: str) -> None:
        """
        Save a transcript to a file.

        Args:
            transcript (str): The transcript as a string.
            save_path (str): The path to save the transcript.

        Returns:
            None
        """

        with open(save_path, 'w') as f:
            f.write(transcript)

        print(f'Transcript saved to {save_path}')

    @classmethod
    @abstractmethod
    def load_model(cls,
                   model: str = "medium",
                   whisper_type: str = 'whisper',
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = None,
                   in_memory: bool = False,
                   *args, **kwargs
                   ) -> None:
        """
        Load whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'
            whisper_type (str):
                                Type of whisper model to load. "whisper" or "whisperx".
            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.
            device (Optional[Union[str, torch.device]], optional): 
                                        Device to load model on. Defaults to None.
            in_memory (bool, optional): Whether to load model in memory. 
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            None: abscract method.
        """
        pass

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        pass

    def __repr__(self) -> str:
        return f"Transcriber(model_name={self.model_name}, model={self.model})"


class WhisperTranscriber(Transcriber):
    def __init__(self, model: whisper, model_name: str) -> None:
        super().__init__(model, model_name)

    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """

        kwargs = self._get_whisper_kwargs(**kwargs)

        if not kwargs.get("verbose"):
            kwargs["verbose"] = None

        result = self.model.transcribe(audio, *args, **kwargs)
        return result["text"]

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = None,
                   in_memory: bool = False,
                   *args, **kwargs
                   ) -> 'WhisperTranscriber':
        """
        Load whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'

            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.

            device (Optional[Union[str, torch.device]], optional): 
                                        Device to load model on. Defaults to None.
            in_memory (bool, optional): Whether to load model in memory. 
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            Transcriber: A Transcriber object initialized with the specified model.
        """

        _model = whisper_load_model(model, download_root=download_root,
                                    device=device, in_memory=in_memory)

        return cls(_model, model_name=model)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        # _possible_kwargs = WhisperModel.transcribe.__code__.co_varnames
        _possible_kwargs = signature(Whisper.transcribe).parameters.keys()

        whisper_kwargs = {k: v for k,
                          v in kwargs.items() if k in _possible_kwargs}

        if (task := kwargs.get("task")):
            whisper_kwargs["task"] = task

        if (language := kwargs.get("language")):
            whisper_kwargs["language"] = language

        return whisper_kwargs

    def __repr__(self) -> str:
        return f"WhisperTranscriber(model_name={self.model_name}, model={self.model})"


class WhisperXTranscriber(Transcriber):
    def __init__(self, model: whisper, model_name: str) -> None:
        super().__init__(model, model_name)

    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """
        kwargs = self._get_whisper_kwargs(**kwargs)

        if isinstance(audio, Tensor):
            audio = audio.cpu().numpy()
        result = self.model.transcribe(audio, *args, **kwargs)
        text = ""
        for seg in result['segments']:
            text += seg['text']
        return text

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = None,
                   *args, **kwargs
                   ) -> 'WhisperXTranscriber':
        """
        Load whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'

            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.

            device (Optional[Union[str, torch.device]], optional): 
                                        Device to load model on. Defaults to None.
            in_memory (bool, optional): Whether to load model in memory. 
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            Transcriber: A Transcriber object initialized with the specified model.
        """
        if device is None:
            device = "cuda" if cuda_is_available() else "cpu"
        if not isinstance(device, str):
            device = str(device)
        compute_type = kwargs.get('compute_type', 'float16')
        if device == 'cpu' and compute_type == 'float16':
            warnings.warn(f'Compute type {compute_type} not compatible with '
                          f'device {device}! Changing compute type to int8.')
            compute_type = 'int8'
        _model = whisperx_load_model(model, download_root=download_root,
                                     device=device, compute_type=compute_type)

        return cls(_model, model_name=model)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        # _possible_kwargs = WhisperModel.transcribe.__code__.co_varnames
        _possible_kwargs = signature(WhisperModel.transcribe).parameters.keys()

        whisper_kwargs = {k: v for k,
                          v in kwargs.items() if k in _possible_kwargs}

        if (task := kwargs.get("task")):
            whisper_kwargs["task"] = task

        if (language := kwargs.get("language")):
            whisper_kwargs["language"] = language

        return whisper_kwargs

    def __repr__(self) -> str:
        return f"WhisperXTranscriber(model_name={self.model_name}, model={self.model})"


def load_transcriber(model: str = "medium",
                     whisper_type: str = 'whisper',
                     download_root: str = WHISPER_DEFAULT_PATH,
                     device: Optional[Union[str, device]] = None,
                     in_memory: bool = False,
                     *args, **kwargs
                     ) -> Union[WhisperTranscriber, WhisperXTranscriber]:
    """
    Load whisper model.

    Args:
        model (str): Whisper model. Available models include:
                    - 'tiny.en'
                    - 'tiny'
                    - 'base.en'
                    - 'base'
                    - 'small.en'
                    - 'small'
                    - 'medium.en'
                    - 'medium'
                    - 'large-v1'
                    - 'large-v2'
                    - 'large-v3'
                    - 'large'
        whisper_type (str):
                            Type of whisper model to load. "whisper" or "whisperx".
        download_root (str, optional): Path to download the model.
                                        Defaults to WHISPER_DEFAULT_PATH.
        device (Optional[Union[str, torch.device]], optional): 
                                    Device to load model on. Defaults to None.
        in_memory (bool, optional): Whether to load model in memory. 
                                    Defaults to False.
        args: Additional arguments only to avoid errors.
        kwargs: Additional keyword arguments only to avoid errors.

    Returns:
        Union[WhisperTranscriber, WhisperXTranscriber]: 
        One of the Whisper variants as Transcrbier object initialized with the specified model.
    """
    if whisper_type.lower() == 'whisper':
        _model = WhisperTranscriber.load_model(
            model, download_root, device, in_memory, *args, **kwargs)
        return _model
    elif whisper_type.lower() == 'whisperx':
        _model = WhisperXTranscriber.load_model(
            model, download_root, device, *args, **kwargs)
        return _model
    else:
        raise ValueError(f'Model type not recognized, exptected "whisper" '
                         f'or "whisperx", got {whisper_type}.')

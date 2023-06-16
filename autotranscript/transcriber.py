import os
from whisper import Whisper, load_model
from typing import TypeVar , Union
from glob import glob

whisper = TypeVar('whisper') 
Tensor = TypeVar('Tensor')
nparray = TypeVar('nparray')
Transcriber = TypeVar('Transcriber')

def get_whisper_default_path() -> str:
    """
    Get default path for whisper models

    Returns
    -------
    str
        path
    """
    _path = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(_path, "models", "whisper")

WHISPER_DEFAULT_PATH = get_whisper_default_path()

class Transcriber:
    def __init__(self, model: whisper ) -> None:
        """
        Initialize Transcriber class with a whisper model
        :param model: whisper model
        """
        self.model = model

    def transcribe(self, audio : Union[str, Tensor, nparray]  ,
                   *args, **kwargs) -> str:
        """
        transcribe audio file
        :param file: audio file to transcribe
        :param args: additional arguments
        :param kwargs: additional keyword arguments
            example:
                - language: language of the audio file    
        :return: transcript as string
        """
        
        kwargs = self._get_whisper_kwargs(**kwargs)
        
        if "verbose" not in kwargs:
            kwargs["verbose"] = False    

        result = self.model.transcribe(audio, *args, **kwargs)
        return result["text"]
    
    @staticmethod
    def save_transcript(transcript : str , save_path : str) -> None:
        """
        Save transcript to file
        :param transcript: transcript as string
        :param savepath: path to save the transcript
        :return: None
        """

        with open(save_path, 'w') as f:
            f.write(transcript)
            f.close()
            
        print(f'Transcript saved to {save_path}')

    @classmethod
    def load_model(cls,
                    model: str = "medium", 
                    local : bool = True,
                    download_root: str = WHISPER_DEFAULT_PATH) -> Transcriber:
        """
        Load whisper module

        Parameters
        ----------
        whisper : str
            whisper model
            available models:

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
                - 'large' 

        local : bool
            If true, load from local cache

        download_root : str
            Path to download the model

            default: /models/whisper
        
        Returns
        -------
        Whisper Object
        """

        if local:
            
            available_models = [os.path.basename(x) for x in 
                                glob(os.path.join(download_root, "*"))]
            
            for i, module in enumerate(available_models):
                available_models[i] = module.split(".")[0]
            
            if model not in available_models:
                raise RuntimeError("Model not found. Consider downloading the "/
                                   "model first. By deactivating the local flag, " /
                                    "the model will be downloaded automatically.")

        _model = load_model(model, download_root=download_root)

        return cls(_model)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model.
        Ensure that kwargs are valid.
        :return: kwargs for whisper model
            :rtype: dict
        """
        _possible_kwargs = Whisper.transcribe.__code__.co_varnames
        
        whisper_kwargs = dict()
        
        for k in kwargs.keys():
            if k in _possible_kwargs:
                whisper_kwargs[k] = kwargs[k]
            
        return whisper_kwargs
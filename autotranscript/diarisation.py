"""
Diarisation class.
This class is used to diarize an audio file using a pretrained model
"""
import os
from pathlib import Path
from typing import TypeVar, Union

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from torch import Tensor

from .misc import PYANNOTE_DEFAULT_CONFIG, PYANNOTE_DEFAULT_PATH
Annotation = TypeVar('Annotation') 

TOKEN_PATH = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), '.pyannotetoken')

class Diariser:
    """
    Diarisation class
    This class is used to diarize an audio file using a pretrained model
    from pyannote.audio.
    :param model: model to use for diarization
    """
    def __init__(self, model) -> None:

        self.model = model

    def diarization(self, audiofile : Union[str, Tensor, dict] ,
                    *args, **kwargs) -> Annotation:
        """
        Diarization of audio file
        :param audiofile: path to audio file or torch.Tensor
        :param args: args for diarization model 
        :param kwargs: kwargs for diarization model
        :return: diarization
        """
        kwargs = self._get_diarisation_kwargs(**kwargs)
            
        diarization = self.model(audiofile,*args, **kwargs)

        out = self.format_diarization_output(diarization)

        return out

    @staticmethod
    def format_diarization_output(dia : Annotation) -> dict:
        """
        Format diarization output to a list of tuples
        :param dia: diarization output
        :return: dict with speaker names as keys and list of tuples
                 as values and list of different speakers
        """

        dia_list  = list(dia.itertracks(yield_label=True))
        diarization_output = {"speakers": [], "segments": []}

        normalized_output = []
        index_start_speaker = 0
        index_end_speaker = 0
        current_speaker = str()
       
        ###
        # Sometimes two consecutive speakers are the same
        # This loop removes these duplicates
        ###
        
        if len(dia_list) == 1:
            normalized_output.append([0, 0, dia_list[0][2]])
        else:
            
            for i, (_, _, speaker) in enumerate(dia_list):
                if i == 0:
                    current_speaker = speaker

                if speaker != current_speaker:

                    index_end_speaker = i - 1

                    normalized_output.append([index_start_speaker,
                                            index_end_speaker,
                                            current_speaker])

                    index_start_speaker = i
                    current_speaker = speaker

                if i == len(diarization_output["speakers"]) - 1:

                    index_end_speaker = i
                    normalized_output.append([index_start_speaker, 
                                            index_end_speaker, 
                                            current_speaker])
        
        for outp in normalized_output:
            start =  dia_list[outp[0]][0].start 
            end =  dia_list[outp[1]][0].end

            diarization_output["segments"].append([start, end])
            diarization_output["speakers"].append(outp[2])
        return diarization_output
        
    @staticmethod
    def _get_token():
        """
        Get token from .pyannotetoken.txt
        :raises ValueError: No token found
        :return: Huggingface token
        :rtype: str
        """
        
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, 'r', encoding="utf-8") as file:
                token = file.read()
        else:
            raise ValueError('No token found.' \
                'Please create a token at https://huggingface.co/settings/token' \
                f'and save it in a file called {TOKEN_PATH}')
        return token

    @staticmethod
    def _save_token(token):
        """
        Save token to .pyannotetoken.txt

        :param token: Huggingface token
        :type token: str
        """
        with open(TOKEN_PATH, 'r', encoding="utf-8") as file:
            file.write(token)
    
    @classmethod
    def load_model(cls, 
                    model: str = PYANNOTE_DEFAULT_CONFIG, 
                    token: str = None,
                    cache_token: bool = False,
                    cache_dir: Union[Path, str] = PYANNOTE_DEFAULT_PATH,
                    hparams_file: Union[str, Path] = None
                    ) -> Pipeline:
        """
        Load modules from pyannote

        Parameters
        ----------
        model : str
            pyannote model 
            default: /models/pyannote/speaker_diarization/config.yaml
        token : str
            HUGGINGFACE_TOKEN
        local : bool
            If true, load from local cache
        
        Returns
        -------
        Pipeline Object
        """
        
        if cache_token and token is not None:
            cls._save_token(token)
            
        if not os.path.exists(model) and token is None:
            token = cls._get_token()
            model = 'pyannote/speaker-diarization'
                
        _model =  Pipeline.from_pretrained(model,
                                           use_auth_token = token,
                                           cache_dir = cache_dir,
                                           hparams_file = hparams_file,)
        
        if model is None:
            raise ValueError('Unable to load model either from local cache' \
                'or from huggingface.co models. Please check your token' \
                'or your local model path')
        return cls(_model)

    @staticmethod
    def _get_diarisation_kwargs(**kwargs) -> dict:
        """
        Get kwargs for pyannote diarization model
        Ensure that kwargs are valid
        :return: kwargs for pyannote diarization model
            :rtype: dict
        """
        _possible_kwargs = SpeakerDiarization.apply.__code__.co_varnames
        
        diarisation_kwargs = dict()
        
        for k in kwargs.keys():
            if k in _possible_kwargs:
               diarisation_kwargs[k] = kwargs[k]
            
        return diarisation_kwargs
    
    def __repr__(self):
        return f"Diarisation(model={self.model})"
    
    def __str__(self):
        return f"Diarisation(model={self.model})"

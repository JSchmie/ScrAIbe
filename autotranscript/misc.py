
from pyannote.audio import Pipeline
from whisper import Whisper, load_model
import os
import glob
from warnings import warn
import yaml

WHISPER_DEFAULT_PATH = os.path.join(os.path.dirname(__file__),
                                     "models", "whisper")

PYANNOTE_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 
                                     "models", "pyannote", 
                                     "speaker_diarization", "config.yaml")


def config_diarization_yaml(file):
    """
    Configure diarization pipeline from yaml file to use the model offline
    and avoid manuel file manipulation.
    
    :param file: yaml file
    :type file: yaml
    """
    
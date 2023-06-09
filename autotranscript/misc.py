
from pyannote.audio import Pipeline
from whisper import Whisper, load_model
import os
import glob

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

def load_whisper_model(model: str ="medium", local : bool = False, download_root: str = WHISPER_DEFAULT_PATH) -> Whisper:
    """
    Load modules from whisper

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
        available_models = [os.path.basename(x) for x in glob.glob(os.path.join(WHISPER_DEFAULT_PATH, "*"))]
        
        for i, module in enumerate(available_models):
            available_models[i] = module.split(".")[0]
        
        if model not in available_models:
            raise RuntimeError("Model not found. Consider downloading the model first. By deactivating the local flag, the model will be downloaded automatically.")

    return load_model(model, download_root=WHISPER_DEFAULT_PATH)

def load_pyannote_model(model: str, token: str = "", local : bool = True) -> Pipeline:
    """
    Load modules from pyannote

    Parameters
    ----------
    model : str
        pyannote model 
    token : str
        HUGGINGFACE_TOKEN
    local : bool
        If true, load from local cache
    
    Returns
    -------
    Pipeline Object
    """

    if local:
        return Pipeline.from_pretrained(model)
    else:
        return Pipeline.from_pretrained(model, use_auth_token = token)

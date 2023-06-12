
from pyannote.audio import Pipeline
from whisper import Whisper, load_model
import os
import glob
from warnings import warn

WHISPER_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "models", "whisper")

PYANNOTE_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", "pyannote", 
                                     "speaker_diarization", "config.yaml")

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
    warn("load_whisper_model is deprecated. Use Transcriptor.load_model() instead.", DeprecationWarning)
    if local:
        available_models = [os.path.basename(x) for x in glob.glob(os.path.join(download_root, "*"))]
        
        for i, module in enumerate(available_models):
            available_models[i] = module.split(".")[0]
        
        if model not in available_models:
            raise RuntimeError("Model not found. Consider downloading the model first. By deactivating the local flag, the model will be downloaded automatically.")

    return load_model(model, download_root=download_root)

def load_pyannote_model(model: str = PYANNOTE_DEFAULT_PATH, 
                        token: str = "",
                        local : bool = True,
                        *args, **kwargs) -> Pipeline:
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
    warn("load_pyannote_model is deprecated. Use Diarisation.load_model() instead.", DeprecationWarning)
    if local:
        return Pipeline.from_pretrained(model,*args, **kwargs)
    else:
        return Pipeline.from_pretrained(model, use_auth_token = token, *args, **kwargs)

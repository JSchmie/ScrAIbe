from pyannote.audio import Pipeline
from whisper import Whisper, load_model
import os
import glob
from warnings import warn
import yaml

CACHE_DIR = os.getenv(
    "AUTOT_CACHE",
    os.path.expanduser("~/.cache/torch/models"),
)

WHISPER_DEFAULT_PATH = os.path.join(CACHE_DIR, "whisper")

PYANNOTE_DEFAULT_PATH = os.path.join(CACHE_DIR, "pyannote")

PYANNOTE_DEFAULT_CONFIG = os.path.join(PYANNOTE_DEFAULT_PATH, "config.yaml")

def config_diarization_yaml(file, path_to_segmentation = None):
    """
    Configure diarization pipeline from yaml file to use the model offline
    and avoid manuel file manipulation.
    
    :param file: yaml file
    :type file: yaml
    """
    with open(file, "r") as stream:
            yml = yaml.safe_load(stream)
            stream.close()
    if path_to_segmentation:
        yml["pipeline"]["params"]["segmentation"] = path_to_segmentation
    else:
        yml["pipeline"]["params"]["segmentation"] = os.path.join(PYANNOTE_DEFAULT_PATH, "pytorch_model.bin")
                                                 
    # if path_to_embedding:
    #     yml["pipeline"]["params"]["embedding"] = path_to_embedding
    # else:
    #     yml["pipeline"]["params"]["embedding"] = os.path.relpath(
    #                                                         os.path.join(
    #                                                         os.path.dirname(__file__),
    #                                                         "models", "pyannote",
    #                                                         "speechbrain",
    #                                                         "spkrec-ecapa-voxceleb",
    #                                                         "embedding_model.ckpt"))
    
    if not os.path.exists(yml["pipeline"]["params"]["segmentation"]):
        raise FileNotFoundError(f"Segmentation model not found at {yml['pipeline']['params']['segmentation']}")
    
    # if not os.path.exists(yml["pipeline"]["params"]["embedding"]):
    #     raise FileNotFoundError(f"Embedding model not found at {yml['pipeline']['params']['embedding']}")
    
    with open(file, "w") as stream:
        yaml.dump(yml, stream)
        stream.close()                               

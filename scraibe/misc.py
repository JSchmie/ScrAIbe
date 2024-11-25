import os
import yaml
from argparse import Action
from ast import literal_eval
from torch.cuda import is_available
from torch import get_num_threads, set_num_threads

CACHE_DIR = os.getenv(
    "AUTOT_CACHE",
    os.path.expanduser("~/.cache/torch/models"),
)
os.environ["PYANNOTE_CACHE"] = os.getenv(
    "PYANNOTE_CACHE",
    os.path.join(CACHE_DIR, "pyannote"),
)

WHISPER_DEFAULT_PATH = os.path.join(CACHE_DIR, "whisper")
PYANNOTE_DEFAULT_PATH = os.path.join(CACHE_DIR, "pyannote")
PYANNOTE_DEFAULT_CONFIG = os.path.join(PYANNOTE_DEFAULT_PATH, "config.yaml") \
    if os.path.exists(os.path.join(PYANNOTE_DEFAULT_PATH, "config.yaml")) \
    else ('Jaikinator/ScrAIbe', 'pyannote/speaker-diarization-3.1')

SCRAIBE_TORCH_DEVICE =  os.getenv("SCRAIBE_TORCH_DEVICE", "cuda" if is_available() else "cpu")

SCRAIBE_NUM_THREADS = os.getenv("SCRAIBE_NUM_THREADS", min(8, get_num_threads()))

def config_diarization_yaml(file_path: str, path_to_segmentation: str = None) -> None:
    """Configure diarization pipeline from a YAML file.

    This function updates the YAML file to use the given segmentation model
    offline, and avoids manual file manipulation.

    Args:
        file_path (str): Path to the YAML file.
        path_to_segmentation (str, optional): Optional path to the segmentation model.

    Raises:
        FileNotFoundError: If the segmentation model file is not found.
    """
    with open(file_path, "r") as stream:
        yml = yaml.safe_load(stream)

    segmentation_path = path_to_segmentation or os.path.join(
        PYANNOTE_DEFAULT_PATH, "pytorch_model.bin")
    yml["pipeline"]["params"]["segmentation"] = segmentation_path

    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(
            f"Segmentation model not found at {segmentation_path}")

    with open(file_path, "w") as stream:
        yaml.dump(yml, stream)


def set_threads(parse_threads=None,
                yaml_threads=None):
    global SCRAIBE_NUM_THREADS
    if parse_threads is not None:
        if not isinstance(parse_threads, int):
            # probably covered with int type of parser arg
            raise ValueError(f"Type of --num-threads must be int, but the type is {type(parse_threads)}")
        elif parse_threads < 1:
            raise ValueError(f"Number of threads must be a positive integer, {parse_threads} was given")
        else:
            set_num_threads(parse_threads)
            SCRAIBE_NUM_THREADS = parse_threads
    elif yaml_threads is not None:
        if not isinstance(yaml_threads, int):
            raise ValueError(f"Type of num_threads must be int, but the type is {type(yaml_threads)}")
        elif yaml_threads < 1:
            raise ValueError(f"Number of threads must be a positive integer, {yaml_threads} was given")
        else:
            set_num_threads(yaml_threads)
            SCRAIBE_NUM_THREADS = yaml_threads

class ParseKwargs(Action):
    """
    Custom argparse action to parse keyword arguments.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            try:
                value = literal_eval(value)
            except:
                pass
            getattr(namespace, self.dest)[key] = value

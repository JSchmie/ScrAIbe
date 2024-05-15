import os
import yaml
from pyannote.audio.core.model import CACHE_DIR as PYANNOTE_CACHE_DIR
from argparse import Action
from ast import literal_eval

CACHE_DIR = os.getenv(
    "AUTOT_CACHE",
    os.path.expanduser("~/.cache/torch/models"),
)

if CACHE_DIR != PYANNOTE_CACHE_DIR:
    os.environ["PYANNOTE_CACHE"] = os.path.join(CACHE_DIR, "pyannote")

WHISPER_DEFAULT_PATH = os.path.join(CACHE_DIR, "whisper")
PYANNOTE_DEFAULT_PATH = os.path.join(CACHE_DIR, "pyannote")
PYANNOTE_DEFAULT_CONFIG = os.path.join(PYANNOTE_DEFAULT_PATH, "config.yaml") \
    if os.path.exists(os.path.join(PYANNOTE_DEFAULT_PATH, "config.yaml")) \
    else ('jaikinator/scraibe', 'pyannote/speaker-diarization-3.1')


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

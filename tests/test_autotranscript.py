import pytest
from scraibe import Transcriber
from unittest.mock import patch, mock_open
import os

def test_load_pyannote_model():
    """
    Test load_pyannote_test
    """
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("models/pyannote/speaker_diarization/config.yaml")
    assert isinstance(pipeline, SpeakerDiarization)

# Test Transcribtion class


@pytest.fixture
def transcriber():
    """
    Prepare Transcriber for testing
    Returns: Transcriber Object
    """

    return Transcriber.load_model("medium", local=True)


def test_Transcriber_init(transcriber):
    """
    Test Transcriber initialization with a whisper model 
    """
    
    assert isinstance(transcriber, Transcriber)

def test_transcription(transcriber):
    """
    Test transcription
    """

    transcript = transcriber.transcribe("tests/test.wav") 
    assert isinstance(transcript, str)
    
def test_save_transcript_to_file(transcriber):
    """
    Test save_transcript_to_file
    """
    transcript = transcriber.transcribe("tests/test.wav")

    Transcriber.save_transcript(transcript, "tests/output.txt")
    
    assert os.path.exists("tests/output.txt")

    os.remove("tests/output.txt")
    
# Test Diaraization class

from scraibe import Diariser

@pytest.fixture
def diarisation():
    """
    Prepare Diarisation for testing
    Returns: Diarisation Object
    """

    return Diariser.load_model("models/pyannote/speaker_diarization/config.yaml", local=True)

def test_Diarisation_init(diarisation):
    """
    Test Diarisation initialization with a pyannote model 
    """
    
    assert isinstance(diarisation, Diariser)

def test_diarisation(diarisation):
    """
    Test diarisation
    """

    diarisation = diarisation.diarization("tests/test.wav") 
    assert isinstance(diarisation, dict)

# Test AudioProcessor

from scraibe import AudioProcessor , TorchAudioProcessor


def test_AudioProcessor_init():
    """
    Test AudioProcessor initialization
    """
    audio = AudioProcessor("tests/test.wav")
    assert isinstance(audio, AudioProcessor)

def test_AudioProcessor_convert():
    """
    Test AudioProcessor convert
    """
    audio = AudioProcessor("tests/test.wav")
    audio.convert_audio("tests/test.mp3", format="mp3")
    assert os.path.exists("tests/test.mp3")
    
def test_TorchAudioProcessor_from_file():
    """
    Test TorchAudioProcessor initialization
    """
    audio = TorchAudioProcessor.from_file("tests/test.wav")
    
    assert isinstance(audio, TorchAudioProcessor)
    
    os.remove("tests/test.mp3")


def test_TorchAudioProcessor_from_ffmpeg():
    """
    Test TorchAudioProcessor initialization
    """
    audio = TorchAudioProcessor.from_ffmpeg("tests/test.wav")
    assert isinstance(audio, TorchAudioProcessor)

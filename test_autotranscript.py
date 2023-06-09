import pytest
from autotranscript import Transcriber
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

    return Transcriber.load_whisper_model("medium", local=True)


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
    
    open_mock = mock_open()
    with patch("autotranscript.Transcriber.save_transcript", open_mock, create=True):
        Transcriber.save_transcript(transcript, "output.txt")

    open_mock.assert_called_with("output.txt", "w")
    open_mock.return_value.write.assert_called_once_with("test-data")

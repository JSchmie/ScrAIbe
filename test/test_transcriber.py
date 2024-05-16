import pytest
from scraibe import Transcriber, WhisperTranscriber, WhisperXTranscriber
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_WAVEFORM = "Hello World"

"""
@pytest.mark.parametrize("audio_file, expected_transcription",[("path_to_test_audiofile", "test_transcription")] )
@patch("scraibe.Transcriber.load_model")

def test_transcriber(mock_load_model, audio_file, expected_transcription):


    Args:
        mock_load_model (_type_): _description_
        audio_file (_type_): _description_
        expected_transcription (_type_): _description_

    mock_model = mock_load_model.return_value
    mock_model.transcribe.return_value ={"text": expected_transcription}

    transcriber = Transcriber.load_model(model="medium")

    transcription_result = transcriber.transcribe(audio=audio_file)

    assert transcription_result == expected_transcription """


@pytest.fixture
def whisper_instance():
    return Transcriber.load_model('medium', whisper_type='whisper')


@pytest.fixture
def whisperx_instance():
    return Transcriber.load_model('medium', whisper_type='whisperx')


def test_whisper_base_initialization(whisper_instance):
    assert isinstance(whisper_instance, Transcriber)


def test_whisperx_base_initialization(whisperx_instance):
    assert isinstance(whisperx_instance, Transcriber)


def test_whisper_transcriber_initialization(whisper_instance):
    assert isinstance(whisper_instance, WhisperTranscriber)


def test_whisperx_transcriber_initialization(whisperx_instance):
    assert isinstance(whisperx_instance, WhisperXTranscriber)


def test_wrong_transcriber_initialization():
    with pytest.raises(ValueError):
        Transcriber.load_model('medium', whisper_type='wrong_whisper')


def test_get_whisper_kwargs():
    kwargs = {"arg1": 1, "arg3": 3}
    valid_kwargs = Transcriber._get_whisper_kwargs(**kwargs)
    assert not valid_kwargs == {"arg1": 1, "arg3": 3}


def test_whisper_transcribe(whisper_instance):
    model = whisper_instance
    # mocker.patch.object(transcriber_instance.model, 'transcribe', return_value={'Hello, World !'} )
    transcript = model.transcribe('test/audio_test_2.mp4')
    assert isinstance(transcript, str)


def test_whisperx_transcribe(whisperx_instance):
    model = whisperx_instance
    # mocker.patch.object(transcriber_instance.model, 'transcribe', return_value={'Hello, World !'} )
    transcript = model.transcribe('test/audio_test_2.mp4')
    assert isinstance(transcript, str)
